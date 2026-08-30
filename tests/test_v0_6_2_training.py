import json
import os

import pytest
import torch

from pixelworld import cli
from pixelworld.evaluation import METRIC_NAMES
from pixelworld.training import seed_everything
from pixelworld.config import DEFAULT_EVALUATION_SEEDS
from pixelworld.versions.v0_6_2.config import SHARED_TARGET_SHA256, PlacementConfig
from pixelworld.versions.v0_6_2.model import create_model
from pixelworld.versions.v0_6_2.study import (
    _validate_completed_v062_run,
    aggregate,
    analysis_cache_is_compatible,
    baseline_run_is_compatible,
    load_analysis_cache,
    require_clean_study_repository,
    require_study_commit,
    run_study,
)
from pixelworld.versions.v0_6_2.training import (
    PlacementRunStore,
    compute_losses,
    gradient_conflict_backward,
    initialize_training,
    load_checkpoint,
    project_conflicting_gradients,
    run_training,
    train_one_epoch,
)


def config(variant, epochs=1):
    return PlacementConfig(
        variant=variant,
        samples=8,
        batch_size=4,
        epochs=epochs,
        seed=42,
        offset_radius=8,
        evaluation_seeds=(500000,),
    )


def gradient_config(mode, epochs=1):
    return PlacementConfig(
        variant="D", gradient_mode=mode, samples=8, batch_size=4, epochs=epochs,
        seed=42, offset_radius=8, evaluation_seeds=(500000,),
    )


def model_state(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


@pytest.mark.parametrize(
    ("variant", "output_count"), [("B", 9), ("C", 10), ("D", 10), ("E", 11)]
)
def test_variant_tensor_shapes(variant, output_count):
    seed_everything(42)
    outputs = create_model(variant)(torch.zeros(2, 81))
    assert len(outputs) == output_count
    if variant in ("C", "D", "E"):
        assert outputs[9].shape == (2, 8, 2)
        assert torch.all(outputs[9].abs() <= 1)
    if variant == "E":
        assert outputs[10].shape == (2, 8, 2)


@pytest.mark.parametrize("variant", ["B", "C", "D", "E"])
def test_all_variant_losses_are_finite(variant):
    objects = initialize_training(config(variant), "cpu", progress=None)
    losses = compute_losses(
        objects.model,
        next(iter(objects.loader)),
        config(variant),
        "cpu",
        objects.coord_values,
        objects.ce,
        objects.bce,
    )
    assert all(torch.isfinite(value) for value in losses.values())


def gradient_sum(module):
    return sum(
        parameter.grad.abs().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_variant_c_detaches_offset_from_placement_slots():
    model = create_model("C")
    model(torch.zeros(2, 81))[9].sum().backward()
    assert gradient_sum(model.offset_head) > 0
    assert gradient_sum(model.placement_encoder) == 0
    assert gradient_sum(model.placement_decoder) == 0


def test_variant_d_shares_offset_gradient_with_placement_path():
    model = create_model("D")
    model(torch.zeros(2, 81))[9].sum().backward()
    assert gradient_sum(model.offset_head) > 0
    assert gradient_sum(model.placement_encoder) > 0
    assert gradient_sum(model.placement_decoder) > 0


def test_measure_mode_is_bit_exact_to_standard_d():
    standard = initialize_training(gradient_config("standard"), "cpu", progress=None)
    train_one_epoch(standard, gradient_config("standard"), "cpu")
    measured = initialize_training(gradient_config("measure"), "cpu", progress=None)
    record = train_one_epoch(measured, gradient_config("measure"), "cpu")
    assert all(torch.equal(model_state(standard.model)[name], tensor)
               for name, tensor in model_state(measured.model).items())
    assert "gradient_cosine_mean" in record


def test_positive_gradient_is_not_projected():
    gd = [torch.tensor([1.0, 0.0])]
    go = [torch.tensor([2.0, 1.0])]
    projected, stats = project_conflicting_gradients(gd, go, project=True)
    assert torch.equal(projected[0], go[0])
    assert not stats["projected"]


def test_negative_gradient_is_projected_orthogonally():
    gd = [torch.tensor([1.0, 0.0])]
    go = [torch.tensor([-2.0, 1.0])]
    projected, stats = project_conflicting_gradients(gd, go, project=True)
    assert stats["projected"]
    assert torch.dot(gd[0], projected[0]).abs() < 1e-6
    assert abs(stats["post_cosine"]) < 1e-6


def test_projection_handles_none_and_zero_norm():
    projected, stats = project_conflicting_gradients(
        [None, torch.zeros(2)], [torch.ones(2), torch.ones(2)], project=True
    )
    assert torch.equal(projected[0], torch.ones(2))
    assert torch.equal(projected[1], torch.ones(2))
    assert stats["discrete_norm"] == 0.0
    assert not stats["projected"]


def test_pcgrad_preserves_head_and_foreign_path_gradients():
    cfg = gradient_config("pcgrad")
    left = initialize_training(cfg, "cpu", progress=None)
    batch = next(iter(left.loader))
    losses = compute_losses(left.model, batch, cfg, "cpu", left.coord_values, left.ce, left.bce)
    left.optimizer.zero_grad(); losses["loss"].backward()
    expected = {name: p.grad.detach().clone() for name, p in left.model.named_parameters() if p.grad is not None}
    right = initialize_training(cfg, "cpu", progress=None)
    losses = compute_losses(right.model, batch, cfg, "cpu", right.coord_values, right.ce, right.bce)
    right.optimizer.zero_grad(); gradient_conflict_backward(right.model, losses, project=True)
    protected = ("offset_head", "region_head", "anchor_head", "terrain_", "presence_", "class_head", "action_head", "trigger_head", "attribute_")
    for name, parameter in right.model.named_parameters():
        if name.startswith(protected) and parameter.grad is not None:
            assert torch.equal(parameter.grad, expected[name]), name


def test_one_optimizer_step_per_batch(monkeypatch):
    objects = initialize_training(gradient_config("pcgrad"), "cpu", progress=None)
    calls = 0
    original = objects.optimizer.step
    def counted_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(objects.optimizer, "step", counted_step)
    record = train_one_epoch(objects, gradient_config("pcgrad"), "cpu")
    assert calls == record["batches"]


def test_variant_e_auxiliary_head_updates_placement_path():
    model = create_model("E")
    model(torch.zeros(2, 81))[10].sum().backward()
    assert gradient_sum(model.xy_auxiliary_head) > 0
    assert gradient_sum(model.placement_encoder) > 0
    assert gradient_sum(model.placement_decoder) > 0


def test_checkpoint_reload_and_incompatible_variant(tmp_path):
    store = PlacementRunStore.create(tmp_path, config("C"), "checkpoint-c")
    run_training(store, device="cpu")
    model, payload = load_checkpoint(store.path / "final.pt", config("C"), "cpu")
    assert payload["variant"] == "C"
    assert payload["slot_latent_dim"] == 8
    assert payload["layout_dim"] == 71
    assert next(model.parameters()).device.type == "cpu"
    assert payload["shared_target_sha256"] == SHARED_TARGET_SHA256
    assert payload["evaluation_seeds"] == [500000]
    assert payload["provenance"]["placement_config"] == config("C").to_dict()
    assert payload["provenance"]["model_parameters"] > 0
    summary = json.loads(store.artifact_path("run_summary.json").read_text(encoding="utf-8"))
    for key in (
        "git_commit",
        "git_dirty",
        "git_branch",
        "python_version",
        "torch_version",
        "cuda_runtime",
        "cuda_available",
        "device",
        "gpu_model",
        "model_parameters",
    ):
        assert key in summary["provenance"]
    assert summary["shared_target_sha256"] == SHARED_TARGET_SHA256
    assert summary["evaluation_seeds"] == [500000]
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(store.path / "final.pt", config("D"), "cpu")


def test_gradient_mode_checkpoint_reload_and_incompatibility(tmp_path):
    cfg = gradient_config("measure")
    store = PlacementRunStore.create(tmp_path, cfg, "checkpoint-d-measure")
    run_training(store, device="cpu")
    _, payload = load_checkpoint(store.path / "final.pt", cfg, "cpu")
    assert payload["gradient_mode"] == "measure"
    assert "gradient_cosine_mean" in payload["training_history"][0]
    with pytest.raises(ValueError, match="gradient mode"):
        load_checkpoint(store.path / "final.pt", gradient_config("pcgrad"), "cpu")


def test_gradient_mode_resume_parity_including_statistics(tmp_path):
    cfg = gradient_config("pcgrad", epochs=2)
    continuous = PlacementRunStore.create(tmp_path, cfg, "continuous-pcgrad")
    run_training(continuous, device="cpu")
    resumed = PlacementRunStore.create(tmp_path, cfg, "resumed-pcgrad")
    run_training(resumed, device="cpu", stop_after_epoch=1)
    run_training(resumed, device="cpu", resume=True)
    left = torch.load(continuous.path / "final.pt", weights_only=True)
    right = torch.load(resumed.path / "final.pt", weights_only=True)
    assert all(torch.equal(left["model_state_dict"][name], tensor)
               for name, tensor in right["model_state_dict"].items())
    for a, b in zip(left["training_history"], right["training_history"]):
        assert {k:v for k,v in a.items() if k != "epoch_seconds"} == {
            k:v for k,v in b.items() if k != "epoch_seconds"
        }


def test_checkpoint_rejects_old_latent_schema(tmp_path):
    store = PlacementRunStore.create(tmp_path, config("C"), "checkpoint-schema")
    run_training(store, device="cpu")
    payload = torch.load(store.path / "final.pt", weights_only=True)
    payload["slot_latent_dim"] = 6
    path = tmp_path / "old-schema.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="latent schema"):
        load_checkpoint(path, config("C"), "cpu")


def test_resume_parity(tmp_path):
    continuous = PlacementRunStore.create(tmp_path, config("C", epochs=2), "continuous-c")
    run_training(continuous, device="cpu")
    resumed = PlacementRunStore.create(tmp_path, config("C", epochs=2), "resumed-c")
    run_training(resumed, device="cpu", stop_after_epoch=1)
    run_training(resumed, device="cpu", resume=True)
    left = torch.load(continuous.path / "final.pt", weights_only=True)
    right = torch.load(resumed.path / "final.pt", weights_only=True)
    assert all(
        torch.equal(left["model_state_dict"][name], tensor)
        for name, tensor in right["model_state_dict"].items()
    )
    for left_epoch, right_epoch in zip(left["training_history"], right["training_history"]):
        assert {k: v for k, v in left_epoch.items() if k != "epoch_seconds"} == {
            k: v for k, v in right_epoch.items() if k != "epoch_seconds"
        }


def test_cli_variant_selection_and_invalid_variant():
    parser = cli.build_parser()
    args = parser.parse_args(["train", "--version", "0.6.2", "--variant", "D"])
    assert args.variant == "D"
    assert args.offset_radius == 8
    args = parser.parse_args(["train", "--version", "0.6.2", "--variant", "D", "--gradient-mode", "pcgrad"])
    assert args.gradient_mode == "pcgrad"
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--version", "0.6.2", "--variant", "Z"])


def test_gradient_modes_restricted_to_d():
    with pytest.raises(ValueError, match="only for variant D"):
        PlacementConfig(variant="C", gradient_mode="measure").validate()


def fake_record(variant, seed, position, interaction):
    metrics = {name: 1.0 for name in METRIC_NAMES}
    metrics.update({"position": position, "interaction": interaction})
    diagnostics = {
        "end_to_end_position_mae": position,
        "end_to_end_interaction_iou": interaction,
        "offset_clipping_rate": 0.0,
        "invalid_placements": 0,
        "water_placements": 0,
        "collisions": 0,
    }
    return {"variant": variant, "seed": seed, "metrics": metrics, "diagnostics": diagnostics}


def test_study_aggregation(tmp_path):
    records = [fake_record("A", 42, 6.0, 0.45), fake_record("B", 42, 5.0, 0.50)]
    summary = aggregate(
        records,
        tmp_path,
        {"selected_radius": 8},
        {"passed": True},
    )
    assert summary["recommended_variant"] == "B"
    for name in (
        "metrics_by_seed.csv",
        "metrics_statistics.csv",
        "seed_matched_benchmark_deltas.csv",
        "placement_diagnostics.json",
        "study_summary.json",
        "recommendation.md",
    ):
        assert (tmp_path / name).is_file()
    assert "not paired target-world" in summary["comparison_scope"]


def _directory_link(target, link):
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"directory links unavailable: {error}")
        import _winapi

        try:
            _winapi.CreateJunction(str(target), str(link))
        except OSError as junction_error:
            pytest.skip(f"Windows junctions unavailable: {junction_error}")


def test_placement_run_store_normal_id_and_traversal(tmp_path):
    store = PlacementRunStore.create(tmp_path, config("B"), "normal-run_1")
    assert store.secure_path().parent == store.root
    for run_id in ("../escape", "a/b", "a\\b", "trailing.", "trailing ", "CON.txt"):
        with pytest.raises(ValueError):
            PlacementRunStore(tmp_path, run_id)


def test_placement_run_store_rejects_run_symlink_or_junction_escape(tmp_path):
    runs = tmp_path / "outputs" / "studies" / "0.6.2-placement" / "runs"
    outside = tmp_path / "outside"
    runs.mkdir(parents=True)
    outside.mkdir()
    _directory_link(outside, runs / "escaped")
    with pytest.raises(ValueError, match="escapes"):
        PlacementRunStore.open(tmp_path, "escaped")


def test_placement_run_store_revalidates_before_every_artifact_access(tmp_path):
    store = PlacementRunStore.create(tmp_path, config("B"), "swapped")
    outside = tmp_path / "outside-run"
    store.path.rename(outside)
    _directory_link(outside, store.path)
    with pytest.raises(ValueError, match="escapes"):
        store.status("running")
    with pytest.raises(ValueError, match="escapes"):
        store.log("must not escape")


def test_analysis_cache_rejects_wrong_digest(tmp_path):
    candidate = {
        "analysis_kind": "explicit_eight_latent_region_relative_anchors",
        "analysis_schema_version": 1,
        "generator_target_version": "0.6.2-explicit-8-latent-region-relative-v1",
        "samples": 14_000,
        "slot_latent_dim": 8,
        "layout_dim": 71,
        "condition_dim": 81,
        "local_offset_pixels": 8,
        "shared_target_sha256": SHARED_TARGET_SHA256,
    }
    assert analysis_cache_is_compatible(candidate)
    candidate["shared_target_sha256"] = "0" * 64
    assert not analysis_cache_is_compatible(candidate)
    path = tmp_path / "offset_analysis.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    assert load_analysis_cache(path) is None


def test_study_dirty_and_commit_mismatch_abort(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "pixelworld.versions.v0_6_2.study.git_provenance",
        lambda _root: {"git_commit": "a" * 40, "git_dirty": True, "git_branch": "branch"},
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        require_clean_study_repository(tmp_path)
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        run_study(tmp_path, seeds=(42,), variants=("B",), samples=1, epochs=1, device="cpu")
    assert not (tmp_path / "outputs").exists()
    monkeypatch.setattr(
        "pixelworld.versions.v0_6_2.study.git_provenance",
        lambda _root: {"git_commit": "b" * 40, "git_dirty": False, "git_branch": "branch"},
    )
    with pytest.raises(RuntimeError, match="commit changed"):
        require_study_commit(tmp_path, "a" * 40)


def test_completed_run_reuse_requires_matching_commit(monkeypatch, tmp_path):
    commit = "a" * 40
    monkeypatch.setattr(
        "pixelworld.versions.v0_6_2.training.environment_provenance",
        lambda _root, device, model_parameters: {
            "git_commit": commit,
            "git_dirty": False,
            "git_branch": "test-branch",
            "python_version": "test",
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu_model": None,
            "model_parameters": model_parameters,
        },
    )
    selected = config("B")
    store = PlacementRunStore.create(tmp_path, selected, "reuse-check")
    run_training(store, device="cpu")
    _validate_completed_v062_run(store, selected, commit, SHARED_TARGET_SHA256)
    with pytest.raises(ValueError, match="provenance"):
        _validate_completed_v062_run(
            store, selected, "b" * 40, SHARED_TARGET_SHA256
        )


def _write_baseline_fixture(path, evaluation_seeds):
    path.mkdir()
    history = [{"epoch": 1, "loss": 1.0}]
    metrics = {"position": 1.0}
    parameters = {
        "version": "0.6.1",
        "seed": 42,
        "python_random_seed": 42,
        "numpy_seed": 42,
        "torch_seed": 42,
        "training_samples": 8,
        "batch_size": 4,
        "epochs": 1,
        "learning_rate": 5e-4,
        "optimizer": "AdamW",
        "num_workers": 0,
        "world_size": 64,
        "max_slots": 8,
        "hidden_size": 320,
        "model_parameters": 1_643_892,
        "loss_weights": {
            "terrain": 1.0,
            "placement": 2.0,
            "presence": 1.0,
            "class": 1.0,
            "action": 1.0,
            "trigger": 1.0,
        },
        "evaluation_seed_count": 30,
    }
    summary = {
        "status": "completed",
        "version": "0.6.1",
        "training_parameters": parameters,
        "final_training_losses": history[-1],
        "evaluation": {"metrics": metrics},
    }
    evaluation = {
        "evaluation_seed_count": 30,
        "evaluation_seeds": list(evaluation_seeds),
        "metrics": metrics,
        "reloaded_final_checkpoint_metrics": metrics,
    }
    (path / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "evaluation_metrics.json").write_text(json.dumps(evaluation), encoding="utf-8")
    (path / "training_history.json").write_text(json.dumps(history), encoding="utf-8")
    (path / "training_history.csv").write_text("epoch,loss\n1,1.0\n", encoding="utf-8")
    torch.save(
        {
            "completed_epochs": 1,
            "training_history": history,
            "evaluation_metrics": metrics,
            "model_state_dict": {"weight": torch.ones(1)},
        },
        path / "pixelworld_0_6_1_final.pt",
    )


def test_baseline_requires_exact_ordered_evaluation_seeds(tmp_path):
    accepted = tmp_path / "accepted"
    _write_baseline_fixture(accepted, DEFAULT_EVALUATION_SEEDS)
    assert baseline_run_is_compatible(accepted, 42, 8, 4, 1)
    rejected = tmp_path / "rejected"
    _write_baseline_fixture(rejected, reversed(DEFAULT_EVALUATION_SEEDS))
    assert not baseline_run_is_compatible(rejected, 42, 8, 4, 1)


@pytest.mark.parametrize("variant", ["", "Z", "AA"])
def test_invalid_variant(variant):
    with pytest.raises(ValueError):
        PlacementConfig(variant=variant).validate()
