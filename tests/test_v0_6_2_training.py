import pytest
import torch

from pixelworld import cli
from pixelworld.evaluation import METRIC_NAMES
from pixelworld.training import seed_everything
from pixelworld.versions.v0_6_2.config import PlacementConfig
from pixelworld.versions.v0_6_2.model import create_model
from pixelworld.versions.v0_6_2.study import aggregate
from pixelworld.versions.v0_6_2.training import (
    PlacementRunStore,
    compute_losses,
    initialize_training,
    load_checkpoint,
    run_training,
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
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(store.path / "final.pt", config("D"), "cpu")


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
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--version", "0.6.2", "--variant", "Z"])


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
        "paired_deltas.csv",
        "placement_diagnostics.json",
        "study_summary.json",
        "recommendation.md",
    ):
        assert (tmp_path / name).is_file()


@pytest.mark.parametrize("variant", ["", "Z", "AA"])
def test_invalid_variant(variant):
    with pytest.raises(ValueError):
        PlacementConfig(variant=variant).validate()
