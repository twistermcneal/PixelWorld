import json

import pytest
import torch

from pixelworld import cli
from pixelworld.artifacts import RunStore, atomic_json, validate_run_id
from pixelworld.config import RunConfig
from pixelworld.training import run_training


def small_config(epochs=2):
    return RunConfig(
        samples=8,
        batch_size=4,
        epochs=epochs,
        seed=42,
        evaluation_seeds=(500000,),
    )


def test_atomic_artifacts_and_weights_only_checkpoint(tmp_path):
    store = RunStore.create(tmp_path, small_config(epochs=1), "atomic-test")
    atomic_json(store.path / "sample.json", {"text": "Grün"})
    store.write_checkpoint({"tensor": torch.arange(3)}, final=True)
    assert json.loads((store.path / "sample.json").read_text(encoding="utf-8"))["text"] == "Grün"
    assert torch.equal(
        torch.load(store.checkpoint_path(final=True), weights_only=True)["tensor"],
        torch.arange(3),
    )
    assert not list(store.path.glob("*.tmp"))


def test_config_records_all_controlled_seeds(tmp_path):
    store = RunStore.create(tmp_path, small_config(), "seed-metadata")
    document = json.loads((store.path / "config.json").read_text(encoding="utf-8"))
    assert document["seeds"] == {
        "python": 42,
        "numpy": 42,
        "pytorch": 42,
        "dataloader": 42,
        "evaluation": [500000],
    }


@pytest.mark.parametrize("run_id", ["../escape", "..", "a/b", "a\\b", ""])
def test_invalid_run_id_and_path_traversal(run_id):
    with pytest.raises(ValueError):
        validate_run_id(run_id)


def test_unknown_run_id(tmp_path):
    with pytest.raises(ValueError, match="Unknown run ID"):
        RunStore.open(tmp_path, "missing")


def test_resume_is_bit_exact_to_continuous_run(tmp_path):
    continuous = RunStore.create(tmp_path, small_config(), "continuous")
    run_training(continuous, device="cpu")

    resumed = RunStore.create(tmp_path, small_config(), "resumed")
    stopped = run_training(resumed, device="cpu", stop_after_epoch=1)
    assert stopped["status"] == "aborted"
    run_training(resumed, device="cpu", resume=True)

    continuous_payload = torch.load(continuous.checkpoint_path(final=True), weights_only=True)
    resumed_payload = torch.load(resumed.checkpoint_path(final=True), weights_only=True)
    for continuous_epoch, resumed_epoch in zip(
        continuous_payload["training_history"], resumed_payload["training_history"]
    ):
        assert {
            key: value for key, value in continuous_epoch.items() if key != "epoch_seconds"
        } == {
            key: value for key, value in resumed_epoch.items() if key != "epoch_seconds"
        }
    assert all(
        torch.equal(continuous_payload["model_state_dict"][name], tensor)
        for name, tensor in resumed_payload["model_state_dict"].items()
    )


def test_cli_smoke_train_evaluate_infer_and_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", tmp_path)
    assert cli.main([
        "train", "--version", "0.6.1", "--samples", "4", "--batch-size", "2",
        "--epochs", "1", "--seed", "42", "--run-id", "cli-smoke", "--device", "cpu",
    ]) == 0
    assert cli.main(["evaluate", "--run", "cli-smoke", "--device", "cpu"]) == 0
    assert cli.main([
        "infer", "--run", "cli-smoke", "--prompt", "tropical coast beach forest rock portal",
        "--seed", "500000", "--device", "cpu",
    ]) == 0
    assert cli.main(["runs"]) == 0
    assert "cli-smoke" in capsys.readouterr().out


def test_cli_reports_invalid_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", tmp_path)
    assert cli.main([
        "train", "--version", "0.6.1", "--samples", "0", "--run-id", "bad",
    ]) == 2
