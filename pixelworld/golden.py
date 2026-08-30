import json
from pathlib import Path

import torch


LOSS_FIELDS = (
    "loss",
    "terrain_loss",
    "placement_loss",
    "presence_loss",
    "class_loss",
    "action_loss",
    "trigger_loss",
)


def compare_run_to_oracle(run_path: Path, oracle_path: Path) -> dict:
    run_history = json.loads((run_path / "training_history.json").read_text(encoding="utf-8"))
    oracle_history = json.loads((oracle_path / "training_history.json").read_text(encoding="utf-8"))
    if len(run_history) != len(oracle_history):
        raise ValueError("History length differs from oracle")
    loss_deltas = [
        abs(run_epoch[field] - oracle_epoch[field])
        for run_epoch, oracle_epoch in zip(run_history, oracle_history)
        for field in LOSS_FIELDS
    ]
    run_metrics = json.loads((run_path / "evaluation_metrics.json").read_text(encoding="utf-8"))["metrics"]
    oracle_metrics = json.loads((oracle_path / "evaluation_metrics.json").read_text(encoding="utf-8"))["metrics"]
    metric_deltas = {name: abs(run_metrics[name] - oracle_metrics[name]) for name in oracle_metrics}
    run_checkpoint = torch.load(run_path / "final.pt", map_location="cpu", weights_only=True)
    oracle_checkpoint = torch.load(
        oracle_path / "pixelworld_0_6_1_final.pt", map_location="cpu", weights_only=True
    )
    run_state = run_checkpoint["model_state_dict"]
    oracle_state = oracle_checkpoint["model_state_dict"]
    bit_exact = run_state.keys() == oracle_state.keys() and all(
        torch.equal(run_state[name], oracle_state[name]) for name in run_state
    )
    return {
        "maximum_loss_deviation": max(loss_deltas, default=0.0),
        "maximum_metric_deviation": max(metric_deltas.values(), default=0.0),
        "metric_deviations": metric_deltas,
        "model_state_dict_bit_exact": bit_exact,
        "passed": max(loss_deltas, default=0.0) == 0.0
        and max(metric_deltas.values(), default=0.0) == 0.0
        and bit_exact,
    }
