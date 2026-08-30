import numpy as np
import pytest
import torch

from pixelworld.config import COORD_CLASSES, DEFAULT_PROMPT, MAX_SLOTS, RunConfig
from pixelworld.evaluation import METRIC_NAMES, evaluate_model
from pixelworld.inference import load_model, predict
from pixelworld.model import LandscapeNet
from pixelworld.training import (
    LandscapeDataset,
    compute_losses,
    create_loss_objects,
    initialize_training,
    seed_everything,
)


def test_landscape_dataset_is_deterministic():
    first = LandscapeDataset(4, progress=None)
    second = LandscapeDataset(4, progress=None)
    assert len(first) == 4
    for left, right in zip(first[2], second[2]):
        assert torch.equal(left, right)


def test_model_tensor_shapes():
    seed_everything(42)
    model = LandscapeNet()
    outputs = model(torch.zeros(3, 65))
    assert [tuple(tensor.shape) for tensor in outputs] == [
        (3, 5, COORD_CLASSES),
        (3, 4),
        (3, 4),
        (3, MAX_SLOTS, 4),
        (3, MAX_SLOTS, 16),
        (3, MAX_SLOTS),
        (3, MAX_SLOTS, 4),
        (3, MAX_SLOTS, 3),
        (3, MAX_SLOTS, 4),
    ]


def test_loss_calculation_is_finite():
    device = torch.device("cpu")
    objects = initialize_training(
        RunConfig(samples=4, batch_size=2, epochs=1, evaluation_seeds=(500000,)),
        device,
        progress=None,
    )
    losses = compute_losses(
        objects.model,
        next(iter(objects.loader)),
        device,
        objects.coord_values,
        objects.ce,
        objects.bce,
    )
    assert len(losses) == 7
    assert all(torch.isfinite(loss).item() for loss in losses)


def test_evaluation_has_all_twelve_metrics():
    seed_everything(42)
    metrics = evaluate_model(LandscapeNet(), "cpu", eval_seeds=(500000,))
    assert tuple(metrics) == METRIC_NAMES
    assert all(np.isfinite(value) for value in metrics.values())


def test_checkpoint_reload_and_inference(tmp_path):
    seed_everything(42)
    model = LandscapeNet()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    loaded, payload = load_model(checkpoint, "cpu")
    assert "model_state_dict" in payload
    assert all(
        torch.equal(model.state_dict()[name], loaded.state_dict()[name])
        for name in model.state_dict()
    )
    result = predict(loaded, DEFAULT_PROMPT, 500000, "cpu")
    assert result[0].shape == (5,)
    assert result[3].shape == (MAX_SLOTS,)


def test_cpu_execution():
    seed_everything(42)
    result = predict(LandscapeNet().to("cpu"), DEFAULT_PROMPT, 500000, "cpu")
    assert len(result) == 9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_execution():
    seed_everything(42)
    model = LandscapeNet().to("cuda")
    result = predict(model, DEFAULT_PROMPT, 500000, "cuda")
    assert len(result) == 9
    assert next(model.parameters()).device.type == "cuda"
