from pathlib import Path

import numpy as np
import torch

from .config import COORD_CLASSES, SIZE
from .model import LandscapeNet, condition_vector


def decode_ordinal(logits, coord_values):
    return (logits.softmax(-1) * coord_values).sum(-1).round().clamp(0, SIZE).long()


def predict(model, prompt, seed, device):
    coord_values = torch.arange(COORD_CLASSES, dtype=torch.float32, device=device)
    x = torch.tensor(condition_vector(prompt, seed))[None].to(device)
    model.eval()
    with torch.no_grad():
        numeric, orientation, biome, regions, anchors, presence, classes, actions, triggers = model(x)
    return (
        decode_ordinal(numeric[0], coord_values).cpu().numpy(),
        int(orientation[0].argmax()),
        int(biome[0].argmax()),
        regions[0].argmax(-1).cpu().numpy(),
        anchors[0].argmax(-1).cpu().numpy(),
        presence[0].sigmoid().cpu().numpy(),
        classes[0].argmax(-1).cpu().numpy(),
        actions[0].argmax(-1).cpu().numpy(),
        triggers[0].argmax(-1).cpu().numpy(),
    )


def load_model(checkpoint_path: str | Path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = LandscapeNet().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def prediction_to_dict(prediction):
    numeric, orientation, biome, regions, anchors, presence, classes, actions, triggers = prediction
    return {
        "terrain_parameters": numeric.tolist(),
        "orientation_id": orientation,
        "biome_id": biome,
        "region_ids": regions.tolist(),
        "anchor_ids": anchors.tolist(),
        "presence_probabilities": presence.tolist(),
        "class_ids": classes.tolist(),
        "action_ids": actions.tolist(),
        "trigger_ids": triggers.tolist(),
    }
