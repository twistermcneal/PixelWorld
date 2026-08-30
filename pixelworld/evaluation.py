import numpy as np

from .config import DEFAULT_EVALUATION_SEEDS, DEFAULT_PROMPT, MAX_SLOTS, SIZE, TERRAINS
from .generation import generate_landscape, render_regions, render_terrain, scatter_vegetation
from .inference import predict
from .placement import rasterize_landmarks
from .training import scene_targets


METRIC_NAMES = (
    "terrain_iou",
    "biome",
    "orientation",
    "params",
    "presence",
    "region",
    "anchor",
    "position",
    "class",
    "action",
    "trigger",
    "interaction",
)


def evaluate_model(model, device, eval_seeds=DEFAULT_EVALUATION_SEEDS, prompt=DEFAULT_PROMPT):
    metrics = {name: [] for name in METRIC_NAMES}
    for seed in eval_seeds:
        target = generate_landscape(prompt, seed)
        numeric_t, orient_t, biome_t, regions_t, anchors_t, presence_t, classes_t, actions_t, triggers_t = scene_targets(target)
        numeric_p, orient_p, biome_p, regions_p, anchors_p, presence_p, classes_p, actions_p, triggers_p = predict(
            model, prompt, seed, device
        )
        predicted_params = (biome_p, orient_p, *map(int, numeric_p))
        predicted_terrain = render_terrain(predicted_params, seed)
        predicted_regions = render_regions(predicted_terrain, predicted_params, seed)
        _, predicted_interaction, predicted_boxes = rasterize_landmarks(
            seed,
            predicted_terrain,
            predicted_regions,
            regions_p,
            anchors_p,
            presence_p,
            classes_p,
        )
        ious = []
        for class_id in range(len(TERRAINS)):
            union = np.logical_or(predicted_terrain == class_id, target.terrain == class_id).sum()
            if union:
                ious.append(
                    np.logical_and(predicted_terrain == class_id, target.terrain == class_id).sum() / union
                )
        metrics["terrain_iou"].append(np.mean(ious))
        metrics["biome"].append(biome_p == biome_t)
        metrics["orientation"].append(orient_p == orient_t)
        metrics["params"].append(np.abs(numeric_p - numeric_t).mean())
        mask = presence_t > .5
        metrics["presence"].append(((presence_p >= .5) == mask).mean())
        metrics["region"].append((regions_p[mask] == regions_t[mask]).mean())
        metrics["anchor"].append((anchors_p[mask] == anchors_t[mask]).mean())
        target_boxes = np.zeros((MAX_SLOTS, 4), np.int64)
        for slot in range(MAX_SLOTS):
            if slot + 1 in target.objects:
                target_boxes[slot] = target.objects[slot + 1]["bbox"]
        metrics["position"].append(np.abs(predicted_boxes[mask, :2] - target_boxes[mask, :2]).mean())
        metrics["class"].append((classes_p[mask] == classes_t[mask]).mean())
        metrics["action"].append((actions_p[mask] == actions_t[mask]).mean())
        metrics["trigger"].append((triggers_p[mask] == triggers_t[mask]).mean())
        actual = target.interaction > 0
        predicted = predicted_interaction > 0
        metrics["interaction"].append(
            np.logical_and(actual, predicted).sum() / max(1, np.logical_or(actual, predicted).sum())
        )
    means = {name: float(np.mean(values)) for name, values in metrics.items()}
    return means


def vegetation_round_trip(prompt=DEFAULT_PROMPT, seed=424242):
    sample = generate_landscape(prompt, seed)
    regenerated = generate_landscape(sample.prompt, sample.seed)
    return np.array_equal(sample.vegetation, regenerated.vegetation)
