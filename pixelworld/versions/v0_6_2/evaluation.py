from collections import defaultdict

import numpy as np
import torch

from pixelworld.config import (
    COORD_CLASSES,
    DEFAULT_PROMPT,
    LANDMARK_CLASSES,
    LANDMARK_SIZES,
    MAX_SLOTS,
    SIZE,
    TERRAINS,
)
from pixelworld.evaluation import METRIC_NAMES
from pixelworld.generation import render_regions, render_terrain
from pixelworld.inference import decode_ordinal

from .generation import condition_vector, generate_landscape, scene_graph_arrays
from .placement import rasterize_landmarks, valid_candidates
from .training import scene_targets


def predict_variant(model, prompt, seed, config, device):
    coordinate_values = torch.arange(COORD_CLASSES, dtype=torch.float32, device=device)
    condition = torch.tensor(condition_vector(prompt, seed))[None].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(condition)
    prediction = {
        "numeric": decode_ordinal(outputs[0][0], coordinate_values).cpu().numpy(),
        "orientation": int(outputs[1][0].argmax()),
        "biome": int(outputs[2][0].argmax()),
        "regions": outputs[3][0].argmax(-1).cpu().numpy(),
        "anchors": outputs[4][0].argmax(-1).cpu().numpy(),
        "presence": outputs[5][0].sigmoid().cpu().numpy(),
        "classes": outputs[6][0].argmax(-1).cpu().numpy(),
        "actions": outputs[7][0].argmax(-1).cpu().numpy(),
        "triggers": outputs[8][0].argmax(-1).cpu().numpy(),
        "offsets": None,
    }
    if config.uses_offset:
        prediction["offsets"] = outputs[9][0].cpu().numpy()
    return prediction


def interaction_iou(actual, predicted):
    actual = actual > 0
    predicted = predicted > 0
    return float(np.logical_and(actual, predicted).sum() / max(1, np.logical_or(actual, predicted).sum()))


def grouped_summary(values):
    return {
        str(key): {
            "count": len(group),
            "mean": float(np.mean(group)),
            "min": float(np.min(group)),
            "max": float(np.max(group)),
        }
        for key, group in sorted(values.items(), key=lambda item: str(item[0]))
        if group
    }


def evaluate_variant(model, config, device, eval_seeds=None, prompt=DEFAULT_PROMPT):
    seeds = tuple(eval_seeds or config.evaluation_seeds)
    metrics = {name: [] for name in METRIC_NAMES}
    conditional = defaultdict(list)
    by_slot = defaultdict(list)
    by_class = defaultdict(list)
    by_region = defaultdict(list)
    candidate_counts = defaultdict(list)
    fallback_usage = defaultdict(int)
    anchor_errors = []
    position_errors = []
    interaction_by_seed = {}
    requested_resolved_mismatches = 0
    present_objects = 0
    invalid_placements = 0
    water_placements = 0
    offset_clipped_components = 0
    offset_components = 0
    offset_absolute_errors = []
    desired_realized_errors = []
    projection_distances = []
    exact_offset_wishes = []
    projection_by_region = defaultdict(list)
    projection_by_anchor = defaultdict(list)
    projection_by_reason = defaultdict(list)

    for seed in seeds:
        target = generate_landscape(prompt, seed)
        targets = scene_targets(target, config.offset_radius)
        (
            numeric_t,
            orientation_t,
            biome_t,
            regions_t,
            anchors_t,
            presence_t,
            classes_t,
            actions_t,
            triggers_t,
            offsets_t,
            _,
        ) = targets
        prediction = predict_variant(model, prompt, seed, config, device)
        predicted_params = (
            prediction["biome"],
            prediction["orientation"],
            *map(int, prediction["numeric"]),
        )
        predicted_terrain = render_terrain(predicted_params, seed)
        predicted_regions_map = render_regions(predicted_terrain, predicted_params, seed)
        predicted = rasterize_landmarks(
            predicted_terrain,
            predicted_regions_map,
            prediction["regions"],
            prediction["anchors"],
            prediction["presence"],
            prediction["classes"],
            offsets=prediction["offsets"],
            offset_radius=config.offset_radius,
            return_details=True,
        )
        predicted_map, predicted_interaction, predicted_boxes, _, invalid, _ = predicted
        invalid_placements += invalid
        water_placements += int(np.any((predicted_map > 0) & (predicted_terrain == TERRAINS["water"])))

        terrain_ious = []
        for class_id in range(len(TERRAINS)):
            union = np.logical_or(predicted_terrain == class_id, target.terrain == class_id).sum()
            if union:
                terrain_ious.append(
                    np.logical_and(
                        predicted_terrain == class_id, target.terrain == class_id
                    ).sum()
                    / union
                )
        metrics["terrain_iou"].append(np.mean(terrain_ious))
        metrics["biome"].append(prediction["biome"] == biome_t)
        metrics["orientation"].append(prediction["orientation"] == orientation_t)
        metrics["params"].append(np.abs(prediction["numeric"] - numeric_t).mean())
        mask = presence_t > 0.5
        metrics["presence"].append(((prediction["presence"] >= 0.5) == mask).mean())
        metrics["region"].append((prediction["regions"][mask] == regions_t[mask]).mean())
        metrics["anchor"].append((prediction["anchors"][mask] == anchors_t[mask]).mean())
        target_graph = scene_graph_arrays(target)
        target_boxes = target_graph["boxes"]
        per_object_position = np.abs(predicted_boxes[mask, :2] - target_boxes[mask, :2]).mean(-1)
        metrics["position"].append(per_object_position.mean())
        metrics["class"].append((prediction["classes"][mask] == classes_t[mask]).mean())
        metrics["action"].append((prediction["actions"][mask] == actions_t[mask]).mean())
        metrics["trigger"].append((prediction["triggers"][mask] == triggers_t[mask]).mean())
        iou = interaction_iou(target.interaction, predicted_interaction)
        metrics["interaction"].append(iou)
        interaction_by_seed[str(seed)] = iou

        gt_terrain_result = rasterize_landmarks(
            target.terrain,
            target.regions,
            prediction["regions"],
            prediction["anchors"],
            prediction["presence"],
            prediction["classes"],
            offsets=prediction["offsets"],
            offset_radius=config.offset_radius,
        )
        conditional["position_mae_ground_truth_terrain_regions"].extend(
            np.abs(gt_terrain_result[2][mask, :2] - target_boxes[mask, :2]).mean(-1).tolist()
        )
        conditional["interaction_iou_ground_truth_terrain_regions"].append(
            interaction_iou(target.interaction, gt_terrain_result[1])
        )
        gt_attributes_result = rasterize_landmarks(
            predicted_terrain,
            predicted_regions_map,
            prediction["regions"],
            prediction["anchors"],
            presence_t,
            classes_t,
            offsets=prediction["offsets"],
            offset_radius=config.offset_radius,
            return_details=True,
        )
        conditional["position_mae_ground_truth_presence_class"].extend(
            np.abs(gt_attributes_result[2][mask, :2] - target_boxes[mask, :2]).mean(-1).tolist()
        )
        conditional["interaction_iou_ground_truth_presence_class"].append(
            interaction_iou(target.interaction, gt_attributes_result[1])
        )
        if config.uses_offset:
            offset_absolute_errors.extend(
                np.abs(prediction["offsets"][mask] - offsets_t[mask]).reshape(-1).tolist()
            )
        for slot in np.flatnonzero(mask):
            detail = gt_attributes_result[5][int(slot)]
            if detail is None:
                continue
            realized_dx = gt_attributes_result[2][slot, 0] - detail["anchor_base_x"]
            realized_dy = gt_attributes_result[2][slot, 1] - detail["anchor_base_y"]
            intended = (
                np.zeros(2, np.float32)
                if prediction["offsets"] is None
                else 8 * prediction["offsets"][slot]
            )
            desired_realized_errors.extend(
                [abs(realized_dx - intended[0]), abs(realized_dy - intended[1])]
            )
            projection_distances.append(detail["projection_distance"])
            exact_offset_wishes.append(detail["projection_distance"] <= 1e-7)
            projection_by_region[int(regions_t[slot])].append(detail["projection_distance"])
            projection_by_anchor[int(anchors_t[slot])].append(detail["projection_distance"])
            projection_by_reason[detail["projection_reason"]].append(
                detail["projection_distance"]
            )

        present_slots = np.flatnonzero(mask)
        for local_index, slot in enumerate(present_slots):
            error = float(per_object_position[local_index])
            region_correct = prediction["regions"][slot] == regions_t[slot]
            anchor_correct = prediction["anchors"][slot] == anchors_t[slot]
            if region_correct:
                conditional["position_mae_correct_region"].append(error)
            if anchor_correct:
                conditional["position_mae_correct_anchor"].append(error)
            if region_correct and anchor_correct:
                conditional["position_mae_correct_region_anchor"].append(error)
            by_slot[int(slot)].append(error)
            by_class[LANDMARK_CLASSES[int(classes_t[slot])]].append(error)
            by_region[int(regions_t[slot])].append(error)
            predicted_row, predicted_column = divmod(int(prediction["anchors"][slot]), 4)
            target_row, target_column = divmod(int(anchors_t[slot]), 4)
            anchor_errors.append(
                float(np.hypot(predicted_row - target_row, predicted_column - target_column))
            )
            position_errors.append(error)
            metadata = target.objects[int(slot) + 1]
            present_objects += 1
            mismatch = metadata["region_id"] != metadata["resolved_region_id"]
            requested_resolved_mismatches += int(mismatch)
            fallback_usage[
                f"{metadata['region_id']}->{metadata['resolved_region_id']}"
            ] += 1
            width, height = LANDMARK_SIZES[metadata["class"]]
            for region_id in range(4):
                count = len(
                    valid_candidates(
                        target.regions, target.terrain, region_id, width, height
                    )
                )
                candidate_counts[
                    f"region={region_id},class={metadata['class']},slot={slot}"
                ].append(count)
            raw = target_graph["offsets"][slot]
            offset_clipped_components += int(np.count_nonzero(np.abs(raw) > 1))
            offset_components += 2

    means = {name: float(np.mean(values)) for name, values in metrics.items()}
    conditional_means = {
        name: float(np.mean(values)) if values else None for name, values in conditional.items()
    }
    diagnostics = {
        "training_seed": config.seed,
        "end_to_end_position_mae": means["position"],
        "end_to_end_interaction_iou": means["interaction"],
        **conditional_means,
        "requested_resolved_mismatch_rate": requested_resolved_mismatches
        / max(1, present_objects),
        "fallback_region_usage": dict(sorted(fallback_usage.items())),
        "candidate_counts": grouped_summary(candidate_counts),
        "position_mae_by_slot": grouped_summary(by_slot),
        "position_mae_by_class": grouped_summary(by_class),
        "position_mae_by_region": grouped_summary(by_region),
        "interaction_iou_by_evaluation_seed": interaction_by_seed,
        "anchor_position_error_correlation": float(
            np.corrcoef(anchor_errors, position_errors)[0, 1]
        )
        if len(anchor_errors) > 1 and np.std(anchor_errors) > 0 and np.std(position_errors) > 0
        else None,
        "invalid_placements": invalid_placements,
        "water_placements": water_placements,
        "collisions": 0,
        "offset_clipping_rate": offset_clipped_components / max(1, offset_components),
        "offset_mae": float(np.mean(offset_absolute_errors))
        if offset_absolute_errors
        else None,
        "desired_realized_displacement_mae": float(np.mean(desired_realized_errors))
        if desired_realized_errors
        else None,
        "projection_distance": float(np.mean(projection_distances))
        if projection_distances
        else None,
        "exact_offset_wish_rate": float(np.mean(exact_offset_wishes))
        if exact_offset_wishes
        else None,
        "projection_distance_by_region": grouped_summary(projection_by_region),
        "projection_distance_by_anchor": grouped_summary(projection_by_anchor),
        "projection_distance_by_reason": grouped_summary(projection_by_reason),
    }
    return means, diagnostics
