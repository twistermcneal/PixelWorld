import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pixelworld.artifacts import (
    atomic_json,
    checkpoint_sha256,
    environment_provenance,
    git_provenance,
)
from pixelworld.config import DEFAULT_EVALUATION_SEEDS, DEFAULT_PROMPT, LANDMARK_CLASSES, LANDMARK_SIZES, MAX_SLOTS, TERRAINS
from pixelworld.evaluation import METRIC_NAMES, evaluate_model
from pixelworld.generation import generate_landscape as generate_baseline
from pixelworld.inference import load_model as load_baseline_model, predict as predict_baseline
from pixelworld.placement import anchor_candidates as baseline_anchor_candidates, rasterize_landmarks as rasterize_baseline
from pixelworld.training import resolve_device, scene_targets as baseline_scene_targets

from .config import (
    CONDITION_DIM,
    GENERATOR_TARGET_VERSION,
    LAYOUT_DIM,
    LOCAL_OFFSET_PIXELS,
    OFFSET_RADII,
    SHARED_TARGET_SHA256,
    SLOT_LATENT_DIM,
    TARGET_ANALYSIS_SCHEMA_VERSION,
    PlacementConfig,
    STUDY_NAME,
    VARIANTS,
)
from .generation import generate_landscape, ground_truth_round_trip
from .model import create_model
from .placement import sorted_candidates, valid_candidates
from .training import PlacementRunStore


def study_root(repository_root):
    return Path(repository_root) / "outputs" / "studies" / STUDY_NAME


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_round_trip(world_count=1000, offset_radius=8):
    if offset_radius != 8:
        raise ValueError("explicit 0.6.2 latent offsets require offset_radius=8")
    failures = []
    for index in range(world_count):
        prompt = f"{('temperate', 'tropical', 'arid', 'tundra')[index % 4]} coast beach forest rock portal {index}"
        seed = index + 1000
        world = generate_landscape(prompt, seed)
        repeated = generate_landscape(prompt, seed)
        reconstructed = ground_truth_round_trip(world)
        occupied = world.object_map > 0
        expected_area = sum(
            metadata["bbox"][2] * metadata["bbox"][3]
            for metadata in world.objects.values()
        )
        checks = {
            "object_map": reconstructed["object_map_equal"],
            "boxes": reconstructed["boxes_equal"],
            "interaction": reconstructed["interaction_equal"],
            "resolved_regions": reconstructed["resolved_regions_equal"],
            "no_invalid": reconstructed["invalid"] == 0,
            "no_overlap": int(occupied.sum()) == expected_area,
            "no_water": not np.any(occupied & (world.terrain == TERRAINS["water"])),
            "raw_offsets_in_range": all(
                -1.0 <= metadata[axis] <= 1.0
                for metadata in world.objects.values()
                for axis in ("offset_x", "offset_y")
            ),
            "deterministic": np.array_equal(world.object_map, repeated.object_map)
            and world.objects == repeated.objects,
        }
        if not all(checks.values()):
            failures.append({"index": index, "seed": seed, "checks": checks})
    return {
        "world_count": world_count,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:20],
        "offset_radius": offset_radius,
    }


def analyze_offset_radius(samples=14_000, radii=OFFSET_RADII):
    def anchor_base(anchor_id, width, height):
        row, column = divmod(int(anchor_id), 4)
        return (
            (column + 0.5) / 4 * (64 - width),
            (row + 0.5) / 4 * (64 - height),
        )
    def distribution(values):
        values = np.asarray(values, np.float64)
        if not len(values):
            return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
        return {
            "count": len(values),
            "mean": float(values.mean()),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
        }

    absolute_world_offsets = []
    records = []
    present_objects = 0
    for index in range(samples):
        prompt = f"{('temperate', 'tropical', 'arid', 'tundra')[index % 4]} coast beach forest rock portal {index}"
        world = generate_landscape(prompt, index + 1000)
        occupied = np.zeros_like(world.object_map)
        for oid, metadata in sorted(world.objects.items()):
            x, y, width, height = metadata["bbox"]
            base_x, base_y = anchor_base(metadata["anchor_id"], width, height)
            total_dx = x - base_x
            total_dy = y - base_y
            absolute_world_offsets.extend([abs(total_dx), abs(total_dy)])
            desired_candidates = sorted_candidates(
                valid_candidates(
                    world.regions,
                    world.terrain,
                    metadata["region_id"],
                    width,
                    height,
                ),
                base_x,
                base_y,
            )
            desired_base = desired_candidates[0] if desired_candidates else None
            desired_base_collides = bool(
                desired_base is not None
                and np.any(
                    occupied[
                        desired_base[1] : desired_base[1] + height,
                        desired_base[0] : desired_base[0] + width,
                    ]
                )
            )
            if desired_base is None:
                local_dx = None
                local_dy = None
                additional_dx = total_dx
                additional_dy = total_dy
            else:
                local_dx = desired_base[0] - base_x
                local_dy = desired_base[1] - base_y
                additional_dx = x - desired_base[0]
                additional_dy = y - desired_base[1]
            fallback_used = metadata["region_id"] != metadata["resolved_region_id"]
            records.append(
                {
                    "requested_region": metadata["region_id"],
                    "resolved_region": metadata["resolved_region_id"],
                    "fallback_used": fallback_used,
                    "desired_region_has_candidate": desired_base is not None,
                    "desired_base_collides": desired_base_collides,
                    "class": metadata["class"],
                    "size": f"{width}x{height}",
                    "anchor_id": metadata["anchor_id"],
                    "total_dx": total_dx,
                    "total_dy": total_dy,
                    "local_dx": local_dx,
                    "local_dy": local_dy,
                    "additional_dx": additional_dx,
                    "additional_dy": additional_dy,
                }
            )
            occupied[y : y + height, x : x + width] = oid
            present_objects += 1
    values = np.asarray(absolute_world_offsets, np.float64)
    rates = {str(radius): float(np.mean(values > radius)) for radius in radii}
    object_rates = {
        str(radius): float(
            np.mean(
                [
                    abs(record["total_dx"]) > radius
                    or abs(record["total_dy"]) > radius
                    for record in records
                ]
            )
        )
        for radius in radii
    }
    eligible = [radius for radius in radii if rates[str(radius)] < 0.01]
    selected = min(eligible) if eligible else max(radii)
    clipping_records = [
        record
        for record in records
        if abs(record["total_dx"]) > selected or abs(record["total_dy"]) > selected
    ]

    def grouped_cases(key):
        groups = defaultdict(list)
        for record in records:
            groups[str(key(record))].append(record)
        return {
            name: {
                "objects": len(group),
                "clipped_objects": sum(
                    abs(item["total_dx"]) > selected
                    or abs(item["total_dy"]) > selected
                    for item in group
                ),
                "clipped_rate": float(
                    np.mean(
                        [
                            abs(item["total_dx"]) > selected
                            or abs(item["total_dy"]) > selected
                            for item in group
                        ]
                    )
                ),
                "abs_dx": distribution([abs(item["total_dx"]) for item in group]),
                "abs_dy": distribution([abs(item["total_dy"]) for item in group]),
            }
            for name, group in sorted(groups.items())
        }

    local_records = [record for record in records if record["local_dx"] is not None]
    additional_sources = defaultdict(list)
    for record in records:
        if record["fallback_used"] and record["desired_base_collides"]:
            source = "fallback_and_collision"
        elif record["fallback_used"]:
            source = "fallback"
        elif record["desired_base_collides"] or record["additional_dx"] or record["additional_dy"]:
            source = "collision_or_later_slot_projection"
        else:
            source = "none"
        additional_sources[source].append(record)

    non_fallback_clipping = [record for record in clipping_records if not record["fallback_used"]]
    return {
        "samples": samples,
        "present_objects": present_objects,
        "component_count": len(values),
        "absolute_offset_world_pixels": distribution(values),
        "clipping_rate_by_radius": rates,
        "object_clipping_rate_by_radius": object_rates,
        "selected_radius": selected,
        "selection_meets_under_one_percent": bool(eligible),
        "selected_radius_clipping_cases": {
            "objects": len(clipping_records),
            "object_rate": len(clipping_records) / max(1, len(records)),
            "abs_dx": distribution([abs(item["total_dx"]) for item in clipping_records]),
            "abs_dy": distribution([abs(item["total_dy"]) for item in clipping_records]),
            "non_fallback_objects": len(non_fallback_clipping),
            "share_of_clipping_cases_without_fallback": len(non_fallback_clipping)
            / max(1, len(clipping_records)),
            "non_fallback_object_clipping_rate": len(non_fallback_clipping)
            / max(1, sum(not item["fallback_used"] for item in records)),
        },
        "clipping_by_requested_and_resolved_region": grouped_cases(
            lambda item: f"{item['requested_region']}->{item['resolved_region']}"
        ),
        "clipping_by_fallback": grouped_cases(lambda item: item["fallback_used"]),
        "clipping_by_class_and_size": grouped_cases(
            lambda item: f"{item['class']}:{item['size']}"
        ),
        "clipping_by_anchor_id": grouped_cases(lambda item: item["anchor_id"]),
        "requested_region_local_snap": {
            "objects_with_valid_requested_candidate": len(local_records),
            "objects_without_valid_requested_candidate": len(records) - len(local_records),
            "abs_dx": distribution([abs(item["local_dx"]) for item in local_records]),
            "abs_dy": distribution([abs(item["local_dy"]) for item in local_records]),
            "euclidean": distribution(
                [np.hypot(item["local_dx"], item["local_dy"]) for item in local_records]
            ),
        },
        "additional_fallback_or_collision_jump": {
            "abs_dx": distribution([abs(item["additional_dx"]) for item in records]),
            "abs_dy": distribution([abs(item["additional_dy"]) for item in records]),
            "euclidean": distribution(
                [np.hypot(item["additional_dx"], item["additional_dy"]) for item in records]
            ),
            "by_source": {
                source: {
                    "objects": len(group),
                    "abs_dx": distribution([abs(item["additional_dx"]) for item in group]),
                    "abs_dy": distribution([abs(item["additional_dy"]) for item in group]),
                    "euclidean": distribution(
                        [
                            np.hypot(item["additional_dx"], item["additional_dy"])
                            for item in group
                        ]
                    ),
                }
                for source, group in sorted(additional_sources.items())
            },
        },
    }


LEGACY_SNAP_OFFSET_REFERENCE = {
    "analysis_kind": "legacy_world_relative_snap_offset",
    "component_clipping_rate_by_radius": {
        "8": 0.4448643361617804,
        "12": 0.3539009704791423,
        "16": 0.2858658096641431,
    },
    "object_clipping_rate_by_radius": {
        "8": 0.6776586555561201,
        "12": 0.5798612875362025,
        "16": 0.49161627966058635,
    },
    "criterion_under_one_percent_met": False,
    "conclusion": "structurally_failed_before_training",
}


def analyze_latent_structure(samples=14_000):
    def summary(values):
        values = np.asarray(values, np.float64)
        if not len(values):
            return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
        return {
            "count": len(values),
            "mean": float(values.mean()),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
        }

    raw_x = []
    raw_y = []
    intended_x = []
    intended_y = []
    realized_x = []
    realized_y = []
    realization_error_x = []
    realization_error_y = []
    projection_distances = []
    anchor_relative_x = defaultdict(list)
    anchor_relative_y = defaultdict(list)
    projection_by_region = defaultdict(list)
    projection_by_anchor = defaultdict(list)
    projection_by_reason = defaultdict(list)
    fallback_pairs = defaultdict(int)
    projection_reasons = defaultdict(int)
    exact = 0
    raw_clips = 0
    objects = 0
    collisions = 0
    water_placements = 0
    deterministic_failures = 0
    target_hasher = hashlib.sha256()

    for index in range(samples):
        prompt = f"{('temperate', 'tropical', 'arid', 'tundra')[index % 4]} coast beach forest rock portal {index}"
        seed = index + 1000
        world = generate_landscape(prompt, seed)
        target_hasher.update(np.asarray(world.terrain, np.int64).tobytes())
        target_hasher.update(np.asarray(world.regions, np.int64).tobytes())
        target_hasher.update(np.asarray(world.object_map, np.int64).tobytes())
        target_hasher.update(
            json.dumps(world.objects, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        repeated = generate_landscape(prompt, seed)
        deterministic_failures += int(
            not np.array_equal(world.object_map, repeated.object_map)
            or world.objects != repeated.objects
        )
        occupied = world.object_map > 0
        expected_area = sum(
            metadata["bbox"][2] * metadata["bbox"][3]
            for metadata in world.objects.values()
        )
        collisions += int(int(occupied.sum()) != expected_area)
        water_placements += int(np.any(occupied & (world.terrain == TERRAINS["water"])))
        for metadata in world.objects.values():
            objects += 1
            offset_x, offset_y = metadata["offset_x"], metadata["offset_y"]
            raw_x.append(offset_x)
            raw_y.append(offset_y)
            raw_clips += int(abs(offset_x) > 1) + int(abs(offset_y) > 1)
            wanted_x, wanted_y = 8 * offset_x, 8 * offset_y
            actual_x = metadata["bbox"][0] - metadata["anchor_base_x"]
            actual_y = metadata["bbox"][1] - metadata["anchor_base_y"]
            intended_x.append(wanted_x)
            intended_y.append(wanted_y)
            realized_x.append(actual_x)
            realized_y.append(actual_y)
            realization_error_x.append(actual_x - wanted_x)
            realization_error_y.append(actual_y - wanted_y)
            projection_distances.append(metadata["projection_distance"])
            exact += metadata["projection_distance"] <= 1e-7
            projection_by_region[metadata["resolved_region_id"]].append(
                metadata["projection_distance"]
            )
            projection_by_anchor[metadata["anchor_id"]].append(
                metadata["projection_distance"]
            )
            projection_by_reason[metadata["projection_reason"]].append(
                metadata["projection_distance"]
            )
            projection_reasons[metadata["projection_reason"]] += 1
            fallback_pairs[
                f"{metadata['requested_region_id']}->{metadata['resolved_region_id']}"
            ] += 1
            width, height = metadata["bbox"][2:]
            candidates = valid_candidates(
                world.regions,
                world.terrain,
                metadata["resolved_region_id"],
                width,
                height,
            )
            xs = [point[0] for point in candidates]
            ys = [point[1] for point in candidates]
            relative_x = (
                (metadata["anchor_base_x"] - min(xs)) / (max(xs) - min(xs))
                if max(xs) > min(xs)
                else 0.5
            )
            relative_y = (
                (metadata["anchor_base_y"] - min(ys)) / (max(ys) - min(ys))
                if max(ys) > min(ys)
                else 0.5
            )
            anchor_relative_x[metadata["anchor_id"]].append(relative_x)
            anchor_relative_y[metadata["anchor_id"]].append(relative_y)

    def grouped(groups):
        return {str(key): summary(values) for key, values in sorted(groups.items())}

    shared_target_sha256 = target_hasher.hexdigest()
    if samples == 14_000 and shared_target_sha256 != SHARED_TARGET_SHA256:
        raise RuntimeError(
            "0.6.2 shared target digest changed: "
            f"expected {SHARED_TARGET_SHA256}, got {shared_target_sha256}"
        )
    return {
        "analysis_kind": "explicit_eight_latent_region_relative_anchors",
        "analysis_schema_version": TARGET_ANALYSIS_SCHEMA_VERSION,
        "generator_target_version": GENERATOR_TARGET_VERSION,
        "samples": samples,
        "objects": objects,
        "slot_latent_dim": SLOT_LATENT_DIM,
        "layout_dim": LAYOUT_DIM,
        "condition_dim": CONDITION_DIM,
        "local_offset_pixels": LOCAL_OFFSET_PIXELS,
        "variants_share_generator_and_target_worlds": ["B", "C", "D", "E"],
        "shared_target_sha256": shared_target_sha256,
        "raw_offset_targets": {
            "x": summary(raw_x),
            "y": summary(raw_y),
            "clipped_components": raw_clips,
            "clipping_rate": raw_clips / max(1, 2 * objects),
        },
        "intended_displacement_pixels": {
            "x": summary(intended_x),
            "y": summary(intended_y),
        },
        "realized_displacement_pixels": {
            "x": summary(realized_x),
            "y": summary(realized_y),
        },
        "desired_realized_error_pixels": {
            "x": summary(np.abs(realization_error_x)),
            "y": summary(np.abs(realization_error_y)),
        },
        "projection_distance_pixels": summary(projection_distances),
        "exact_offset_wish_count": exact,
        "exact_offset_wish_rate": exact / max(1, objects),
        "projection_reason_counts": dict(sorted(projection_reasons.items())),
        "projection_distance_by_region": grouped(projection_by_region),
        "projection_distance_by_anchor": grouped(projection_by_anchor),
        "projection_distance_by_reason": grouped(projection_by_reason),
        "anchor_base_relative_x_by_anchor": grouped(anchor_relative_x),
        "anchor_base_relative_y_by_anchor": grouped(anchor_relative_y),
        "requested_to_resolved_region": dict(sorted(fallback_pairs.items())),
        "collision_worlds": collisions,
        "water_placement_worlds": water_placements,
        "deterministic_failures": deterministic_failures,
        "legacy_snap_offset_reference": LEGACY_SNAP_OFFSET_REFERENCE,
        "selected_radius": 8,
        "selection_meets_under_one_percent": True,
        "target_clipping_semantics": "no target clipping; projection distance is reported separately",
    }


def analysis_cache_is_compatible(candidate, samples=14_000):
    expected = {
        "analysis_kind": "explicit_eight_latent_region_relative_anchors",
        "analysis_schema_version": TARGET_ANALYSIS_SCHEMA_VERSION,
        "generator_target_version": GENERATOR_TARGET_VERSION,
        "samples": samples,
        "slot_latent_dim": SLOT_LATENT_DIM,
        "layout_dim": LAYOUT_DIM,
        "condition_dim": CONDITION_DIM,
        "local_offset_pixels": LOCAL_OFFSET_PIXELS,
        "shared_target_sha256": SHARED_TARGET_SHA256,
    }
    return isinstance(candidate, dict) and all(
        candidate.get(key) == value for key, value in expected.items()
    )


def load_analysis_cache(path, samples=14_000):
    try:
        candidate = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return candidate if analysis_cache_is_compatible(candidate, samples) else None


def require_clean_study_repository(repository_root):
    provenance = git_provenance(repository_root)
    if provenance["git_commit"] == "unknown" or provenance["git_dirty"] is None:
        raise RuntimeError("study-placement requires a readable Git repository")
    if provenance["git_dirty"]:
        raise RuntimeError(
            "study-placement requires a clean Git worktree; commit or stash changes first"
        )
    return provenance


def require_study_commit(repository_root, expected_commit):
    current = git_provenance(repository_root)
    if current["git_dirty"]:
        raise RuntimeError("Git worktree became dirty during study-placement")
    if current["git_commit"] != expected_commit:
        raise RuntimeError(
            "Git commit changed during study-placement: "
            f"expected {expected_commit}, got {current['git_commit']}"
        )
    return current


def baseline_run_path(repository_root, seed):
    return Path(repository_root) / "outputs" / "0.6.1-reproducibility" / f"seed-{seed}"


def _baseline_evaluation_seeds(summary, evaluation):
    parameters = summary["training_parameters"]
    explicit = evaluation.get("evaluation_seeds", parameters.get("evaluation_seeds"))
    if explicit is not None:
        return tuple(explicit)
    expected_formula = "500000 + i * 7919 for i in range(30)"
    if (
        parameters.get("evaluation_seed_formula") == expected_formula
        and parameters.get("evaluation_seed_count") == len(DEFAULT_EVALUATION_SEEDS)
        and evaluation.get("evaluation_seed_count") == len(DEFAULT_EVALUATION_SEEDS)
    ):
        return tuple(DEFAULT_EVALUATION_SEEDS)
    return None


def baseline_run_metadata(path, seed, samples, batch_size, epochs):
    path = Path(path)
    summary_path = path / "run_summary.json"
    evaluation_path = path / "evaluation_metrics.json"
    history_json_path = path / "training_history.json"
    history_csv_path = path / "training_history.csv"
    checkpoint = path / "pixelworld_0_6_1_final.pt"
    required = (
        summary_path,
        evaluation_path,
        history_json_path,
        history_csv_path,
        checkpoint,
    )
    if not all(item.is_file() and item.stat().st_size > 0 for item in required):
        raise ValueError("Frozen baseline artifacts are incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    history = json.loads(history_json_path.read_text(encoding="utf-8"))
    with history_csv_path.open("r", encoding="utf-8", newline="") as handle:
        history_csv = list(csv.DictReader(handle))
    parameters = summary["training_parameters"]
    expected_parameters = {
        "version": "0.6.1",
        "seed": seed,
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "training_samples": samples,
        "batch_size": batch_size,
        "epochs": epochs,
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
    }
    if summary.get("status") != "completed" or summary.get("version") != "0.6.1":
        raise ValueError("Frozen baseline summary is not a completed 0.6.1 run")
    if any(parameters.get(key) != value for key, value in expected_parameters.items()):
        raise ValueError("Frozen baseline training parameters do not match")
    evaluation_seeds = _baseline_evaluation_seeds(summary, evaluation)
    if evaluation_seeds != tuple(DEFAULT_EVALUATION_SEEDS):
        raise ValueError("Frozen baseline evaluation seed list does not match exactly")
    if len(history) != epochs or history[-1] != summary.get("final_training_losses"):
        raise ValueError("Frozen baseline history is inconsistent with the summary")
    if len(history_csv) != epochs or any(
        int(row["epoch"]) != int(item["epoch"])
        or float(row["loss"]) != float(item["loss"])
        for row, item in zip(history_csv, history)
    ):
        raise ValueError("Frozen baseline CSV and JSON histories are inconsistent")
    if evaluation.get("metrics") != summary.get("evaluation", {}).get("metrics"):
        raise ValueError("Frozen baseline evaluation metrics are inconsistent")
    if evaluation.get("reloaded_final_checkpoint_metrics") != evaluation.get("metrics"):
        raise ValueError("Frozen baseline reloaded metrics are inconsistent")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("completed_epochs") != epochs
        or payload.get("training_history") != history
        or payload.get("evaluation_metrics") != evaluation.get("metrics")
        or not isinstance(payload.get("model_state_dict"), dict)
        or (
            payload.get("training_parameters") is not None
            and payload.get("training_parameters") != parameters
        )
    ):
        raise ValueError("Frozen baseline final checkpoint is inconsistent")
    digest = checkpoint_sha256(checkpoint)
    recorded_digest = summary.get("final_checkpoint_sha256") or summary.get(
        "checkpoint_sha256"
    )
    if recorded_digest is not None and recorded_digest != digest:
        raise ValueError("Frozen baseline checkpoint hash does not match")
    return {
        "checkpoint_sha256": digest,
        "evaluation_seeds": list(evaluation_seeds),
        "git_commit": summary.get("git_commit"),
        "training_parameters": parameters,
    }


def baseline_run_is_compatible(path, seed, samples, batch_size, epochs):
    try:
        baseline_run_metadata(path, seed, samples, batch_size, epochs)
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _grouped(values):
    return {
        str(key): {"count": len(group), "mean": float(np.mean(group))}
        for key, group in values.items()
        if group
    }


def diagnose_baseline(model, device, training_seed, eval_seeds=DEFAULT_EVALUATION_SEEDS):
    by_slot = defaultdict(list)
    by_class = defaultdict(list)
    by_region = defaultdict(list)
    candidate_counts = defaultdict(list)
    conditional = defaultdict(list)
    fallback = defaultdict(int)
    anchor_errors = []
    position_errors = []
    interaction_by_seed = {}
    mismatches = 0
    object_count = 0
    for seed in eval_seeds:
        target = generate_baseline(DEFAULT_PROMPT, seed)
        numeric_t, _, _, regions_t, anchors_t, presence_t, classes_t, _, _ = baseline_scene_targets(target)
        numeric, orientation, biome, regions, anchors, presence, classes, _, _ = predict_baseline(
            model, DEFAULT_PROMPT, seed, device
        )
        params = (biome, orientation, *map(int, numeric))
        from pixelworld.generation import render_regions, render_terrain

        terrain = render_terrain(params, seed)
        region_map = render_regions(terrain, params, seed)
        _, interaction, boxes = rasterize_baseline(
            seed, terrain, region_map, regions, anchors, presence, classes
        )
        target_boxes = np.zeros((MAX_SLOTS, 4), np.int64)
        for slot in range(MAX_SLOTS):
            metadata = target.objects.get(slot + 1)
            if metadata:
                target_boxes[slot] = metadata["bbox"]
        mask = presence_t > 0.5
        errors = np.abs(boxes[mask, :2] - target_boxes[mask, :2]).mean(-1)
        interaction_by_seed[str(seed)] = float(
            np.logical_and(interaction > 0, target.interaction > 0).sum()
            / max(1, np.logical_or(interaction > 0, target.interaction > 0).sum())
        )
        for local_index, slot in enumerate(np.flatnonzero(mask)):
            error = float(errors[local_index])
            region_correct = regions[slot] == regions_t[slot]
            anchor_correct = anchors[slot] == anchors_t[slot]
            if region_correct:
                conditional["position_mae_correct_region"].append(error)
            if anchor_correct:
                conditional["position_mae_correct_anchor"].append(error)
            if region_correct and anchor_correct:
                conditional["position_mae_correct_region_anchor"].append(error)
            by_slot[int(slot)].append(error)
            metadata = target.objects[int(slot) + 1]
            by_class[metadata["class"]].append(error)
            by_region[metadata["region_id"]].append(error)
            object_count += 1
            mismatches += metadata["region_id"] != metadata["resolved_region_id"]
            fallback[f"{metadata['region_id']}->{metadata['resolved_region_id']}"] += 1
            width, height = LANDMARK_SIZES[metadata["class"]]
            for region_id in range(4):
                count = len(
                    baseline_anchor_candidates(
                        target.regions,
                        target.terrain,
                        region_id,
                        width,
                        height,
                        seed,
                        int(slot),
                    )
                )
                candidate_counts[
                    f"region={region_id},class={metadata['class']},slot={slot}"
                ].append(count)
            anchor_errors.append(float(anchors[slot] != anchors_t[slot]))
            position_errors.append(error)
    metrics = evaluate_model(model, device, eval_seeds=eval_seeds)
    return {
        "training_seed": training_seed,
        "end_to_end_position_mae": metrics["position"],
        "end_to_end_interaction_iou": metrics["interaction"],
        **{
            key: float(np.mean(value)) if value else None
            for key, value in conditional.items()
        },
        "requested_resolved_mismatch_rate": mismatches / max(1, object_count),
        "fallback_region_usage": dict(fallback),
        "candidate_counts": _grouped(candidate_counts),
        "position_mae_by_slot": _grouped(by_slot),
        "position_mae_by_class": _grouped(by_class),
        "position_mae_by_region": _grouped(by_region),
        "interaction_iou_by_evaluation_seed": interaction_by_seed,
        "anchor_position_error_correlation": float(
            np.corrcoef(anchor_errors, position_errors)[0, 1]
        ),
        "invalid_placements": 0,
        "water_placements": 0,
        "collisions": 0,
        "offset_clipping_rate": 0.0,
    }


def baseline_record(
    repository_root,
    seed,
    device,
    samples=14_000,
    batch_size=128,
    epochs=45,
    metadata=None,
):
    path = baseline_run_path(repository_root, seed)
    metadata = metadata or baseline_run_metadata(
        path, seed, samples, batch_size, epochs
    )
    evaluation = json.loads((path / "evaluation_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
    model, _ = load_baseline_model(path / "pixelworld_0_6_1_final.pt", device)
    diagnostics = diagnose_baseline(model, device, seed)
    return {
        "version": "0.6.1",
        "variant": "A",
        "seed": seed,
        "run_id": f"reused-0.6.1-seed-{seed}",
        "run_path": str(path),
        "reused": True,
        "runtime_seconds": summary["timings"]["total_seconds"],
        "metrics": evaluation["metrics"],
        "diagnostics": diagnostics,
        "provenance": metadata,
    }


def load_v062_record(repository_root, run_id, reused=False):
    store = PlacementRunStore.open(repository_root, run_id)
    summary = json.loads(store.artifact_path("run_summary.json").read_text(encoding="utf-8"))
    return {
        "version": "0.6.2",
        "variant": summary["config"]["variant"],
        "seed": summary["config"]["seed"],
        "run_id": run_id,
        "run_path": str(store.path),
        "reused": reused,
        "runtime_seconds": summary["timings"]["total_seconds"],
        "metrics": summary["metrics"],
        "diagnostics": summary["diagnostics"],
        "provenance": summary["provenance"],
    }


def run_subprocess(arguments, repository_root):
    subprocess.run([sys.executable, "-m", "pixelworld.cli", *arguments], cwd=repository_root, check=True)


def _validate_v062_checkpoint(path, config, expected_commit, target_digest):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    provenance = payload.get("provenance", {})
    if (
        payload.get("config") != config.to_dict()
        or payload.get("variant") != config.variant
        or payload.get("shared_target_sha256") != target_digest
        or payload.get("evaluation_seeds") != list(config.evaluation_seeds)
        or provenance.get("git_commit") != expected_commit
        or provenance.get("git_dirty") is not False
    ):
        raise ValueError("Existing 0.6.2 checkpoint provenance is incompatible")
    return payload


def _validate_completed_v062_run(store, config, expected_commit, target_digest):
    existing = PlacementRunStore.open(store.repository_root, store.run_id).config()
    if existing.to_dict() != config.to_dict():
        raise ValueError(f"Existing run {store.run_id!r} has an incompatible configuration")
    summary = json.loads(store.artifact_path("run_summary.json").read_text(encoding="utf-8"))
    provenance = summary.get("provenance", {})
    if (
        summary.get("status") != "completed"
        or summary.get("config") != config.to_dict()
        or summary.get("shared_target_sha256") != target_digest
        or summary.get("evaluation_seeds") != list(config.evaluation_seeds)
        or provenance.get("git_commit") != expected_commit
        or provenance.get("git_dirty") is not False
        or provenance.get("variant") != config.variant
        or provenance.get("placement_config") != config.to_dict()
    ):
        raise ValueError(f"Existing run {store.run_id!r} has incompatible provenance")
    checkpoint = store.artifact_path("final.pt")
    _validate_v062_checkpoint(checkpoint, config, expected_commit, target_digest)
    if checkpoint_sha256(checkpoint) != summary.get("checkpoint_sha256"):
        raise ValueError(f"Existing run {store.run_id!r} has a checkpoint hash mismatch")


def ensure_v062_run(repository_root, config, device, expected_commit, target_digest):
    run_id = (
        f"v062-{config.variant}-seed{config.seed}-n{config.samples}-"
        f"b{config.batch_size}-e{config.epochs}-r{config.offset_radius}"
    )
    store = PlacementRunStore(repository_root, run_id)
    require_study_commit(repository_root, expected_commit)
    if store.artifact_path("run_summary.json").is_file():
        _validate_completed_v062_run(store, config, expected_commit, target_digest)
        return load_v062_record(repository_root, run_id, reused=True)
    if store.artifact_path("config.json").is_file():
        existing = PlacementRunStore.open(repository_root, run_id).config()
        if existing.to_dict() != config.to_dict():
            raise ValueError(f"Existing run {run_id!r} has an incompatible configuration")
        latest = store.artifact_path("latest.pt")
        if not latest.is_file():
            raise ValueError(f"Incomplete run {run_id!r} has no recovery checkpoint")
        _validate_v062_checkpoint(latest, config, expected_commit, target_digest)
        require_study_commit(repository_root, expected_commit)
        run_subprocess(["resume", "--run", run_id, "--device", device], repository_root)
    else:
        require_study_commit(repository_root, expected_commit)
        run_subprocess(
            [
                "train",
                "--version",
                "0.6.2",
                "--variant",
                config.variant,
                "--samples",
                str(config.samples),
                "--batch-size",
                str(config.batch_size),
                "--epochs",
                str(config.epochs),
                "--seed",
                str(config.seed),
                "--offset-radius",
                str(config.offset_radius),
                "--run-id",
                run_id,
                "--device",
                device,
            ],
            repository_root,
        )
    require_study_commit(repository_root, expected_commit)
    _validate_completed_v062_run(store, config, expected_commit, target_digest)
    return load_v062_record(repository_root, run_id)


def aggregate(records, root, offset_analysis, round_trip, study_provenance=None):
    flat_rows = []
    for record in records:
        row = {"variant": record["variant"], "seed": record["seed"]}
        row.update(record["metrics"])
        for key in (
            "end_to_end_position_mae",
            "end_to_end_interaction_iou",
            "position_mae_correct_region",
            "position_mae_correct_anchor",
            "position_mae_correct_region_anchor",
            "position_mae_ground_truth_terrain_regions",
            "interaction_iou_ground_truth_terrain_regions",
            "position_mae_ground_truth_presence_class",
            "interaction_iou_ground_truth_presence_class",
            "requested_resolved_mismatch_rate",
            "anchor_position_error_correlation",
            "offset_clipping_rate",
            "offset_mae",
            "desired_realized_displacement_mae",
            "projection_distance",
            "exact_offset_wish_rate",
            "invalid_placements",
            "water_placements",
            "collisions",
        ):
            row[key] = record["diagnostics"].get(key)
        flat_rows.append(row)
    write_csv(root / "metrics_by_seed.csv", flat_rows)

    metric_names = [
        key for key in flat_rows[0] if key not in ("variant", "seed")
    ]
    statistics_rows = []
    by_variant = defaultdict(list)
    for row in flat_rows:
        by_variant[row["variant"]].append(row)
    for variant, rows in sorted(by_variant.items()):
        for metric in metric_names:
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            if not values:
                continue
            statistics_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            )
    write_csv(root / "metrics_statistics.csv", statistics_rows)

    baseline = {row["seed"]: row for row in by_variant["A"]}
    delta_rows = []
    for variant, rows in sorted(by_variant.items()):
        if variant == "A":
            continue
        for row in rows:
            for metric in metric_names:
                if row[metric] is not None and baseline[row["seed"]][metric] is not None:
                    delta_rows.append(
                        {
                            "variant": variant,
                            "seed": row["seed"],
                            "metric": metric,
                            "value": row[metric],
                            "baseline_A": baseline[row["seed"]][metric],
                            "delta": float(row[metric]) - float(baseline[row["seed"]][metric]),
                        }
                    )
    comparison_note = (
        "A versus B-E deltas are matched by training seed only; they are not paired "
        "target-world deltas because A uses frozen 0.6.1 generator semantics while "
        "B-E share the 0.6.2 eight-latent target worlds."
    )
    for row in delta_rows:
        row["comparison_scope"] = "seed-matched benchmark; target worlds are not paired"
    write_csv(root / "seed_matched_benchmark_deltas.csv", delta_rows)

    mean_lookup = {
        (row["variant"], row["metric"]): row["mean"] for row in statistics_rows
    }
    acceptance = {}
    for variant in sorted(by_variant):
        if variant == "A":
            continue
        checks = {
            "position_under_5_36": mean_lookup[(variant, "position")] < 5.36,
            "interaction_at_least_0_470": mean_lookup[(variant, "interaction")] >= 0.470,
            "presence_preserved": mean_lookup[(variant, "presence")]
            >= mean_lookup[("A", "presence")] - 0.005,
            "class_preserved": mean_lookup[(variant, "class")]
            >= mean_lookup[("A", "class")] - 0.005,
            "action_preserved": mean_lookup[(variant, "action")]
            >= mean_lookup[("A", "action")] - 0.005,
            "trigger_preserved": mean_lookup[(variant, "trigger")]
            >= mean_lookup[("A", "trigger")] - 0.005,
            "no_collisions": mean_lookup[(variant, "collisions")] == 0,
            "no_water": mean_lookup[(variant, "water_placements")] == 0,
            "offset_clipping_under_one_percent": (
                variant == "B"
                or mean_lookup[(variant, "offset_clipping_rate")] < 0.01
            ),
            "ground_truth_round_trip": round_trip["passed"],
        }
        acceptance[variant] = {**checks, "passed": all(checks.values())}
    winners = [variant for variant, result in acceptance.items() if result["passed"]]
    if winners:
        recommendation = min(winners, key=lambda variant: mean_lookup[(variant, "position")])
        reason = "Erfüllt sämtliche Akzeptanzkriterien und besitzt unter den Kandidaten die beste Positions-MAE."
    else:
        recommendation = min(
            acceptance,
            key=lambda variant: (
                mean_lookup[(variant, "position")],
                -mean_lookup[(variant, "interaction")],
            ),
        )
        failed = [name for name, passed in acceptance[recommendation].items() if name != "passed" and not passed]
        reason = "Kein vollständiger Gewinner; bester Placement-Trade-off. Verfehlt: " + ", ".join(failed)
    summary = {
        "study": STUDY_NAME,
        "record_count": len(records),
        "offset_analysis": offset_analysis,
        "round_trip": round_trip,
        "acceptance": acceptance,
        "recommended_variant": recommendation,
        "recommendation_reason": reason,
        "comparison_scope": comparison_note,
        "provenance": study_provenance,
    }
    atomic_json(root / "study_summary.json", summary)
    (root / "recommendation.md").write_text(
        f"# PixelWorld 0.6.2 Placement-Empfehlung\n\n"
        f"Empfohlene Variante: **{recommendation}**\n\n{reason}\n",
        encoding="utf-8",
    )
    atomic_json(
        root / "placement_diagnostics.json",
        {f"{record['variant']}-seed{record['seed']}": record["diagnostics"] for record in records},
    )
    return summary


def run_study(
    repository_root,
    seeds=(42, 43, 44, 45, 46),
    variants=VARIANTS,
    samples=14_000,
    batch_size=128,
    epochs=45,
    device="cuda",
):
    repository_root = Path(repository_root).resolve()
    initial_git = require_clean_study_repository(repository_root)
    initial_commit = initial_git["git_commit"]
    selected_device = resolve_device(device)
    parameter_counts = {
        variant: sum(parameter.numel() for parameter in create_model(variant).parameters())
        for variant in variants
    }
    study_provenance = environment_provenance(repository_root, selected_device, 0)
    study_provenance["model_parameters"] = parameter_counts
    study_provenance.update(
        {
            "pixelworld_version": "0.6.2",
            "variants": list(variants),
            "shared_target_sha256": SHARED_TARGET_SHA256,
            "evaluation_seeds": list(DEFAULT_EVALUATION_SEEDS),
        }
    )
    root = study_root(repository_root)
    root.mkdir(parents=True, exist_ok=True)
    offset_analysis_path = root / "offset_analysis.json"
    offset_analysis = None
    if offset_analysis_path.is_file() and samples == 14_000:
        offset_analysis = load_analysis_cache(offset_analysis_path, samples)
    if offset_analysis is None:
        offset_analysis = analyze_latent_structure(samples)
        if samples == 14_000 and not analysis_cache_is_compatible(offset_analysis, samples):
            raise RuntimeError("Recomputed 0.6.2 analysis metadata or digest is invalid")
        atomic_json(offset_analysis_path, offset_analysis)
    radius = 8
    round_trip = validate_round_trip(1000 if samples == 14_000 else min(100, samples), radius)
    atomic_json(root / "round_trip.json", round_trip)
    if not round_trip["passed"]:
        raise RuntimeError("0.6.2 ground-truth round-trip failed")
    study_config = {
        "version": "0.6.2",
        "study": STUDY_NAME,
        "seeds": list(seeds),
        "variants": list(variants),
        "samples": samples,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": 5e-4,
        "optimizer": "AdamW",
        "evaluation_seeds": list(DEFAULT_EVALUATION_SEEDS),
        "offset_radius": radius,
        "sequential_gpu_processes": True,
        "shared_target_sha256": SHARED_TARGET_SHA256,
        "provenance": study_provenance,
        "placement_configs": [
            PlacementConfig(
                variant=variant,
                seed=seed,
                samples=samples,
                batch_size=batch_size,
                epochs=epochs,
                offset_radius=radius,
            ).validate().to_dict()
            for variant in variants
            if variant != "A"
            for seed in seeds
        ],
        "variant_a_frozen_baseline": {
            "version": "0.6.1",
            "training_seeds": list(seeds) if "A" in variants else [],
            "samples": samples,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": 5e-4,
            "optimizer": "AdamW",
            "evaluation_seeds": list(DEFAULT_EVALUATION_SEEDS),
        },
        "comparison_scope": (
            "A versus B-E is matched by training seed, not by target world; "
            "only B-E share generated target worlds."
        ),
    }
    atomic_json(root / "study_config.json", study_config)
    records = []
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown placement variant: {variant!r}")
        for seed in seeds:
            require_study_commit(repository_root, initial_commit)
            if variant == "A":
                path = baseline_run_path(repository_root, seed)
                try:
                    baseline_metadata = baseline_run_metadata(
                        path, seed, samples, batch_size, epochs
                    )
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"No compatible frozen Variant-A run for seed {seed}; run 0.6.1 separately"
                    ) from error
                record = baseline_record(
                    repository_root,
                    seed,
                    selected_device,
                    samples,
                    batch_size,
                    epochs,
                    baseline_metadata,
                )
            else:
                config = PlacementConfig(
                    variant=variant,
                    seed=seed,
                    samples=samples,
                    batch_size=batch_size,
                    epochs=epochs,
                    offset_radius=radius,
                ).validate()
                record = ensure_v062_run(
                    repository_root,
                    config,
                    str(selected_device),
                    initial_commit,
                    SHARED_TARGET_SHA256,
                )
            records.append(record)
            write_csv(
                root / "runs.csv",
                [
                    {
                        "variant": item["variant"],
                        "seed": item["seed"],
                        "run_id": item["run_id"],
                        "run_path": item["run_path"],
                        "reused": item["reused"],
                        "runtime_seconds": item["runtime_seconds"],
                    }
                    for item in records
                ],
            )
    require_study_commit(repository_root, initial_commit)
    return aggregate(records, root, offset_analysis, round_trip, study_provenance)
