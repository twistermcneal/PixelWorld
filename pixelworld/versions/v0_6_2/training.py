import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from pixelworld.artifacts import (
    atomic_json,
    atomic_torch_save,
    capture_rng_state,
    checkpoint_sha256,
    environment_provenance,
    resolve_contained_run_path,
    resolve_run_artifact,
    restore_rng_state,
    validate_run_id,
)
from pixelworld.config import (
    ACTIONS,
    BIOMES,
    COORD_CLASSES,
    LANDMARK_CLASSES,
    MAX_SLOTS,
    SIZE,
    TRIGGER_TYPES,
)
from pixelworld.training import masked_ce, ordinal_loss, resolve_device, seed_everything

from .config import (
    LAYOUT_DIM,
    SHARED_TARGET_SHA256,
    SLOT_LATENT_DIM,
    PlacementConfig,
    STUDY_NAME,
)
from .generation import condition_vector, generate_landscape, scene_graph_arrays
from .model import create_model


def scene_targets(world, offset_radius):
    if offset_radius != 8:
        raise ValueError("explicit latent targets require offset_radius=8")
    biome, orientation, shore, width, rock, forest, density = world.terrain_params
    graph = scene_graph_arrays(world)
    actions = np.zeros(MAX_SLOTS, np.int64)
    triggers = np.zeros(MAX_SLOTS, np.int64)
    xy = np.zeros((MAX_SLOTS, 2), np.float32)
    for slot in range(MAX_SLOTS):
        metadata = world.objects.get(slot + 1)
        if metadata is None:
            continue
        actions[slot] = ACTIONS.index(metadata["action"])
        triggers[slot] = TRIGGER_TYPES.index(metadata["trigger_type"])
        x, y, _, _ = metadata["bbox"]
        xy[slot] = [x / (SIZE - 1), y / (SIZE - 1)]
    return (
        np.asarray([shore, width, rock, forest, density]),
        orientation,
        biome,
        graph["regions"],
        graph["anchors"],
        graph["presence"],
        graph["classes"],
        actions,
        triggers,
        graph["offsets"],
        xy,
    )


class LandscapeDataset062(Dataset):
    def __init__(self, n=14_000, offset_radius=8, progress=print):
        self.samples = []
        if progress:
            progress(f"Erzeuge {n:,} 0.6.2-Trainingslandschaften einmalig ...")
        for index in range(n):
            prompt = f"{BIOMES[index % 4]} coast beach forest rock portal {index}"
            seed = index + 1000
            targets = scene_targets(generate_landscape(prompt, seed), offset_radius)
            self.samples.append(
                tuple(torch.tensor(value) for value in (condition_vector(prompt, seed), *targets))
            )
            if progress and (index + 1) % 2000 == 0:
                progress(f"  {index + 1:,}/{n:,}")
        if progress:
            progress("0.6.2-Datensatz bereit.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def masked_smooth_l1(prediction, target, presence, beta=0.25):
    errors = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta).mean(-1)
    return (errors * presence).sum() / presence.sum().clamp_min(1)


def compute_losses(model, batch, config, device, coord_values, ce, bce):
    (
        condition,
        numeric,
        orientation,
        biome,
        regions,
        anchors,
        presence,
        classes,
        actions,
        triggers,
        offsets,
        xy,
    ) = [tensor.to(device, non_blocking=True) for tensor in batch]
    outputs = model(condition)
    terrain_loss = (
        ordinal_loss(outputs[0], numeric, coord_values).mean()
        + ce(outputs[1], orientation).mean()
        + ce(outputs[2], biome).mean()
    )
    region_loss = masked_ce(outputs[3], regions, presence, ce)
    anchor_loss = masked_ce(outputs[4], anchors, presence, ce)
    offset_loss = torch.zeros((), device=device)
    auxiliary_loss = torch.zeros((), device=device)
    placement_loss = region_loss + anchor_loss
    if config.uses_offset:
        offset_loss = masked_smooth_l1(outputs[9], offsets, presence)
        placement_loss = placement_loss + config.offset_loss_weight * offset_loss
    if config.uses_auxiliary_xy:
        auxiliary_loss = masked_smooth_l1(outputs[10], xy, presence)
        placement_loss = placement_loss + config.auxiliary_loss_weight * auxiliary_loss
    presence_loss = (
        bce(outputs[5], presence) * torch.where(presence > 0.5, 1.0, 2.0)
    ).mean()
    class_loss = masked_ce(outputs[6], classes, presence, ce)
    action_loss = masked_ce(outputs[7], actions, presence, ce)
    trigger_loss = masked_ce(outputs[8], triggers, presence, ce)
    loss = (
        terrain_loss
        + 2 * placement_loss
        + presence_loss
        + class_loss
        + action_loss
        + trigger_loss
    )
    return {
        "loss": loss,
        "terrain_loss": terrain_loss,
        "placement_loss": placement_loss,
        "region_loss": region_loss,
        "anchor_loss": anchor_loss,
        "offset_loss": offset_loss,
        "auxiliary_loss": auxiliary_loss,
        "presence_loss": presence_loss,
        "class_loss": class_loss,
        "action_loss": action_loss,
        "trigger_loss": trigger_loss,
    }


CONFLICT_METRIC_FIELDS = (
    "gradient_cosine_mean",
    "gradient_cosine_median",
    "gradient_cosine_std",
    "gradient_cosine_min",
    "gradient_cosine_max",
    "gradient_cosine_p10",
    "gradient_cosine_p25",
    "gradient_cosine_p75",
    "gradient_cosine_p90",
    "gradient_negative_rate",
    "gradient_projected_rate",
    "gradient_discrete_norm_mean",
    "gradient_offset_norm_mean",
    "gradient_removed_norm_mean",
    "gradient_removed_ratio_mean",
    "gradient_cosine_post_mean",
    "gradient_encoder_cosine_mean",
    "gradient_decoder_cosine_mean",
)


def project_conflicting_gradients(discrete, offset, project=False, eps=1e-12):
    """Measure and optionally project offset gradients without mutating inputs."""
    pairs = [(gd, go) for gd, go in zip(discrete, offset) if gd is not None and go is not None]
    if not pairs:
        return list(offset), {
            "cosine": 0.0,
            "post_cosine": 0.0,
            "discrete_norm": 0.0,
            "offset_norm": 0.0,
            "removed_norm": 0.0,
            "removed_ratio": 0.0,
            "negative": False,
            "projected": False,
        }
    dot = sum((gd * go).sum() for gd, go in pairs)
    discrete_sq = sum(gd.square().sum() for gd, _ in pairs)
    offset_sq = sum(go.square().sum() for _, go in pairs)
    discrete_norm = discrete_sq.sqrt()
    offset_norm = offset_sq.sqrt()
    denominator = discrete_norm * offset_norm + eps
    cosine = dot / denominator if discrete_sq > 0 and offset_sq > 0 else dot.new_zeros(())
    should_project = bool(project and dot.item() < 0 and discrete_sq.item() > 0)
    coefficient = dot / (discrete_sq + eps) if should_project else dot.new_zeros(())
    projected = [
        None if go is None else go - coefficient * gd if gd is not None else go
        for gd, go in zip(discrete, offset)
    ]
    removed_sq = sum(
        (go - gp).square().sum()
        for go, gp in zip(offset, projected)
        if go is not None and gp is not None
    )
    post_pairs = [(gd, gp) for gd, gp in zip(discrete, projected) if gd is not None and gp is not None]
    post_dot = sum((gd * gp).sum() for gd, gp in post_pairs)
    post_sq = sum(gp.square().sum() for _, gp in post_pairs)
    post_cosine = (
        post_dot / (discrete_norm * post_sq.sqrt() + eps)
        if discrete_sq.item() > 0 and post_sq.item() > 0
        else dot.new_zeros(())
    )
    removed_norm = removed_sq.sqrt()
    return projected, {
        "cosine": float(cosine.detach()),
        "post_cosine": float(post_cosine.detach()),
        "discrete_norm": float(discrete_norm.detach()),
        "offset_norm": float(offset_norm.detach()),
        "removed_norm": float(removed_norm.detach()),
        "removed_ratio": float((removed_norm / (offset_norm + eps)).detach()),
        "negative": bool(cosine.item() < 0),
        "projected": should_project,
    }


def gradient_conflict_backward(model, losses, project=False):
    groups = (
        ("encoder", tuple(model.placement_encoder.parameters())),
        ("decoder", tuple(model.placement_decoder.parameters())),
    )
    shared = tuple(parameter for _, parameters in groups for parameter in parameters)
    discrete_loss = 2.0 * (losses["region_loss"] + losses["anchor_loss"])
    offset_loss = losses["offset_loss"]
    discrete = torch.autograd.grad(discrete_loss, shared, retain_graph=True, allow_unused=True)
    offset = torch.autograd.grad(offset_loss, shared, retain_graph=True, allow_unused=True)
    projected, stats = project_conflicting_gradients(discrete, offset, project=project)
    index = 0
    for name, parameters in groups:
        count = len(parameters)
        _, group_stats = project_conflicting_gradients(
            discrete[index : index + count], offset[index : index + count], project=False
        )
        stats[f"{name}_cosine"] = group_stats["cosine"]
        index += count
    losses["loss"].backward()
    if project:
        for parameter, gd, gp in zip(shared, discrete, projected):
            if gd is None and gp is None:
                continue
            replacement = (torch.zeros_like(parameter) if gd is None else gd) + (
                torch.zeros_like(parameter) if gp is None else gp
            )
            parameter.grad = replacement.detach().clone()
    return stats


def summarize_conflict_batches(records):
    cosine = np.asarray([item["cosine"] for item in records], dtype=np.float64)
    def mean(name):
        return float(np.mean([item[name] for item in records]))
    return {
        "gradient_cosine_mean": float(cosine.mean()),
        "gradient_cosine_median": float(np.median(cosine)),
        "gradient_cosine_std": float(cosine.std()),
        "gradient_cosine_min": float(cosine.min()),
        "gradient_cosine_max": float(cosine.max()),
        "gradient_cosine_p10": float(np.quantile(cosine, 0.10)),
        "gradient_cosine_p25": float(np.quantile(cosine, 0.25)),
        "gradient_cosine_p75": float(np.quantile(cosine, 0.75)),
        "gradient_cosine_p90": float(np.quantile(cosine, 0.90)),
        "gradient_negative_rate": mean("negative"),
        "gradient_projected_rate": mean("projected"),
        "gradient_discrete_norm_mean": mean("discrete_norm"),
        "gradient_offset_norm_mean": mean("offset_norm"),
        "gradient_removed_norm_mean": mean("removed_norm"),
        "gradient_removed_ratio_mean": mean("removed_ratio"),
        "gradient_cosine_post_mean": mean("post_cosine"),
        "gradient_encoder_cosine_mean": mean("encoder_cosine"),
        "gradient_decoder_cosine_mean": mean("decoder_cosine"),
    }


@dataclass
class TrainingObjects:
    model: nn.Module
    dataset: Dataset
    loader: DataLoader
    optimizer: torch.optim.Optimizer
    coord_values: torch.Tensor
    ce: nn.Module
    bce: nn.Module
    dataset_preparation_seconds: float


def initialize_training(config, device, progress=print):
    config.validate()
    if config.variant == "A":
        raise ValueError("Variant A must use the frozen PixelWorld 0.6.1 training path")
    seed_everything(config.seed)
    model = create_model(
        config.variant, detach_placement_queries=config.detaches_placement_queries
    ).to(device)
    started = time.perf_counter()
    dataset = LandscapeDataset062(config.samples, config.offset_radius, progress)
    preparation = time.perf_counter() - started
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.device(device).type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    return TrainingObjects(
        model,
        dataset,
        loader,
        optimizer,
        torch.arange(COORD_CLASSES, dtype=torch.float32, device=device),
        nn.CrossEntropyLoss(reduction="none"),
        nn.BCEWithLogitsLoss(reduction="none"),
        preparation,
    )


def train_one_epoch(objects, config, device):
    objects.model.train()
    started = time.perf_counter()
    names = (
        "loss",
        "terrain_loss",
        "placement_loss",
        "region_loss",
        "anchor_loss",
        "offset_loss",
        "auxiliary_loss",
        "presence_loss",
        "class_loss",
        "action_loss",
        "trigger_loss",
    )
    totals = {name: 0.0 for name in names}
    batches = 0
    conflict_batches = []
    for batch in objects.loader:
        losses = compute_losses(
            objects.model,
            batch,
            config,
            device,
            objects.coord_values,
            objects.ce,
            objects.bce,
        )
        objects.optimizer.zero_grad()
        if config.measures_gradient_conflicts:
            conflict_batches.append(
                gradient_conflict_backward(
                    objects.model, losses, project=config.projects_gradient_conflicts
                )
            )
        else:
            losses["loss"].backward()
        objects.optimizer.step()
        for name in names:
            totals[name] += losses[name].item()
        batches += 1
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    result = {
        "batches": batches,
        **{name: totals[name] / max(1, batches) for name in names},
        "epoch_seconds": time.perf_counter() - started,
        "learning_rate": objects.optimizer.param_groups[0]["lr"],
    }
    if conflict_batches:
        result.update(summarize_conflict_batches(conflict_batches))
    return result


class PlacementRunStore:
    def __init__(self, repository_root, run_id):
        self.repository_root = Path(repository_root).resolve()
        self.run_id = validate_run_id(run_id)
        configured_root = self.repository_root / "outputs" / "studies" / STUDY_NAME / "runs"
        self.root, self.path = resolve_contained_run_path(configured_root, self.run_id)

    def artifact_path(self, filename):
        return resolve_run_artifact(self.root, self.run_id, filename)

    def secure_path(self):
        _, current = resolve_contained_run_path(self.root, self.run_id)
        return current

    @classmethod
    def create(cls, repository_root, config, run_id):
        store = cls(repository_root, run_id)
        store.root.mkdir(parents=True, exist_ok=True)
        path = store.secure_path()
        path.mkdir(exist_ok=False)
        atomic_json(store.artifact_path("config.json"), config.to_dict())
        store.status("queued")
        store.artifact_path("training.log").write_text("", encoding="utf-8")
        return store

    @classmethod
    def open(cls, repository_root, run_id):
        store = cls(repository_root, run_id)
        if not store.secure_path().is_dir() or not store.artifact_path("config.json").is_file():
            raise ValueError(f"Unknown 0.6.2 run ID: {run_id!r}")
        return store

    def config(self):
        return PlacementConfig.from_dict(
            json.loads(self.artifact_path("config.json").read_text(encoding="utf-8"))
        )

    def status(self, status, **details):
        atomic_json(self.artifact_path("status.json"), {"run_id": self.run_id, "status": status, **details})

    def log(self, message):
        print(message, flush=True)
        with self.artifact_path("training.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


HISTORY_FIELDS = (
    "epoch",
    "batches",
    "loss",
    "terrain_loss",
    "placement_loss",
    "region_loss",
    "anchor_loss",
    "offset_loss",
    "auxiliary_loss",
    "presence_loss",
    "class_loss",
    "action_loss",
    "trigger_loss",
    "epoch_seconds",
    "learning_rate",
) + CONFLICT_METRIC_FIELDS


def write_history(store, history):
    atomic_json(store.artifact_path("training_history.json"), history)
    temporary = store.artifact_path("training_history.csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(store.artifact_path("training_history.csv"))


def checkpoint_payload(objects, config, history, completed_epochs, timings, provenance):
    return {
        "format_version": 3,
        "pixelworld_version": config.version,
        "variant": config.variant,
        "gradient_mode": config.gradient_mode,
        "detach_placement_queries": config.detaches_placement_queries,
        "slot_latent_dim": SLOT_LATENT_DIM,
        "layout_dim": LAYOUT_DIM,
        "offset_radius": config.offset_radius,
        "completed_epochs": completed_epochs,
        "model_state_dict": objects.model.state_dict(),
        "optimizer_state_dict": objects.optimizer.state_dict(),
        "config": config.to_dict(),
        "provenance": provenance,
        "shared_target_sha256": SHARED_TARGET_SHA256,
        "evaluation_seeds": list(config.evaluation_seeds),
        "rng_state": capture_rng_state(),
        "training_history": history,
        "timings": timings,
    }


def load_checkpoint(path, config, device, with_optimizer=False):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("pixelworld_version") != config.version or payload.get("variant") != config.variant:
        raise ValueError("Checkpoint version or variant is incompatible with the requested run")
    if payload.get("slot_latent_dim") != SLOT_LATENT_DIM or payload.get("layout_dim") != LAYOUT_DIM:
        raise ValueError("Checkpoint latent schema is incompatible with explicit 8-latent slots")
    if int(payload.get("offset_radius")) != config.offset_radius:
        raise ValueError("Checkpoint offset radius is incompatible with the requested run")
    if payload.get("gradient_mode", "standard") != config.gradient_mode:
        raise ValueError("Checkpoint gradient mode is incompatible with the requested run")
    if payload.get("detach_placement_queries", False) != config.detaches_placement_queries:
        raise ValueError("Checkpoint query-detach semantics are incompatible with the requested run")
    model = create_model(
        config.variant, detach_placement_queries=config.detaches_placement_queries
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def run_training(store, device=None, resume=False, stop_after_epoch=None):
    from .evaluation import evaluate_variant

    config = store.config()
    selected_device = resolve_device(device)
    store.status("running", resumed=resume)
    store.log(f"Version {config.version}, Variante {config.variant}, Device {selected_device}")
    if selected_device.type == "cuda":
        store.log(f"GPU: {torch.cuda.get_device_name(selected_device)}")
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(selected_device)
    wall_started = time.perf_counter()
    try:
        objects = initialize_training(config, selected_device, store.log)
        provenance = environment_provenance(
            store.repository_root,
            selected_device,
            sum(parameter.numel() for parameter in objects.model.parameters()),
        )
        provenance.update(
            {
                "pixelworld_version": config.version,
                "variant": config.variant,
                "placement_config": config.to_dict(),
                "shared_target_sha256": SHARED_TARGET_SHA256,
                "evaluation_seeds": list(config.evaluation_seeds),
            }
        )
        history = []
        start_epoch = 0
        prior_total = 0.0
        prior_training = 0.0
        preparation = objects.dataset_preparation_seconds
        if resume:
            latest = store.artifact_path("latest.pt")
            if not latest.is_file():
                raise ValueError(f"Run {store.run_id!r} has no latest.pt")
            payload = torch.load(latest, map_location=selected_device, weights_only=True)
            if payload.get("variant") != config.variant or payload.get("pixelworld_version") != config.version:
                raise ValueError("Incompatible recovery checkpoint")
            if payload.get("gradient_mode", "standard") != config.gradient_mode:
                raise ValueError("Incompatible recovery checkpoint gradient mode")
            if payload.get("detach_placement_queries", False) != config.detaches_placement_queries:
                raise ValueError("Incompatible recovery checkpoint query-detach semantics")
            if payload.get("slot_latent_dim") != SLOT_LATENT_DIM or payload.get("layout_dim") != LAYOUT_DIM:
                raise ValueError("Incompatible recovery checkpoint latent schema")
            objects.model.load_state_dict(payload["model_state_dict"])
            objects.optimizer.load_state_dict(payload["optimizer_state_dict"])
            history = list(payload["training_history"])
            start_epoch = int(payload["completed_epochs"])
            prior_total = float(payload["timings"]["total_seconds"])
            prior_training = float(payload["timings"]["training_seconds"])
            preparation += float(payload["timings"]["dataset_preparation_seconds"])
            restore_rng_state(payload["rng_state"])
        training_started = time.perf_counter()
        for epoch in range(start_epoch, config.epochs):
            record = {"epoch": epoch + 1, **train_one_epoch(objects, config, selected_device)}
            history.append(record)
            timings = {
                "dataset_preparation_seconds": preparation,
                "training_seconds": prior_training + time.perf_counter() - training_started,
                "total_seconds": prior_total + time.perf_counter() - wall_started,
                "epoch_seconds": [item["epoch_seconds"] for item in history],
            }
            atomic_torch_save(
                store.artifact_path("latest.pt"),
                checkpoint_payload(objects, config, history, epoch + 1, timings, provenance),
            )
            write_history(store, history)
            store.log(
                f"Epoch {epoch + 1:02d}: loss={record['loss']:.3f} "
                f"placement={record['placement_loss']:.3f} offset={record['offset_loss']:.3f}"
            )
            if stop_after_epoch and epoch + 1 >= stop_after_epoch:
                store.status("aborted", completed_epochs=epoch + 1)
                return {"status": "aborted", "completed_epochs": epoch + 1}
        timings = {
            "dataset_preparation_seconds": preparation,
            "training_seconds": prior_training + time.perf_counter() - training_started,
            "total_seconds": prior_total + time.perf_counter() - wall_started,
            "epoch_seconds": [item["epoch_seconds"] for item in history],
        }
        payload = checkpoint_payload(objects, config, history, config.epochs, timings, provenance)
        atomic_torch_save(store.artifact_path("final.pt"), payload)
        loaded_model, _ = load_checkpoint(store.artifact_path("final.pt"), config, selected_device)
        metrics, diagnostics = evaluate_variant(loaded_model, config, selected_device)
        atomic_json(
            store.artifact_path("evaluation_metrics.json"),
            {"metrics": metrics, "diagnostics": diagnostics, "checkpoint_reloaded": True},
        )
        payload["evaluation_metrics"] = metrics
        payload["placement_diagnostics"] = diagnostics
        atomic_torch_save(store.artifact_path("final.pt"), payload)
        reloaded, _ = load_checkpoint(store.artifact_path("final.pt"), config, selected_device)
        verification_metrics, _ = evaluate_variant(reloaded, config, selected_device)
        if verification_metrics != metrics:
            raise RuntimeError("Evaluation changed after checkpoint reload")
        summary = {
            "run_id": store.run_id,
            "status": "completed",
            "config": config.to_dict(),
            "timings": timings,
            "final_losses": history[-1],
            "metrics": metrics,
            "diagnostics": diagnostics,
            "provenance": provenance,
            "shared_target_sha256": SHARED_TARGET_SHA256,
            "evaluation_seeds": list(config.evaluation_seeds),
            "checkpoint_sha256": checkpoint_sha256(store.artifact_path("final.pt")),
            "checkpoint_reload_verified": True,
            "device": str(selected_device),
            "gpu": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(selected_device) if selected_device.type == "cuda" else 0,
        }
        atomic_json(store.artifact_path("run_summary.json"), summary)
        store.status("completed", completed_epochs=config.epochs)
        return summary
    except Exception as error:
        store.status("failed", error=f"{type(error).__name__}: {error}")
        raise
