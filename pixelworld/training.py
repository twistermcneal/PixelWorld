import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .config import (
    ACTIONS,
    BIOMES,
    COORD_CLASSES,
    LANDMARK_CLASSES,
    MAX_SLOTS,
    SIZE,
    TRIGGER_TYPES,
    RunConfig,
)
from .generation import generate_landscape
from .model import LandscapeNet, condition_vector


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scene_targets(world):
    biome, orientation, shore, width, rock, forest, density = world.terrain_params
    regions = np.zeros(MAX_SLOTS, np.int64)
    anchors = np.zeros(MAX_SLOTS, np.int64)
    presence = np.zeros(MAX_SLOTS, np.float32)
    classes = np.zeros(MAX_SLOTS, np.int64)
    actions = np.zeros(MAX_SLOTS, np.int64)
    triggers = np.zeros(MAX_SLOTS, np.int64)
    for slot in range(MAX_SLOTS):
        oid = slot + 1
        if oid in world.objects:
            metadata = world.objects[oid]
            presence[slot] = 1
            regions[slot] = metadata["region_id"]
            anchors[slot] = metadata["anchor_id"]
            classes[slot] = LANDMARK_CLASSES.index(metadata["class"])
            actions[slot] = ACTIONS.index(metadata["action"])
            triggers[slot] = TRIGGER_TYPES.index(metadata["trigger_type"])
    return (
        np.asarray([shore, width, rock, forest, density]),
        orientation,
        biome,
        regions,
        anchors,
        presence,
        classes,
        actions,
        triggers,
    )


class LandscapeDataset(Dataset):
    def __init__(self, n=14_000, progress: Callable[[str], None] | None = print):
        self.samples = []
        if progress:
            progress(f"Erzeuge {n:,} Trainingslandschaften einmalig ...")
        for i in range(n):
            prompt = f"{BIOMES[i % 4]} coast beach forest rock portal {i}"
            seed = i + 1000
            targets = scene_targets(generate_landscape(prompt, seed))
            self.samples.append(
                tuple(torch.tensor(value) for value in (condition_vector(prompt, seed), *targets))
            )
            if progress and (i + 1) % 2000 == 0:
                progress(f"  {i + 1:,}/{n:,}")
        if progress:
            progress("Datensatz bereit.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def ordinal_loss(logits, target, coord_values, sigma=1.):
    d = coord_values - target[..., None].float()
    soft = torch.exp(-.5 * (d / sigma) ** 2)
    soft /= soft.sum(-1, keepdim=True)
    expected = (logits.softmax(-1) * coord_values).sum(-1)
    return -(soft * logits.log_softmax(-1)).sum(-1) + 4 * F.smooth_l1_loss(
        expected, target.float(), reduction="none"
    ) / SIZE


def masked_ce(logits, target, presence, ce):
    errors = ce(logits.flatten(0, 1), target.flatten()).reshape_as(target)
    return (errors * presence).sum() / presence.sum().clamp_min(1)


def create_loss_objects(device):
    return (
        torch.arange(COORD_CLASSES, dtype=torch.float32, device=device),
        nn.CrossEntropyLoss(reduction="none"),
        nn.BCEWithLogitsLoss(reduction="none"),
    )


def compute_losses(model, batch, device, coord_values, ce, bce):
    condition, numeric, orient, biome, regions, anchors, presence, classes, actions, triggers = [
        tensor.to(device, non_blocking=True) for tensor in batch
    ]
    outputs = model(condition)
    num_l, orient_l, biome_l, region_l, anchor_l, pres_l, cls_l, act_l, trig_l = outputs
    terrain_loss = (
        ordinal_loss(num_l, numeric, coord_values).mean()
        + ce(orient_l, orient).mean()
        + ce(biome_l, biome).mean()
    )
    placement_loss = masked_ce(region_l, regions, presence, ce) + masked_ce(
        anchor_l, anchors, presence, ce
    )
    presence_loss = (bce(pres_l, presence) * torch.where(presence > .5, 1., 2.)).mean()
    class_loss = masked_ce(cls_l, classes, presence, ce)
    action_loss = masked_ce(act_l, actions, presence, ce)
    trigger_loss = masked_ce(trig_l, triggers, presence, ce)
    loss = (
        terrain_loss
        + 2 * placement_loss
        + presence_loss
        + class_loss
        + action_loss
        + trigger_loss
    )
    return loss, terrain_loss, placement_loss, presence_loss, class_loss, action_loss, trigger_loss


@dataclass
class TrainingObjects:
    model: LandscapeNet
    dataset: LandscapeDataset
    loader: DataLoader
    optimizer: torch.optim.Optimizer
    coord_values: torch.Tensor
    ce: nn.Module
    bce: nn.Module
    dataset_preparation_seconds: float


def initialize_training(config: RunConfig, device, progress=print) -> TrainingObjects:
    config.validate()
    seed_everything(config.seed)
    model = LandscapeNet().to(device)
    dataset_started = time.perf_counter()
    dataset = LandscapeDataset(config.samples, progress=progress)
    dataset_preparation_seconds = time.perf_counter() - dataset_started
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(torch.device(device).type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    coord_values, ce, bce = create_loss_objects(device)
    return TrainingObjects(
        model,
        dataset,
        loader,
        optimizer,
        coord_values,
        ce,
        bce,
        dataset_preparation_seconds,
    )


def train_one_epoch(objects: TrainingObjects, device) -> dict[str, float]:
    started = time.perf_counter()
    objects.model.train()
    totals = np.zeros(8)
    batches = 0
    for batch in objects.loader:
        losses = compute_losses(
            objects.model, batch, device, objects.coord_values, objects.ce, objects.bce
        )
        loss = losses[0]
        objects.optimizer.zero_grad()
        loss.backward()
        objects.optimizer.step()
        totals += [*(value.item() for value in losses), 1]
        batches += 1
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    values = totals / max(1, batches)
    return {
        "batches": batches,
        "loss": float(values[0]),
        "terrain_loss": float(values[1]),
        "placement_loss": float(values[2]),
        "presence_loss": float(values[3]),
        "class_loss": float(values[4]),
        "action_loss": float(values[5]),
        "trigger_loss": float(values[6]),
        "epoch_seconds": time.perf_counter() - started,
        "learning_rate": objects.optimizer.param_groups[0]["lr"],
    }


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_payload(objects, config, history, completed_epochs, environment, timings):
    from .artifacts import capture_rng_state

    return {
        "format_version": 1,
        "pixelworld_version": config.version,
        "completed_epochs": completed_epochs,
        "model_state_dict": objects.model.state_dict(),
        "optimizer_state_dict": objects.optimizer.state_dict(),
        "config": config.to_dict(),
        "rng_state": capture_rng_state(),
        "training_history": history,
        "environment": environment,
        "timings": timings,
    }


def run_training(store, device=None, resume=False, stop_after_epoch=None):
    from .artifacts import checkpoint_sha256, restore_rng_state
    from .evaluation import evaluate_model
    from .inference import load_model, predict

    selected_device = resolve_device(str(device) if device is not None else None)
    config = store.config()
    store.set_status("running", resumed=bool(resume))
    store.log(f"Device: {selected_device}")
    if selected_device.type == "cuda":
        store.log(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(selected_device)
    started = time.perf_counter()
    try:
        objects = initialize_training(config, selected_device, progress=store.log)
        environment = store.environment(
            selected_device, sum(parameter.numel() for parameter in objects.model.parameters())
        )
        history = []
        start_epoch = 0
        prior_training_seconds = 0.0
        prior_total_seconds = 0.0
        dataset_preparation_seconds = objects.dataset_preparation_seconds
        if resume:
            latest = store.checkpoint_path(final=False)
            if not latest.is_file():
                raise ValueError(f"Run {store.run_id!r} has no latest.pt checkpoint")
            payload = torch.load(latest, map_location=selected_device, weights_only=True)
            objects.model.load_state_dict(payload["model_state_dict"])
            objects.optimizer.load_state_dict(payload["optimizer_state_dict"])
            history = list(payload["training_history"])
            start_epoch = int(payload["completed_epochs"])
            prior_training_seconds = float(payload["timings"]["training_seconds"])
            prior_total_seconds = float(payload["timings"]["total_seconds"])
            dataset_preparation_seconds += float(
                payload["timings"]["dataset_preparation_seconds"]
            )
            restore_rng_state(payload["rng_state"])
            store.log(f"Resume after epoch {start_epoch}")
        training_started = time.perf_counter()
        for epoch in range(start_epoch, config.epochs):
            record = train_one_epoch(objects, selected_device)
            record = {"epoch": epoch + 1, **record}
            history.append(record)
            training_seconds = prior_training_seconds + (time.perf_counter() - training_started)
            timings = {
                "dataset_preparation_seconds": dataset_preparation_seconds,
                "training_seconds": training_seconds,
                "total_seconds": prior_total_seconds + (time.perf_counter() - started),
                "epoch_seconds": [item["epoch_seconds"] for item in history],
            }
            store.write_checkpoint(
                _checkpoint_payload(objects, config, history, epoch + 1, environment, timings)
            )
            store.write_history(history)
            store.log(
                f"Epoch {epoch + 1:02d}: loss={record['loss']:.3f} "
                f"terrain={record['terrain_loss']:.3f} placement={record['placement_loss']:.3f} "
                f"presence={record['presence_loss']:.3f} class={record['class_loss']:.3f} "
                f"action={record['action_loss']:.3f} trigger={record['trigger_loss']:.3f}"
            )
            if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
                store.set_status("aborted", completed_epochs=epoch + 1, reason="requested stop")
                return {"status": "aborted", "run_id": store.run_id, "completed_epochs": epoch + 1}

        training_seconds = prior_training_seconds + (time.perf_counter() - training_started)
        if selected_device.type == "cuda":
            environment.update(
                {
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(selected_device),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(selected_device),
                }
            )
        timings = {
            "dataset_preparation_seconds": dataset_preparation_seconds,
            "training_seconds": training_seconds,
            "total_seconds": prior_total_seconds + (time.perf_counter() - started),
            "epoch_seconds": [item["epoch_seconds"] for item in history],
        }
        final_payload = _checkpoint_payload(
            objects, config, history, config.epochs, environment, timings
        )
        store.write_checkpoint(final_payload, final=True)

        loaded_model, loaded_payload = load_model(store.checkpoint_path(final=True), selected_device)
        evaluation_started = time.perf_counter()
        metrics = evaluate_model(
            loaded_model, selected_device, eval_seeds=config.evaluation_seeds
        )
        if selected_device.type == "cuda":
            torch.cuda.synchronize()
        evaluation_seconds = time.perf_counter() - evaluation_started
        inference_started = time.perf_counter()
        predict(loaded_model, "tropical coast beach forest rock portal", 500000, selected_device)
        if selected_device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        metrics_document = {
            "evaluation_seeds": list(config.evaluation_seeds),
            "metrics": metrics,
            "evaluation_seconds": evaluation_seconds,
            "checkpoint_reloaded": True,
            "single_inference_seconds": inference_seconds,
        }
        from .artifacts import atomic_json

        atomic_json(store.path / "evaluation_metrics.json", metrics_document)
        final_payload["evaluation_metrics"] = metrics
        final_payload["timings"].update(
            {
                "evaluation_seconds": evaluation_seconds,
                "single_inference_seconds": inference_seconds,
                "total_seconds": prior_total_seconds + (time.perf_counter() - started),
            }
        )
        store.write_checkpoint(final_payload, final=True)
        verification_model, _ = load_model(store.checkpoint_path(final=True), selected_device)
        verification_metrics = evaluate_model(
            verification_model, selected_device, eval_seeds=config.evaluation_seeds
        )
        if verification_metrics != metrics:
            raise RuntimeError("Evaluation changed after final checkpoint reload")
        final_hash = checkpoint_sha256(store.checkpoint_path(final=True))
        summary = {
            "run_id": store.run_id,
            "status": "completed",
            "config": config.to_dict(),
            "environment": environment,
            "timings": final_payload["timings"],
            "training_history_epochs": len(history),
            "final_losses": history[-1],
            "evaluation_metrics": metrics,
            "final_checkpoint_sha256": final_hash,
            "checkpoint_reload_verified": True,
            "single_inference_verified": True,
        }
        atomic_json(store.path / "run_summary.json", summary)
        store.set_status("completed", completed_epochs=config.epochs, final_checkpoint_sha256=final_hash)
        return summary
    except KeyboardInterrupt:
        store.set_status("aborted", reason="KeyboardInterrupt")
        raise
    except Exception as error:
        store.set_status("failed", error=f"{type(error).__name__}: {error}")
        store.log(f"FAILED: {type(error).__name__}: {error}")
        raise
