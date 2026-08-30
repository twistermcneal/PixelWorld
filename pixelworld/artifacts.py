import csv
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import RunConfig


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
VALID_STATUSES = {"queued", "running", "completed", "failed", "aborted"}
HISTORY_FIELDS = (
    "epoch",
    "batches",
    "loss",
    "terrain_loss",
    "placement_loss",
    "presence_loss",
    "class_loss",
    "action_loss",
    "trigger_loss",
    "epoch_seconds",
    "learning_rate",
)


def validate_run_id(run_id: str) -> str:
    windows_stem = run_id.split(".", 1)[0].upper()
    if (
        not RUN_ID_PATTERN.fullmatch(run_id)
        or run_id.endswith(".")
        or windows_stem in WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(f"Invalid run ID: {run_id!r}")
    return run_id


def make_run_id(seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"pw061-{timestamp}-seed{seed}-{secrets.token_hex(4)}"


def resolve_output_subdirectory(repository_root: Path, requested_path: str | Path) -> Path:
    repository_root = repository_root.resolve(strict=True)
    outputs_root = (repository_root / "outputs").resolve(strict=True)
    requested = Path(requested_path)
    candidate = requested if requested.is_absolute() else repository_root / requested
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(outputs_root)
    except ValueError as error:
        raise ValueError("Oracle path must be inside the repository outputs directory") from error
    if relative == Path(".") or not resolved.is_dir():
        raise ValueError("Oracle path must name a subdirectory of the repository outputs directory")
    return resolved


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _validate_atomic_paths(path, temporary)
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _validate_atomic_paths(path, temporary)
    torch.save(data, temporary)
    os.replace(temporary, path)


def atomic_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _validate_atomic_paths(path, temporary)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def _validate_atomic_paths(path: Path, temporary: Path) -> None:
    canonical_parent = path.parent.resolve()
    if path.resolve().parent != canonical_parent or temporary.resolve().parent != canonical_parent:
        raise ValueError("Atomic artifact path escapes its configured directory")


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_provenance(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    commit = git_commit(root)
    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
        dirty = bool(porcelain.strip())
    except (OSError, subprocess.SubprocessError):
        branch = None
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty, "git_branch": branch}


def environment_provenance(root: Path, device, model_parameters: int) -> dict[str, Any]:
    device_object = torch.device(device)
    document = {
        **git_provenance(root),
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device_object),
        "gpu_model": (
            torch.cuda.get_device_name(device_object)
            if device_object.type == "cuda"
            else None
        ),
        "model_parameters": int(model_parameters),
    }
    if device_object.type == "cuda":
        properties = torch.cuda.get_device_properties(device_object)
        document.update(
            {
                "gpu_compute_capability": [properties.major, properties.minor],
                "gpu_total_vram_bytes": properties.total_memory,
            }
        )
    return document


def resolve_contained_run_path(runs_root: Path, run_id: str) -> tuple[Path, Path]:
    """Resolve a validated run ID without permitting symlink or junction escapes."""
    validate_run_id(run_id)
    logical_root = Path(runs_root).absolute()
    canonical_parent = logical_root.parent.resolve()
    canonical_root = logical_root.resolve()
    if canonical_root.parent != canonical_parent:
        raise ValueError("Runs directory escapes its configured study directory")
    candidate = (canonical_root / run_id).resolve()
    if candidate.parent != canonical_root:
        raise ValueError("Run path escapes its configured runs directory")
    return canonical_root, candidate


def resolve_run_artifact(runs_root: Path, run_id: str, filename: str) -> Path:
    """Revalidate the run directory and one direct artifact immediately before I/O."""
    if not filename or Path(filename).name != filename:
        raise ValueError(f"Invalid run artifact name: {filename!r}")
    canonical_root, run_path = resolve_contained_run_path(runs_root, run_id)
    if run_path.parent != canonical_root:
        raise ValueError("Run path escapes its configured runs directory")
    artifact = (run_path / filename).resolve()
    if artifact.parent != run_path:
        raise ValueError("Run artifact path escapes its run directory")
    return artifact


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state = {
        "python": __import__("random").getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    __import__("random").setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class RunStore:
    def __init__(self, repository_root: Path, run_id: str):
        self.repository_root = repository_root.resolve()
        self.runs_root = (self.repository_root / "outputs" / "runs").resolve()
        self.run_id = validate_run_id(run_id)
        self.path = (self.runs_root / self.run_id).resolve()
        if self.path.parent != self.runs_root:
            raise ValueError("Run path escapes outputs/runs")

    @classmethod
    def create(cls, repository_root: Path, config: RunConfig, run_id: str | None = None):
        selected = run_id or make_run_id(config.seed)
        store = cls(repository_root, selected)
        store.runs_root.mkdir(parents=True, exist_ok=True)
        try:
            store.path.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise ValueError(f"Run already exists: {selected}") from error
        atomic_json(store.path / "config.json", config.to_dict())
        store.set_status("queued")
        (store.path / "training.log").write_text("", encoding="utf-8")
        return store

    @classmethod
    def open(cls, repository_root: Path, run_id: str):
        store = cls(repository_root, run_id)
        if not store.path.is_dir() or not (store.path / "config.json").is_file():
            raise ValueError(f"Unknown run ID: {run_id}")
        return store

    def config(self) -> RunConfig:
        return RunConfig.from_dict(json.loads((self.path / "config.json").read_text(encoding="utf-8")))

    def set_status(self, status: str, **details) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid run status: {status}")
        document = {
            "run_id": self.run_id,
            "status": status,
            "updated_at": datetime.now().astimezone().isoformat(),
            **details,
        }
        atomic_json(self.path / "status.json", document)

    def log(self, message: str) -> None:
        line = f"{datetime.now().astimezone().isoformat()} {message}"
        print(message, flush=True)
        with (self.path / "training.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_history(self, history: list[dict[str, Any]]) -> None:
        atomic_json(self.path / "training_history.json", history)
        atomic_history_csv(self.path / "training_history.csv", history)

    def checkpoint_path(self, final: bool = False) -> Path:
        return self.path / ("final.pt" if final else "latest.pt")

    def write_checkpoint(self, payload: dict[str, Any], final: bool = False) -> Path:
        path = self.checkpoint_path(final)
        atomic_torch_save(path, payload)
        return path

    def environment(self, device, model_parameters: int) -> dict[str, Any]:
        device_object = torch.device(device)
        environment = environment_provenance(
            self.repository_root, device_object, model_parameters
        )
        environment["cuda_version"] = environment["cuda_runtime"]
        environment["gpu"] = environment["gpu_model"]
        if device_object.type == "cuda":
            properties = torch.cuda.get_device_properties(device_object)
            environment.update(
                {
                    "gpu_compute_capability": [properties.major, properties.minor],
                    "gpu_total_vram_bytes": properties.total_memory,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device_object),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device_object),
                }
            )
        return environment


def list_runs(repository_root: Path) -> list[dict[str, Any]]:
    root = repository_root / "outputs" / "runs"
    if not root.is_dir():
        return []
    results = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            validate_run_id(directory.name)
            status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            results.append({"run_id": directory.name, "status": status["status"], "config": config})
        except (ValueError, OSError, json.JSONDecodeError, KeyError):
            continue
    return results
