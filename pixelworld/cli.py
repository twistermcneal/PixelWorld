import argparse
import json
import sys
import time
from pathlib import Path

from .artifacts import RunStore, atomic_json, list_runs, resolve_output_subdirectory
from .config import RunConfig
from .evaluation import evaluate_model
from .golden import compare_run_to_oracle
from .inference import load_model, predict, prediction_to_dict
from .training import resolve_device, run_training


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def print_device(device):
    import torch

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(prog="pixelworld")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="start a PixelWorld training run")
    train.add_argument("--version", default="0.6.1")
    train.add_argument("--samples", type=int, default=14_000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--epochs", type=int, default=45)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--variant", choices=("A", "B", "C", "D", "E"))
    train.add_argument(
        "--gradient-mode",
        choices=(
            "standard", "measure", "pcgrad", "qdet-measure", "qdet-pcgrad",
            "split-measure",
        ),
        default="standard",
        help="D-only gradient conflict ablation mode",
    )
    train.add_argument("--offset-radius", type=int, choices=(8, 12, 16), default=8)
    train.add_argument("--run-id")
    train.add_argument("--device", choices=("cpu", "cuda"))
    train.add_argument("--stop-after-epoch", type=int, help=argparse.SUPPRESS)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved run")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"))

    infer = subparsers.add_parser("infer", help="run one prediction")
    infer.add_argument("--run", required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--seed", required=True, type=int)
    infer.add_argument("--device", choices=("cpu", "cuda"))

    subparsers.add_parser("runs", help="list local runs")

    resume = subparsers.add_parser("resume", help="resume an interrupted run")
    resume.add_argument("--run", required=True)
    resume.add_argument("--device", choices=("cpu", "cuda"))

    golden = subparsers.add_parser("golden", help="compare a run with the Seed-42 oracle")
    golden.add_argument("--run", required=True)
    golden.add_argument("--oracle", default="outputs/0.6.1-reference")

    study = subparsers.add_parser("study-placement", help="run the PixelWorld 0.6.2 placement study")
    study.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    study.add_argument("--variants", nargs="+", choices=("A", "B", "C", "D", "E"), default=list("ABCDE"))
    study.add_argument("--samples", type=int, default=14_000)
    study.add_argument("--batch-size", type=int, default=128)
    study.add_argument("--epochs", type=int, default=45)
    study.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def command_train(args):
    if args.version == "0.6.2":
        from .versions.v0_6_2.config import PlacementConfig
        from .versions.v0_6_2.training import PlacementRunStore, run_training as run_training_062

        variant = args.variant or "B"
        config = PlacementConfig(
            variant=variant,
            samples=args.samples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
            learning_rate=args.learning_rate,
            offset_radius=args.offset_radius,
            gradient_mode=args.gradient_mode,
        ).validate()
        if variant == "A":
            raise ValueError("Variant A uses the frozen 0.6.1 path and is managed by study-placement")
        mode_suffix = "" if args.gradient_mode == "standard" else f"-{args.gradient_mode.upper()}"
        run_id = args.run_id or f"v062-{variant}{mode_suffix}-seed{args.seed}"
        store = PlacementRunStore.create(REPOSITORY_ROOT, config, run_id)
        print(f"Run ID: {store.run_id}", flush=True)
        result = run_training_062(
            store,
            device=args.device,
            stop_after_epoch=args.stop_after_epoch,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    config = RunConfig(
        version=args.version,
        samples=args.samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
    ).validate()
    store = RunStore.create(REPOSITORY_ROOT, config, run_id=args.run_id)
    print(f"Run ID: {store.run_id}", flush=True)
    result = run_training(
        store,
        device=args.device,
        stop_after_epoch=args.stop_after_epoch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_evaluate(args):
    store = RunStore.open(REPOSITORY_ROOT, args.run)
    config = store.config()
    device = resolve_device(args.device)
    print_device(device)
    checkpoint = store.checkpoint_path(final=True)
    if not checkpoint.is_file():
        checkpoint = store.checkpoint_path(final=False)
    model, _ = load_model(checkpoint, device)
    started = time.perf_counter()
    metrics = evaluate_model(model, device, eval_seeds=config.evaluation_seeds)
    document = {
        "evaluation_seeds": list(config.evaluation_seeds),
        "metrics": metrics,
        "evaluation_seconds": time.perf_counter() - started,
        "checkpoint_reloaded": True,
    }
    atomic_json(store.path / "evaluation_metrics.json", document)
    print(json.dumps(document, ensure_ascii=False, indent=2))


def command_infer(args):
    store = RunStore.open(REPOSITORY_ROOT, args.run)
    device = resolve_device(args.device)
    print_device(device)
    checkpoint = store.checkpoint_path(final=True)
    if not checkpoint.is_file():
        checkpoint = store.checkpoint_path(final=False)
    model, _ = load_model(checkpoint, device)
    result = prediction_to_dict(predict(model, args.prompt, args.seed, device))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_runs(_args):
    print(json.dumps(list_runs(REPOSITORY_ROOT), ensure_ascii=False, indent=2))


def command_resume(args):
    from .versions.v0_6_2.training import PlacementRunStore, run_training as run_training_062

    placement_store = PlacementRunStore(REPOSITORY_ROOT, args.run)
    if (placement_store.path / "config.json").is_file():
        result = run_training_062(placement_store, device=args.device, resume=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    store = RunStore.open(REPOSITORY_ROOT, args.run)
    status = json.loads((store.path / "status.json").read_text(encoding="utf-8"))["status"]
    if status == "completed":
        raise ValueError(f"Run {store.run_id!r} is already completed")
    result = run_training(store, device=args.device, resume=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_golden(args):
    store = RunStore.open(REPOSITORY_ROOT, args.run)
    oracle = resolve_output_subdirectory(REPOSITORY_ROOT, args.oracle)
    result = compare_run_to_oracle(store.path, oracle)
    atomic_json(store.path / "golden_comparison.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise RuntimeError("Golden parity failed")


def command_study_placement(args):
    from .versions.v0_6_2.study import run_study

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    result = run_study(
        REPOSITORY_ROOT,
        seeds=tuple(args.seeds),
        variants=tuple(args.variants),
        samples=args.samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "train": command_train,
        "evaluate": command_evaluate,
        "infer": command_infer,
        "runs": command_runs,
        "resume": command_resume,
        "golden": command_golden,
        "study-placement": command_study_placement,
    }
    try:
        handlers[args.command](args)
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
