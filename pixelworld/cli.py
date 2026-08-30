import argparse
import json
import sys
from pathlib import Path

from .artifacts import RunStore, atomic_json, list_runs
from .config import RunConfig
from .evaluation import evaluate_model
from .golden import compare_run_to_oracle
from .inference import load_model, predict, prediction_to_dict
from .training import resolve_device, run_training


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
    return parser


def command_train(args):
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
    checkpoint = store.checkpoint_path(final=True)
    if not checkpoint.is_file():
        checkpoint = store.checkpoint_path(final=False)
    model, _ = load_model(checkpoint, device)
    metrics = evaluate_model(model, device, eval_seeds=config.evaluation_seeds)
    document = {"evaluation_seeds": list(config.evaluation_seeds), "metrics": metrics, "checkpoint_reloaded": True}
    atomic_json(store.path / "evaluation_metrics.json", document)
    print(json.dumps(document, ensure_ascii=False, indent=2))


def command_infer(args):
    store = RunStore.open(REPOSITORY_ROOT, args.run)
    device = resolve_device(args.device)
    checkpoint = store.checkpoint_path(final=True)
    if not checkpoint.is_file():
        checkpoint = store.checkpoint_path(final=False)
    model, _ = load_model(checkpoint, device)
    result = prediction_to_dict(predict(model, args.prompt, args.seed, device))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_runs(_args):
    print(json.dumps(list_runs(REPOSITORY_ROOT), ensure_ascii=False, indent=2))


def command_resume(args):
    store = RunStore.open(REPOSITORY_ROOT, args.run)
    status = json.loads((store.path / "status.json").read_text(encoding="utf-8"))["status"]
    if status == "completed":
        raise ValueError(f"Run {store.run_id!r} is already completed")
    result = run_training(store, device=args.device, resume=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_golden(args):
    store = RunStore.open(REPOSITORY_ROOT, args.run)
    oracle = (REPOSITORY_ROOT / args.oracle).resolve()
    result = compare_run_to_oracle(store.path, oracle)
    atomic_json(store.path / "golden_comparison.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise RuntimeError("Golden parity failed")


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
    }
    try:
        handlers[args.command](args)
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
