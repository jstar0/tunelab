from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .objective import IrisObjective, SurfaceObjective, WineObjective
from .report import write_benchmark_index, write_report
from .runner import InjectedFailure, run_experiment
from .sweep import DEFAULT_DATA_SEED, parse_seed_list, write_sweep


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _seed_list(value: str) -> tuple[int, ...]:
    try:
        return parse_seed_list(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _objective(name: str, *, seed: int, folds: int) -> tuple[Any, list[Any], str]:
    if name == "surface":
        objective = SurfaceObjective()
    elif name == "iris":
        objective = IrisObjective(seed=seed, folds=folds)
    elif name == "wine":
        objective = WineObjective(seed=seed, folds=folds)
    else:  # argparse choices should make this unreachable
        raise ValueError(f"Unsupported objective: {name}")
    return objective, list(objective.specs), objective.name


def _run_one(
    *,
    objective_name: str,
    algorithm: str,
    trials: int,
    seed: int,
    folds: int,
    run_id: str,
    output_dir: Path,
    checkpoint: Path | None,
    fail_after: int | None = None,
) -> dict[str, Any]:
    objective, specs, canonical_name = _objective(objective_name, seed=seed, folds=folds)
    result = run_experiment(
        run_id=run_id,
        objective=objective,
        objective_name=canonical_name,
        specs=specs,
        algorithm=algorithm,  # type: ignore[arg-type]
        max_trials=trials,
        seed=seed,
        checkpoint_path=checkpoint,
        fail_after=fail_after,
    )
    artifacts = write_report(result, output_dir)
    return {
        "run_id": result.run_id,
        "objective": result.objective,
        "algorithm": result.algorithm,
        "seed": result.seed,
        "trials": len(result.trials),
        "completed_trials": len(result.completed_trials),
        "failed_trials": len(result.failed_trials),
        "best_score": result.best_score,
        "artifacts": artifacts,
    }


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--objective", choices=("surface", "iris", "wine"), default="surface")
    parser.add_argument("--algorithm", choices=("grid", "random", "tpe", "genetic"), default="tpe")
    parser.add_argument("--trials", type=_positive_int, default=16, help="Maximum number of objective evaluations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=_positive_int, default=3, help="Public-data cross-validation folds")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--checkpoint", type=Path, default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tunelab",
        description="Run transparent, reproducible, resumable hyperparameter experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one objective with one search strategy")
    _add_common_run_args(run_parser)
    run_parser.add_argument(
        "--fail-after",
        type=_positive_int,
        default=None,
        help="Inject a failure after N new trials (for resume demonstrations)",
    )

    demo_parser = subparsers.add_parser("demo", help="Generate a fast four-strategy comparison")
    demo_parser.add_argument("--trials", type=_positive_int, default=48)
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))

    recovery_parser = subparsers.add_parser(
        "recovery-demo", help="Run an intentional interruption then resume the same experiment"
    )
    recovery_parser.add_argument("--trials", type=_positive_int, default=8)
    recovery_parser.add_argument("--seed", type=int, default=17)
    recovery_parser.add_argument("--fail-after", type=_positive_int, default=3)
    recovery_parser.add_argument("--output-dir", type=Path, default=Path("outputs/recovery-demo"))

    benchmark_parser = subparsers.add_parser("benchmark", help="Compare all strategies on a public dataset")
    benchmark_parser.add_argument("--objective", choices=("iris", "wine"), default="wine")
    benchmark_parser.add_argument("--trials", type=_positive_int, default=8)
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument("--folds", type=_positive_int, default=3)
    benchmark_parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))

    sweep_parser = subparsers.add_parser(
        "sweep", help="Measure public-dataset strategy stability over explicit seeds"
    )
    sweep_parser.add_argument("--objective", choices=("iris", "wine"), default="wine")
    sweep_parser.add_argument("--seeds", type=_seed_list, default=(17, 29, 41))
    sweep_parser.add_argument(
        "--data-seed", type=int, default=DEFAULT_DATA_SEED,
        help="Fixed data split and model seed; search seeds vary independently",
    )
    sweep_parser.add_argument("--trials", type=_positive_int, default=8)
    sweep_parser.add_argument("--folds", type=_positive_int, default=3)
    sweep_parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweep"))

    serve_parser = subparsers.add_parser("serve", help="Serve the local TuneLab workbench")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=_positive_int, default=5202)
    serve_parser.add_argument("--runtime-dir", type=Path, default=Path("runtime/tunelab"))

    return parser


def _command_run(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    checkpoint = args.checkpoint or output_dir / "checkpoint.json"
    run_id = args.run_id or f"{args.objective}-{args.algorithm}-seed{args.seed}"
    try:
        summary = _run_one(
            objective_name=args.objective,
            algorithm=args.algorithm,
            trials=args.trials,
            seed=args.seed,
            folds=args.folds,
            run_id=run_id,
            output_dir=output_dir,
            checkpoint=checkpoint,
            fail_after=args.fail_after,
        )
    except InjectedFailure as exc:
        print(json.dumps({"status": "injected_failure", "message": str(exc), "checkpoint": str(checkpoint)}))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _command_demo(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    results = []
    summaries = []
    for algorithm in ("grid", "random", "tpe", "genetic"):
        run_dir = output_dir / algorithm
        checkpoint = run_dir / "checkpoint.json"
        objective, specs, canonical_name = _objective("surface", seed=args.seed, folds=3)
        result = run_experiment(
            run_id=f"surface-{algorithm}-seed{args.seed}",
            objective=objective,
            objective_name=canonical_name,
            specs=specs,
            algorithm=algorithm,  # type: ignore[arg-type]
            max_trials=args.trials,
            seed=args.seed,
            checkpoint_path=checkpoint,
        )
        results.append(result)
        artifacts = write_report(result, run_dir)
        summaries.append(
            {
                "algorithm": algorithm,
                "best_score": result.best_score,
                "trials": len(result.trials),
                "artifacts": artifacts,
            }
        )
    index = write_benchmark_index(results, output_dir)
    print(json.dumps({"objective": "synthetic-quadratic-surface", "runs": summaries, "benchmark": index}, indent=2))
    return 0


def _command_recovery_demo(args: argparse.Namespace) -> int:
    """Create a reviewable checkpoint interruption and prove its continuation."""

    output_dir: Path = args.output_dir
    checkpoint = output_dir / "checkpoint.json"
    objective, specs, canonical_name = _objective("surface", seed=args.seed, folds=3)
    interrupted = False
    try:
        run_experiment(
            run_id=f"surface-recovery-seed{args.seed}",
            objective=objective,
            objective_name=canonical_name,
            specs=specs,
            algorithm="random",
            max_trials=args.trials,
            seed=args.seed,
            checkpoint_path=checkpoint,
            fail_after=args.fail_after,
        )
    except InjectedFailure:
        interrupted = True
    result = run_experiment(
        run_id=f"surface-recovery-seed{args.seed}",
        objective=objective,
        objective_name=canonical_name,
        specs=specs,
        algorithm="random",
        max_trials=args.trials,
        seed=args.seed,
        checkpoint_path=checkpoint,
    )
    artifacts = write_report(result, output_dir)
    print(
        json.dumps(
            {
                "interruption_recorded": interrupted,
                "resumed_status": result.status,
                "trials": len(result.trials),
                "checkpoint": str(checkpoint),
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _command_benchmark(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    results = []
    summaries = []
    for algorithm in ("grid", "random", "tpe", "genetic"):
        run_dir = output_dir / algorithm
        checkpoint = run_dir / "checkpoint.json"
        objective, specs, canonical_name = _objective(args.objective, seed=args.seed, folds=args.folds)
        result = run_experiment(
            run_id=f"{args.objective}-{algorithm}-seed{args.seed}",
            objective=objective,
            objective_name=canonical_name,
            specs=specs,
            algorithm=algorithm,  # type: ignore[arg-type]
            max_trials=args.trials,
            seed=args.seed,
            checkpoint_path=checkpoint,
        )
        results.append(result)
        artifacts = write_report(result, run_dir)
        summaries.append(
            {
                "algorithm": algorithm,
                "best_score": result.best_score,
                "trials": len(result.trials),
                "artifacts": artifacts,
            }
        )
    index = write_benchmark_index(results, output_dir)
    print(json.dumps({"objective": canonical_name, "runs": summaries, "benchmark": index}, indent=2))
    return 0


def _command_sweep(args: argparse.Namespace) -> int:
    payload = write_sweep(
        args.output_dir,
        args.seeds,
        objective_name=args.objective,
        data_seed=args.data_seed,
        trials=args.trials,
        folds=args.folds,
    )
    print(
        json.dumps(
            {
                "objective": payload["objective"],
                "seeds": payload["seeds"],
                "data_seed": payload["dataSeed"],
                "trials": payload["trials"],
                "folds": payload["folds"],
                "algorithms": payload["algorithms"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app.main import create_app

    uvicorn.run(create_app(args.runtime_dir), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _command_run(args)
    if args.command == "demo":
        return _command_demo(args)
    if args.command == "recovery-demo":
        return _command_recovery_demo(args)
    if args.command == "benchmark":
        return _command_benchmark(args)
    if args.command == "sweep":
        return _command_sweep(args)
    if args.command == "serve":
        return _command_serve(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
