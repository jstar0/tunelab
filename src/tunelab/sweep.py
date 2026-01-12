"""Multi-seed stability artifacts for TuneLab's public-data benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from math import sqrt
from pathlib import Path
from statistics import fmean
from typing import Any

from .objective import IrisObjective, WineObjective
from .runner import run_experiment

ALGORITHM_ORDER = ("grid", "random", "tpe", "genetic")
DEFAULT_DATA_SEED = 2026


def parse_seed_list(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of unique integer seeds."""

    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("seeds must be a comma-separated list of integers")
    try:
        seeds = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError("seeds must be a comma-separated list of integers") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def _validate_inputs(seeds: tuple[int, ...], data_seed: int, trials: int, folds: int) -> None:
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if isinstance(data_seed, bool) or not isinstance(data_seed, int):
        raise ValueError("data seed must be an integer")
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if folds < 2:
        raise ValueError("folds must be at least 2")


def _objective(name: str, *, seed: int, folds: int) -> IrisObjective | WineObjective:
    if name == "iris":
        return IrisObjective(seed=seed, folds=folds)
    if name == "wine":
        return WineObjective(seed=seed, folds=folds)
    raise ValueError("objective must be 'iris' or 'wine'")


def seed_sweep_payload(
    seeds: Iterable[int], *, objective_name: str = "wine", data_seed: int = DEFAULT_DATA_SEED,
    trials: int = 8, folds: int = 3
) -> dict[str, Any]:
    """Compare all search strategies on the same public-data contract.

    Runtime duration is intentionally excluded from this payload. Scores and
    candidate choices are the reproducibility contract; wall-clock timings can
    be inspected in individual run reports without making the sweep hash
    unstable across machines.
    """

    seed_tuple = tuple(seeds)
    _validate_inputs(seed_tuple, data_seed, trials, folds)
    runs: list[dict[str, Any]] = []
    for seed in seed_tuple:
        for algorithm in ALGORITHM_ORDER:
            objective = _objective(objective_name, seed=data_seed, folds=folds)
            result = run_experiment(
                run_id=f"{objective_name}-{algorithm}-search{seed}-data{data_seed}-sweep",
                objective=objective,
                objective_name=objective.name,
                specs=list(objective.specs),
                algorithm=algorithm,  # type: ignore[arg-type]
                max_trials=trials,
                seed=seed,
            )
            runs.append(
                {
                    "seed": seed,
                    "algorithm": algorithm,
                    "bestScore": None if result.best_score is None else round(float(result.best_score), 8),
                    "completedTrials": len(result.completed_trials),
                    "failedTrials": len(result.failed_trials),
                    "totalTrials": len(result.trials),
                    "bestParams": result.best_trial.params if result.best_trial else {},
                }
            )

    aggregate: dict[str, dict[str, float]] = {}
    for algorithm in ALGORITHM_ORDER:
        scores = [
            float(run["bestScore"])
            for run in runs
            if run["algorithm"] == algorithm and run["bestScore"] is not None
        ]
        if not scores:
            aggregate[algorithm] = {
                "meanBestScore": 0.0,
                "populationStdBestScore": 0.0,
                "minBestScore": 0.0,
                "maxBestScore": 0.0,
                "successRate": 0.0,
            }
            continue
        mean = fmean(scores)
        aggregate[algorithm] = {
            "meanBestScore": round(mean, 8),
            "populationStdBestScore": round(sqrt(fmean((score - mean) ** 2 for score in scores)), 8),
            "minBestScore": round(min(scores), 8),
            "maxBestScore": round(max(scores), 8),
            "successRate": round(len(scores) / len(seed_tuple), 8),
        }

    return {
        "sweepVersion": 1,
        "objective": _objective(objective_name, seed=seed_tuple[0], folds=folds).name,
        "seeds": list(seed_tuple),
        "dataSeed": data_seed,
        "trials": trials,
        "folds": folds,
        "algorithms": list(ALGORITHM_ORDER),
        "runs": runs,
        "aggregate": aggregate,
        "boundary": {
            "publicData": _objective(objective_name, seed=data_seed, folds=folds).public_data,
            "fixedDataSeed": True,
            "sameBudget": True,
            "timingsExcluded": True,
            "claim": "stability descriptor over one public dataset, not a universal ranking",
        },
    }


def write_sweep(
    output_dir: str | Path, seeds: Iterable[int], *, objective_name: str = "wine",
    data_seed: int = DEFAULT_DATA_SEED, trials: int = 8, folds: int = 3
) -> dict[str, Any]:
    """Write deterministic JSON/CSV/Markdown artifacts and a SHA-256 manifest."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = seed_sweep_payload(
        seeds, objective_name=objective_name, data_seed=data_seed, trials=trials, folds=folds
    )
    write_text(directory / "sweep.json", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_csv(directory / "sweep.csv", payload["runs"])
    write_text(directory / "report.md", markdown_report(payload))
    manifest = {
        "schemaVersion": 1,
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()),
            }
            for path in sorted(directory.glob("*"))
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    write_text(directory / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return payload


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    if not runs:
        raise ValueError("runs must not be empty")
    fields = ("seed", "algorithm", "bestScore", "completedTrials", "failedTrials", "totalTrials", "bestParams")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    **run,
                    "bestParams": json.dumps(run["bestParams"], ensure_ascii=False, sort_keys=True),
                }
            )


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# TuneLab {payload['boundary']['publicData']} seed sweep",
        "",
        "The same public-data objective, search space, trial budget, and CV folds are evaluated across explicit seeds.",
        "",
        f"- seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"- fixed data/model seed: `{payload['dataSeed']}`",
        f"- trials per algorithm: `{payload['trials']}`",
        f"- cross-validation folds: `{payload['folds']}`",
        "",
        "| seed | algorithm | best score | completed | failed |",
        "|---:|---|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        score = "—" if run["bestScore"] is None else f"{run['bestScore']:.8f}"
        lines.append(
            f"| {run['seed']} | {run['algorithm']} | {score} | {run['completedTrials']} | {run['failedTrials']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| algorithm | mean best score | population σ | min | max | success rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for algorithm in ALGORITHM_ORDER:
        summary = payload["aggregate"][algorithm]
        lines.append(
            f"| {algorithm} | {summary['meanBestScore']:.8f} | {summary['populationStdBestScore']:.8f} | "
            f"{summary['minBestScore']:.8f} | {summary['maxBestScore']:.8f} | {summary['successRate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Scores are a stability descriptor for one public-data contract, not a universal algorithm ranking. "
            "Timing is intentionally omitted from hash-bound artifacts because it is machine-dependent.",
            "",
        ]
    )
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
