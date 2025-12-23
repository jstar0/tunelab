from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Any, Literal

from .models import Algorithm, ObjectiveResult, ParamSpec, RunResult, Trial
from .space import (
    crossover,
    grid_cardinality,
    iter_grid_candidates,
    mutate,
    params_signature,
    sample_params,
    sample_tpe_like,
)


class InjectedFailure(RuntimeError):
    """Raised by the demo to prove that a checkpoint can resume safely."""


Objective = Callable[[dict[str, Any]], ObjectiveResult]
RunControl = Callable[[Trial, tuple[Trial, ...]], Literal["continue", "pause", "cancel"] | None]


def _canonical_config(
    *,
    run_id: str,
    objective: str,
    algorithm: Algorithm,
    specs: list[ParamSpec],
    max_trials: int,
    seed: int,
    objective_fingerprint: str = "",
) -> str:
    payload = {
        "run_id": run_id,
        "objective": objective,
        "objective_fingerprint": objective_fingerprint,
        "algorithm": algorithm,
        "specs": [spec.as_dict() for spec in specs],
        "max_trials": max_trials,
        "seed": seed,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _config_digest(**kwargs: Any) -> str:
    return hashlib.sha256(_canonical_config(**kwargs).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _observations_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name("observations.json")


def _load_observations(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("trials", [])
        if payload.get("schema_version") != 1 or not isinstance(rows, list):
            return {}
        return {
            int(row["number"]): float(row["duration_seconds"])
            for row in rows
            if isinstance(row, dict)
            and type(row.get("number")) is int
            and isinstance(row.get("duration_seconds"), (int, float))
            and not isinstance(row.get("duration_seconds"), bool)
            and float(row["duration_seconds"]) >= 0
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _load_checkpoint(path: Path, digest: str, specs: list[ParamSpec]) -> list[Trial]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint {path}: {exc}") from exc
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2} or payload.get("config_digest") != digest:
        raise ValueError("Checkpoint does not match the requested experiment configuration")
    raw_trials = payload.get("trials")
    if not isinstance(raw_trials, list):
        raise ValueError("Checkpoint trials must be a list")
    try:
        trials = [Trial.from_dict(item) for item in raw_trials]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint contains an invalid trial: {exc}") from exc
    if [trial.number for trial in trials] != list(range(len(trials))):
        raise ValueError("Checkpoint trial numbers must be contiguous from zero")
    expected = {spec.name: spec for spec in specs}
    if any(set(trial.params) != set(expected) for trial in trials):
        raise ValueError("Checkpoint trial parameters do not match the requested search space")
    if any(not all(expected[key].validate_value(value) for key, value in trial.params.items()) for trial in trials):
        raise ValueError("Checkpoint contains parameters outside the requested search space")
    if len({params_signature(trial.params) for trial in trials}) != len(trials):
        raise ValueError("Checkpoint contains duplicate parameter configurations")
    observations = _load_observations(_observations_path(path))
    return [
        replace(
            trial,
            duration_seconds=observations.get(
                trial.number,
                None if schema_version == 1 and trial.duration_seconds == 0 else trial.duration_seconds,
            ),
        )
        for trial in trials
    ]


def _write_checkpoint(
    path: Path,
    *,
    digest: str,
    run_id: str,
    objective: str,
    algorithm: Algorithm,
    seed: int,
    max_trials: int,
    specs: list[ParamSpec],
    trials: Iterable[Trial],
    status: str = "running",
) -> None:
    canonical_trials = [replace(trial, duration_seconds=None).as_dict() for trial in trials]
    _atomic_write_json(
        path,
        {
            "schema_version": 2,
            "config_digest": digest,
            "run_id": run_id,
            "objective": objective,
            "algorithm": algorithm,
            "seed": seed,
            "max_trials": max_trials,
            "search_space": [spec.as_dict() for spec in specs],
            "status": status,
            "trials": canonical_trials,
        },
    )


def _write_observation(path: Path, trial: Trial, started_at: float, finished_at: float) -> None:
    """Persist machine-local timing separately from canonical experiment facts."""

    existing: dict[int, dict[str, Any]] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("trials", []):
                if isinstance(row, dict) and type(row.get("number")) is int:
                    existing[row["number"]] = row
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    existing[trial.number] = {
        "number": trial.number,
        "duration_seconds": None if trial.duration_seconds is None else round(trial.duration_seconds, 6),
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
    }
    _atomic_write_json(path, {"schema_version": 1, "trials": [existing[key] for key in sorted(existing)]})


def _next_grid_candidate(
    candidates: Iterable[dict[str, Any]], seen: set[tuple[tuple[str, str], ...]]
) -> dict[str, Any] | None:
    for candidate in candidates:
        if params_signature(candidate) not in seen:
            return candidate
    return None


def _next_genetic_candidate(
    specs: list[ParamSpec], trials: list[Trial], rng: Random
) -> dict[str, Any]:
    completed = [
        trial for trial in trials if trial.state == "completed" and trial.score is not None
    ]
    if len(completed) < 4:
        return sample_params(specs, rng)
    parent_count = max(2, len(completed) // 3)
    parents = sorted(completed, key=lambda trial: float(trial.score), reverse=True)[:parent_count]
    first = rng.choice(parents).params
    second = rng.choice(parents).params
    return mutate(crossover(first, second, specs, rng), specs, rng)


def _propose(
    algorithm: Algorithm,
    specs: list[ParamSpec],
    trials: list[Trial],
    rng: Random,
    grid: Iterable[dict[str, Any]] | None,
    seen: set[tuple[tuple[str, str], ...]],
) -> dict[str, Any] | None:
    if algorithm == "grid":
        if grid is None:
            raise RuntimeError("Grid candidates were not initialized")
        return _next_grid_candidate(grid, seen)
    history = [
        (trial.params, float(trial.score))
        for trial in trials
        if trial.state == "completed" and trial.score is not None
    ]
    for _ in range(128):
        if algorithm == "random":
            candidate = sample_params(specs, rng)
        elif algorithm == "tpe":
            candidate = sample_tpe_like(specs, history, rng)
        else:
            candidate = _next_genetic_candidate(specs, trials, rng)
        if params_signature(candidate) not in seen:
            return candidate
    # A finite space can be exhausted before max_trials.  Returning None is a
    # normal completion signal, not an error.
    return None


def run_experiment(
    *,
    run_id: str,
    objective: Objective,
    objective_name: str,
    specs: list[ParamSpec],
    algorithm: Algorithm,
    max_trials: int,
    seed: int = 42,
    checkpoint_path: Path | None = None,
    control: RunControl | None = None,
    fail_after: int | None = None,
) -> RunResult:
    """Run a deterministic search with atomic checkpoint/resume semantics."""

    if not specs:
        raise ValueError("At least one search-space parameter is required")
    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")
    if algorithm not in {"grid", "random", "tpe", "genetic"}:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if fail_after is not None and fail_after < 1:
        raise ValueError("fail_after must be at least 1")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Search-space parameter names must be unique")
    if algorithm == "grid" and grid_cardinality(specs) > 10_000:
        raise ValueError("Grid has more than 10,000 combinations; choose random, TPE-like, or genetic search")

    # The digest must cover the objective's *internal* configuration, not just
    # its name. Without this, a checkpoint taken with one cross-validation fold
    # count resumes happily under another and the run mixes incomparable scores.
    objective_fingerprint = str(getattr(objective, "fingerprint", objective_name))
    digest = _config_digest(
        run_id=run_id,
        objective=objective_name,
        objective_fingerprint=objective_fingerprint,
        algorithm=algorithm,
        specs=specs,
        max_trials=max_trials,
        seed=seed,
    )
    trials = (
        _load_checkpoint(checkpoint_path, digest, specs)
        if checkpoint_path and checkpoint_path.exists()
        else []
    )
    if len(trials) > max_trials:
        raise ValueError("Checkpoint contains more trials than max_trials")
    seen = {params_signature(trial.params) for trial in trials}
    grid = iter_grid_candidates(specs) if algorithm == "grid" else None
    new_trials = 0
    while len(trials) < max_trials:
        number = len(trials)
        # Derive an independent stream per trial. This makes a resumed run
        # propose exactly the same candidates as a fresh run, even when an
        # adaptive strategy takes a variable number of random draws.
        proposal_rng = Random(seed + number * 1_000_003)
        params = _propose(algorithm, specs, trials, proposal_rng, grid, seen)
        if params is None:
            break
        started_at = time.time()
        started_clock = time.perf_counter()
        try:
            outcome = objective(params)
        except Exception as exc:  # record an objective failure and keep the run inspectable
            trial = Trial(
                number=number,
                params=dict(params),
                score=None,
                metrics={},
                state="failed",
                duration_seconds=time.perf_counter() - started_clock,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            # A bad adapter contract is a caller error, not an objective result.
            # Let it stop the run before an invalid row can be checkpointed.
            if not isinstance(outcome, ObjectiveResult):
                raise TypeError("Objective must return ObjectiveResult")
            trial = Trial(
                number=number,
                params=dict(params),
                score=float(outcome.score),
                metrics={str(k): float(v) for k, v in outcome.metrics.items()},
                state="completed",
                duration_seconds=time.perf_counter() - started_clock,
            )
        finished_at = time.time()
        trials.append(trial)
        seen.add(params_signature(params))
        new_trials += 1
        if checkpoint_path:
            _write_observation(_observations_path(checkpoint_path), trial, started_at, finished_at)
            _write_checkpoint(
                checkpoint_path,
                digest=digest,
                run_id=run_id,
                objective=objective_name,
                algorithm=algorithm,
                seed=seed,
                max_trials=max_trials,
                specs=specs,
                trials=trials,
            )
        if control is not None:
            action = control(trial, tuple(trials)) or "continue"
            if action not in {"continue", "pause", "cancel"}:
                raise ValueError("Run control must return continue, pause, cancel, or None")
            if action != "continue":
                status = "paused" if action == "pause" else "cancelled"
                if checkpoint_path:
                    _write_checkpoint(
                        checkpoint_path,
                        digest=digest,
                        run_id=run_id,
                        objective=objective_name,
                        algorithm=algorithm,
                        seed=seed,
                        max_trials=max_trials,
                        specs=specs,
                        trials=trials,
                        status=status,
                    )
                return RunResult(
                    run_id=run_id,
                    algorithm=algorithm,
                    objective=objective_name,
                    seed=seed,
                    max_trials=max_trials,
                    trials=tuple(trials),
                    duration_seconds=sum(item.duration_seconds or 0.0 for item in trials),
                    status=status,
                )
        if fail_after is not None and new_trials >= fail_after:
            raise InjectedFailure(f"Injected failure after {new_trials} new trial(s)")

    completed = any(trial.state == "completed" for trial in trials)
    return RunResult(
        run_id=run_id,
        algorithm=algorithm,
        objective=objective_name,
        seed=seed,
        max_trials=max_trials,
        trials=tuple(trials),
        duration_seconds=sum(trial.duration_seconds or 0.0 for trial in trials) if trials else None,
        status="completed" if completed else "failed",
    )


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Read a small checkpoint summary without loading an objective."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "run_id": payload.get("run_id"),
        "algorithm": payload.get("algorithm"),
        "objective": payload.get("objective"),
        "completed_trials": len(payload.get("trials", [])),
        "config_digest": payload.get("config_digest"),
        "updated_at": payload.get("updated_at"),
    }
