from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

ParamKind = Literal["int", "float", "categorical", "bool"]
Algorithm = Literal["grid", "random", "tpe", "genetic"]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """A small, serializable search-space declaration."""

    name: str
    kind: ParamKind
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    choices: tuple[Any, ...] = ()
    log_scale: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Parameter name cannot be blank")
        if self.kind in {"categorical", "bool"}:
            if not self.choices:
                raise ValueError(f"{self.name} requires at least one choice")
            if self.kind == "bool" and any(not isinstance(item, bool) for item in self.choices):
                raise ValueError(f"{self.name} bool choices must contain only booleans")
            return
        if self.low is None or self.high is None or self.low >= self.high:
            raise ValueError(f"{self.name} requires ascending numeric bounds")
        if self.log_scale and (self.low <= 0 or self.high <= 0):
            raise ValueError(f"{self.name} log_scale requires positive bounds")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"{self.name} step must be positive")
        if self.kind == "int" and any(
            not isinstance(value, int) for value in (self.low, self.high)
        ):
            raise ValueError(f"{self.name} int bounds must be integers")

    def grid_values(self) -> list[Any]:
        if self.choices:
            return list(self.choices)
        assert self.low is not None and self.high is not None
        if self.kind == "int":
            step = int(self.step or 1)
            return list(range(int(self.low), int(self.high) + 1, step))
        if self.step is None:
            raise ValueError(f"{self.name} float grid search requires an explicit step")
        step = float(self.step)
        values: list[float] = []
        current = float(self.low)
        while current <= float(self.high) + 1e-12:
            values.append(round(current, 10))
            current += step
        return values

    def validate_value(self, value: Any) -> bool:
        """Return whether one persisted proposal belongs to this exact space."""

        if self.choices:
            if self.kind == "bool":
                return type(value) is bool and value in self.choices
            return value in self.choices
        if self.low is None or self.high is None:
            return False
        if self.kind == "int":
            if type(value) is not int or not int(self.low) <= value <= int(self.high):
                return False
            step = int(self.step or 1)
            return (value - int(self.low)) % step == 0
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False
        numeric = float(value)
        tolerance = 1e-9
        if numeric < float(self.low) - tolerance or numeric > float(self.high) + tolerance:
            return False
        if self.step is None:
            return True
        offset = (numeric - float(self.low)) / float(self.step)
        return abs(offset - round(offset)) <= tolerance

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.low is not None:
            payload["low"] = self.low
        if self.high is not None:
            payload["high"] = self.high
        if self.step is not None:
            payload["step"] = self.step
        if self.choices:
            payload["choices"] = list(self.choices)
        if self.log_scale:
            payload["log_scale"] = True
        return payload


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    score: float
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.score)):
            raise ValueError("Objective score must be finite")
        for key, value in self.metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Objective metric {key!r} must be finite")


@dataclass(frozen=True, slots=True)
class Trial:
    number: int
    params: dict[str, Any]
    score: float | None
    metrics: dict[str, float]
    state: Literal["completed", "failed"]
    duration_seconds: float | None
    error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.number, bool) or not isinstance(self.number, int) or self.number < 0:
            raise ValueError("Trial number must be a non-negative integer")
        if self.state not in {"completed", "failed"}:
            raise ValueError("Trial state must be completed or failed")
        if self.duration_seconds is not None and (
            not math.isfinite(float(self.duration_seconds)) or self.duration_seconds < 0
        ):
            raise ValueError("Trial duration_seconds must be finite and non-negative when recorded")
        if self.state == "completed":
            if self.score is None or not math.isfinite(float(self.score)):
                raise ValueError("Completed trial score must be finite")
            if self.error is not None:
                raise ValueError("Completed trial cannot include an error")
        else:
            if self.score is not None:
                raise ValueError("Failed trial score must be null")
            if self.metrics:
                raise ValueError("Failed trial cannot include metrics")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("Trial error must be a string or null")
        for key, value in self.metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Trial metric {key!r} must be finite")

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "params": self.params,
            "score": self.score,
            "metrics": self.metrics,
            "state": self.state,
            "duration_seconds": None if self.duration_seconds is None else round(float(self.duration_seconds), 6),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Trial:
        if not isinstance(payload, dict):
            raise ValueError("Trial payload must be an object")
        number = payload.get("number")
        params = payload.get("params")
        metrics = payload.get("metrics", {})
        state = payload.get("state")
        duration_seconds = payload.get("duration_seconds")
        score = payload.get("score")
        error = payload.get("error")
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError("Trial number must be an integer")
        if not isinstance(params, dict):
            raise ValueError("Trial params must be an object")
        if not isinstance(metrics, dict):
            raise ValueError("Trial metrics must be an object")
        if state not in {"completed", "failed"}:
            raise ValueError("Trial state must be completed or failed")
        if duration_seconds is not None and (
            isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float))
        ):
            raise ValueError("Trial duration_seconds must be numeric or null")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
            raise ValueError("Trial score must be numeric or null")
        if error is not None and not isinstance(error, str):
            raise ValueError("Trial error must be a string or null")
        return cls(
            number=number,
            params=dict(params),
            score=None if score is None else float(score),
            metrics={str(k): float(v) for k, v in metrics.items()},
            state=state,
            duration_seconds=None if duration_seconds is None else float(duration_seconds),
            error=error,
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    algorithm: Algorithm
    objective: str
    seed: int
    max_trials: int
    trials: tuple[Trial, ...]
    duration_seconds: float | None
    status: Literal["completed", "failed", "paused", "cancelled"] = "completed"

    @property
    def completed_trials(self) -> tuple[Trial, ...]:
        return tuple(trial for trial in self.trials if trial.state == "completed")

    @property
    def failed_trials(self) -> tuple[Trial, ...]:
        return tuple(trial for trial in self.trials if trial.state == "failed")

    @property
    def best_trial(self) -> Trial | None:
        candidates = [trial for trial in self.completed_trials if trial.score is not None]
        return max(candidates, key=lambda trial: (float(trial.score), -trial.number), default=None)

    @property
    def best_score(self) -> float | None:
        best = self.best_trial
        return best.score if best else None

    def as_dict(self) -> dict[str, Any]:
        best = self.best_trial
        convergence: list[dict[str, Any]] = []
        running_best: float | None = None
        for trial in self.trials:
            if trial.score is not None:
                running_best = (
                    trial.score if running_best is None else max(running_best, trial.score)
                )
            convergence.append({"trial": trial.number, "score": trial.score, "best": running_best})
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "objective": self.objective,
            "seed": self.seed,
            "max_trials": self.max_trials,
            "status": self.status,
            "total_trials": len(self.trials),
            "completed_trials": len(self.completed_trials),
            "failed_trials": len(self.failed_trials),
            "best_score": self.best_score,
            "best_params": best.params if best else {},
            "best_metrics": best.metrics if best else {},
            "mean_trial_seconds": (
                sum(trial.duration_seconds or 0.0 for trial in self.trials if trial.duration_seconds is not None)
                / sum(trial.duration_seconds is not None for trial in self.trials)
                if any(trial.duration_seconds is not None for trial in self.trials)
                else 0.0
            ),
            "successful_trial_ratio": (
                len(self.completed_trials) / len(self.trials) if self.trials else 0.0
            ),
            "convergence": convergence,
            "trials": [trial.as_dict() for trial in self.trials],
            "duration_seconds": None if self.duration_seconds is None else round(float(self.duration_seconds), 6),
        }
