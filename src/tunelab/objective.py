from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .models import ObjectiveResult, ParamSpec


class SurfaceObjective:
    """A quick deterministic objective used for tests and algorithm demos."""

    name = "synthetic-quadratic-surface"
    # Closed-form and parameter-free, so the name fully identifies it.
    fingerprint = "synthetic-quadratic-surface"

    def __init__(self, *, fail_on_activation: str | None = None) -> None:
        # This is deliberately opt-in and only exposed as a named tutorial.
        # It gives the workbench a reproducible failed-trial path without
        # pretending that the synthetic surface is a benchmark.
        self.fail_on_activation = fail_on_activation
        self.specs = [
            # The grid is intentionally coarse enough to be inspectable in a
            # short demo while still containing the surface optimum.
            ParamSpec("learning_rate", "float", low=0.01, high=0.49, step=0.08),
            ParamSpec("depth", "int", low=1, high=11, step=2),
            ParamSpec("activation", "categorical", choices=("relu", "tanh", "gelu")),
        ]

    def __call__(self, params: dict[str, Any]) -> ObjectiveResult:
        rate = float(params["learning_rate"])
        depth = int(params["depth"])
        if params["activation"] == self.fail_on_activation:
            raise RuntimeError(
                f"Tutorial failure: activation={self.fail_on_activation!r} is intentionally unavailable"
            )
        activation_bonus = {"relu": 0.0, "tanh": 0.018, "gelu": 0.012}[str(params["activation"])]
        score = 1.0 - ((rate - 0.17) / 0.16) ** 2 - ((depth - 7) / 7) ** 2 * 0.18 + activation_bonus
        score = max(0.0, min(1.0, score))
        return ObjectiveResult(score=score, metrics={"surface_score": score})


class IrisObjective:
    """Cross-validated RandomForest objective over sklearn's public Iris data."""

    name = "iris-random-forest"
    public_data = "sklearn-iris"

    def __init__(self, *, seed: int = 42, folds: int = 3) -> None:
        self.seed = int(seed)
        self.folds = int(folds)
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        # Everything that changes a reported score belongs here, so a resumed
        # checkpoint cannot silently mix 3-fold and 9-fold results.
        self.fingerprint = f"{self.name}:seed={self.seed}:folds={self.folds}"
        dataset = load_iris()
        self.features = dataset.data
        self.target = dataset.target
        self.specs = [
            ParamSpec("n_estimators", "int", low=20, high=100, step=20),
            ParamSpec("max_depth", "int", low=2, high=12, step=2),
            ParamSpec("min_samples_split", "int", low=2, high=10, step=2),
            ParamSpec("max_features", "categorical", choices=("sqrt", "log2", 1.0)),
        ]

    def __call__(self, params: dict[str, Any]) -> ObjectiveResult:
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            max_features=params["max_features"],
            random_state=self.seed,
            n_jobs=1,
        )
        splitter = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        scores = cross_val_score(
            model, self.features, self.target, cv=splitter, scoring="accuracy", n_jobs=1
        )
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        return ObjectiveResult(score=mean, metrics={"cv_mean": mean, "cv_std": std})


def make_iris_objective(*, seed: int = 42, folds: int = 3) -> tuple[IrisObjective, list[ParamSpec]]:
    objective = IrisObjective(seed=seed, folds=folds)
    return objective, list(objective.specs)


class WineObjective:
    """Cross-validated RandomForest objective over sklearn's public Wine data.

    This uses the same visible search space as Iris so comparison behavior stays
    inspectable, but the larger feature set makes the public benchmark less
    score-saturated under a small evaluation budget.
    """

    name = "wine-random-forest"
    public_data = "sklearn-wine"

    def __init__(self, *, seed: int = 42, folds: int = 3) -> None:
        self.seed = int(seed)
        self.folds = int(folds)
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        self.fingerprint = f"{self.name}:seed={self.seed}:folds={self.folds}"
        dataset = load_wine()
        self.features = dataset.data
        self.target = dataset.target
        self.specs = [
            ParamSpec("n_estimators", "int", low=20, high=100, step=20),
            ParamSpec("max_depth", "int", low=2, high=12, step=2),
            ParamSpec("min_samples_split", "int", low=2, high=10, step=2),
            ParamSpec("max_features", "categorical", choices=("sqrt", "log2", 1.0)),
        ]

    def __call__(self, params: dict[str, Any]) -> ObjectiveResult:
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            max_features=params["max_features"],
            random_state=self.seed,
            n_jobs=1,
        )
        splitter = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        scores = cross_val_score(
            model, self.features, self.target, cv=splitter, scoring="accuracy", n_jobs=1
        )
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        return ObjectiveResult(score=mean, metrics={"cv_mean": mean, "cv_std": std})


def make_wine_objective(*, seed: int = 42, folds: int = 3) -> tuple[WineObjective, list[ParamSpec]]:
    objective = WineObjective(seed=seed, folds=folds)
    return objective, list(objective.specs)


Objective = Callable[[dict[str, Any]], ObjectiveResult]
