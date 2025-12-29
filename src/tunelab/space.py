from __future__ import annotations

import itertools
import math
from random import Random
from typing import Any

from .models import ParamSpec


def params_signature(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return a stable, hashable identity for a parameter assignment."""

    return tuple(sorted((str(key), repr(value)) for key, value in params.items()))


def grid_cardinality(specs: list[ParamSpec]) -> int:
    """Count a discrete grid without allocating every configuration."""

    total = 1
    for spec in specs:
        total *= len(spec.grid_values())
    return total


def iter_grid_candidates(specs: list[ParamSpec]):
    """Yield deterministic grid points lazily for grid search only."""

    values = [spec.grid_values() for spec in specs]
    yield from (
        {spec.name: value for spec, value in zip(specs, combination, strict=True)}
        for combination in itertools.product(*values)
    )


def grid_candidates(specs: list[ParamSpec]) -> list[dict[str, Any]]:
    """Materialize a grid for callers that explicitly need every candidate."""

    return list(iter_grid_candidates(specs))


def sample_value(spec: ParamSpec, rng: Random) -> Any:
    if spec.choices:
        return rng.choice(list(spec.choices))
    assert spec.low is not None and spec.high is not None
    if spec.kind == "int":
        return int(rng.choice(spec.grid_values()))
    if spec.log_scale:
        value = math.exp(rng.uniform(math.log(float(spec.low)), math.log(float(spec.high))))
    else:
        value = rng.uniform(float(spec.low), float(spec.high))
    if spec.step is not None:
        step = float(spec.step)
        value = float(spec.low) + round((value - float(spec.low)) / step) * step
        value = min(float(spec.high), max(float(spec.low), value))
    return round(float(value), 10)


def sample_params(specs: list[ParamSpec], rng: Random) -> dict[str, Any]:
    return {spec.name: sample_value(spec, rng) for spec in specs}


def _sample_near(spec: ParamSpec, anchor: Any, rng: Random) -> Any:
    if spec.choices:
        choices = list(spec.choices)
        if anchor in choices and rng.random() < 0.75:
            return anchor
        return rng.choice(choices)
    assert spec.low is not None and spec.high is not None
    span = float(spec.high) - float(spec.low)
    if spec.log_scale:
        low = math.log(float(spec.low))
        high = math.log(float(spec.high))
        value = math.exp(
            rng.gauss(math.log(max(float(spec.low), float(anchor))), (high - low) * 0.18)
        )
    else:
        value = rng.gauss(float(anchor), span * 0.18)
    value = min(float(spec.high), max(float(spec.low), value))
    if spec.kind == "int":
        step = int(spec.step or 1)
        return int(round((value - int(spec.low)) / step) * step + int(spec.low))
    if spec.step is not None:
        step = float(spec.step)
        value = float(spec.low) + round((value - float(spec.low)) / step) * step
    return round(min(float(spec.high), max(float(spec.low), value)), 10)


def sample_tpe_like(
    specs: list[ParamSpec], history: list[tuple[dict[str, Any], float]], rng: Random
) -> dict[str, Any]:
    """A dependency-free density-guided sampler inspired by TPE.

    It deliberately exposes a small, deterministic policy rather than claiming
    to be a drop-in replacement for a full Bayesian optimizer: random warm-up,
    a good-set split, and local proposals around the good set.
    """

    if len(history) < 4:
        return sample_params(specs, rng)
    ordered = sorted(history, key=lambda item: item[1], reverse=True)
    good = ordered[: max(2, math.ceil(len(ordered) * 0.25))]
    anchor = rng.choice(good)[0]
    return {spec.name: _sample_near(spec, anchor[spec.name], rng) for spec in specs}


def crossover(
    parent_a: dict[str, Any], parent_b: dict[str, Any], specs: list[ParamSpec], rng: Random
) -> dict[str, Any]:
    return {
        spec.name: parent_a[spec.name] if rng.random() < 0.5 else parent_b[spec.name]
        for spec in specs
    }


def mutate(
    params: dict[str, Any], specs: list[ParamSpec], rng: Random, rate: float = 0.25
) -> dict[str, Any]:
    mutated = dict(params)
    for spec in specs:
        if rng.random() < rate:
            mutated[spec.name] = sample_value(spec, rng)
    return mutated
