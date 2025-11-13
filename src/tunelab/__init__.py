"""TuneLab: reproducible and resumable hyperparameter experiments."""

from .models import Algorithm, ObjectiveResult, ParamKind, ParamSpec, RunResult, Trial
from .objective import IrisObjective, SurfaceObjective, WineObjective, make_iris_objective, make_wine_objective
from .runner import InjectedFailure, run_experiment
from .sweep import parse_seed_list, seed_sweep_payload, write_sweep

__version__ = "0.1.0"

__all__ = [
    "Algorithm",
    "IrisObjective",
    "InjectedFailure",
    "ObjectiveResult",
    "ParamKind",
    "ParamSpec",
    "RunResult",
    "SurfaceObjective",
    "WineObjective",
    "Trial",
    "__version__",
    "make_iris_objective",
    "make_wine_objective",
    "parse_seed_list",
    "run_experiment",
    "seed_sweep_payload",
    "write_sweep",
]
