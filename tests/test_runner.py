import json
from pathlib import Path

import pytest

from tunelab.models import ObjectiveResult, ParamSpec
from tunelab.objective import SurfaceObjective
from tunelab.runner import InjectedFailure, run_experiment


def _surface_run(
    tmp_path: Path, algorithm: str, *, checkpoint: Path | None = None, trials: int = 10
):
    objective = SurfaceObjective()
    return run_experiment(
        run_id=f"test-{algorithm}",
        objective=objective,
        objective_name=objective.name,
        specs=objective.specs,
        algorithm=algorithm,  # type: ignore[arg-type]
        max_trials=trials,
        seed=19,
        checkpoint_path=checkpoint,
    )


@pytest.mark.parametrize("algorithm", ["grid", "random", "tpe", "genetic"])
def test_all_algorithms_produce_unique_trials(tmp_path: Path, algorithm: str) -> None:
    result = _surface_run(tmp_path, algorithm, trials=8)
    assert len(result.trials) == 8
    signatures = [
        tuple(sorted((key, repr(value)) for key, value in trial.params.items()))
        for trial in result.trials
    ]
    assert len(signatures) == len(set(signatures))
    assert result.best_trial is not None


@pytest.mark.parametrize("algorithm", ["random", "tpe", "genetic"])
def test_resume_matches_uninterrupted_candidate_sequence(tmp_path: Path, algorithm: str) -> None:
    checkpoint = tmp_path / f"{algorithm}.json"
    objective = SurfaceObjective()
    with pytest.raises(InjectedFailure):
        run_experiment(
            run_id=f"resume-{algorithm}",
            objective=objective,
            objective_name=objective.name,
            specs=objective.specs,
            algorithm=algorithm,  # type: ignore[arg-type]
            max_trials=9,
            seed=23,
            checkpoint_path=checkpoint,
            fail_after=3,
        )
    resumed = run_experiment(
        run_id=f"resume-{algorithm}",
        objective=objective,
        objective_name=objective.name,
        specs=objective.specs,
        algorithm=algorithm,  # type: ignore[arg-type]
        max_trials=9,
        seed=23,
        checkpoint_path=checkpoint,
    )
    fresh = run_experiment(
        run_id=f"resume-{algorithm}",
        objective=objective,
        objective_name=objective.name,
        specs=objective.specs,
        algorithm=algorithm,  # type: ignore[arg-type]
        max_trials=9,
        seed=23,
    )
    assert [trial.params for trial in resumed.trials] == [trial.params for trial in fresh.trials]
    assert [trial.score for trial in resumed.trials] == [trial.score for trial in fresh.trials]


def test_objective_failure_is_recorded_and_checkpoint_is_valid(tmp_path: Path) -> None:
    specs = [ParamSpec("x", "int", low=1, high=3)]

    def objective(params: dict[str, object]) -> ObjectiveResult:
        if params["x"] == 2:
            raise RuntimeError("known bad candidate")
        return ObjectiveResult(score=float(params["x"]))

    checkpoint = tmp_path / "run.json"
    result = run_experiment(
        run_id="failure",
        objective=objective,
        objective_name="failure-objective",
        specs=specs,
        algorithm="grid",
        max_trials=3,
        checkpoint_path=checkpoint,
    )
    assert len(result.failed_trials) == 1
    assert "known bad candidate" in (result.failed_trials[0].error or "")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert len(payload["trials"]) == 3


def test_objective_contract_error_stops_before_a_bad_trial_is_checkpointed(tmp_path: Path) -> None:
    specs = [ParamSpec("x", "int", low=1, high=2)]

    def objective(_params: dict[str, object]) -> object:
        return {"score": 1.0}

    checkpoint = tmp_path / "contract.json"
    with pytest.raises(TypeError, match="Objective must return ObjectiveResult"):
        run_experiment(
            run_id="contract-error",
            objective=objective,  # type: ignore[arg-type]
            objective_name="contract-error",
            specs=specs,
            algorithm="grid",
            max_trials=2,
            checkpoint_path=checkpoint,
        )
    assert not checkpoint.exists()


def test_checkpoint_digest_rejects_changed_configuration(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    _surface_run(tmp_path, "grid", checkpoint=checkpoint, trials=2)
    objective = SurfaceObjective()
    with pytest.raises(ValueError, match="does not match"):
        run_experiment(
            run_id="test-grid",
            objective=objective,
            objective_name=objective.name,
            specs=objective.specs,
            algorithm="grid",
            max_trials=3,
            seed=19,
            checkpoint_path=checkpoint,
        )


def _checkpoint_payload(tmp_path: Path, name: str) -> tuple[Path, dict]:
    checkpoint = tmp_path / f"{name}.json"
    _surface_run(tmp_path, "grid", checkpoint=checkpoint, trials=2)
    return checkpoint, json.loads(checkpoint.read_text(encoding="utf-8"))


def test_checkpoint_rejects_invalid_trial_state_and_duration(tmp_path: Path) -> None:
    checkpoint, payload = _checkpoint_payload(tmp_path, "invalid-state")
    payload["trials"][0]["state"] = "skipped"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid trial"):
        _surface_run(tmp_path, "grid", checkpoint=checkpoint, trials=2)

    checkpoint, payload = _checkpoint_payload(tmp_path, "invalid-duration")
    payload["trials"][0]["duration_seconds"] = float("nan")
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid trial"):
        _surface_run(tmp_path, "grid", checkpoint=checkpoint, trials=2)


def test_checkpoint_rejects_noncanonical_trial_numbers_and_params(tmp_path: Path) -> None:
    checkpoint, payload = _checkpoint_payload(tmp_path, "noncanonical-numbers")
    payload["trials"][1]["number"] = 0
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="numbers must be contiguous"):
        _surface_run(tmp_path, "grid", checkpoint=checkpoint, trials=2)

    checkpoint, payload = _checkpoint_payload(tmp_path, "noncanonical-params")
    first_param = next(iter(payload["trials"][0]["params"]))
    payload["trials"][0]["params"][first_param] = 999_999
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the requested search space"):
        _surface_run(tmp_path, "grid", checkpoint=checkpoint, trials=2)


def test_checkpoint_rejects_duplicate_parameter_configurations(tmp_path: Path) -> None:
    checkpoint, payload = _checkpoint_payload(tmp_path, "duplicate-params")
    payload["trials"][1]["params"] = payload["trials"][0]["params"]
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate parameter configurations"):
        _surface_run(tmp_path, "grid", checkpoint=checkpoint, trials=2)


def test_checkpoint_rejects_a_changed_objective_configuration(tmp_path) -> None:
    """A fold count change must invalidate the checkpoint, not blend scores.

    The objective name is a constant, so before objectives carried a
    fingerprint a 3-fold checkpoint resumed happily under 9-fold settings and
    the run compared incomparable scores against each other.
    """
    from tunelab.objective import IrisObjective

    checkpoint = tmp_path / "checkpoint.json"
    three = IrisObjective(seed=17, folds=3)
    run_experiment(
        run_id="folds",
        objective=three,
        objective_name=three.name,
        specs=list(three.specs),
        algorithm="random",
        max_trials=2,
        seed=17,
        checkpoint_path=checkpoint,
    )
    assert checkpoint.exists()

    nine = IrisObjective(seed=17, folds=9)
    assert nine.fingerprint != three.fingerprint
    with pytest.raises(ValueError, match="does not match"):
        run_experiment(
            run_id="folds",
            objective=nine,
            objective_name=nine.name,
            specs=list(nine.specs),
            algorithm="random",
            max_trials=2,
            seed=17,
            checkpoint_path=checkpoint,
        )


def test_objective_without_a_fingerprint_still_works() -> None:
    """The fingerprint is optional: a plain callable objective is still valid."""

    def objective(params):
        return ObjectiveResult(score=float(params["x"]), metrics={})

    result = run_experiment(
        run_id="plain",
        objective=objective,
        objective_name="plain-callable",
        specs=[ParamSpec("x", "float", low=0.0, high=1.0, step=0.5)],
        algorithm="grid",
        max_trials=3,
        seed=5,
    )
    assert result.best_score == 1.0


def test_continuous_random_checkpoint_resumes_and_rejects_out_of_bounds_values(tmp_path: Path) -> None:
    specs = [ParamSpec("rate", "float", low=0.01, high=0.99)]

    def objective(params: dict[str, object]) -> ObjectiveResult:
        return ObjectiveResult(score=float(params["rate"]))

    checkpoint = tmp_path / "continuous.json"
    with pytest.raises(InjectedFailure):
        run_experiment(
            run_id="continuous",
            objective=objective,
            objective_name="continuous",
            specs=specs,
            algorithm="random",
            max_trials=6,
            seed=17,
            checkpoint_path=checkpoint,
            fail_after=3,
        )
    resumed = run_experiment(
        run_id="continuous",
        objective=objective,
        objective_name="continuous",
        specs=specs,
        algorithm="random",
        max_trials=6,
        seed=17,
        checkpoint_path=checkpoint,
    )
    assert len(resumed.trials) == 6
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["trials"][0]["params"]["rate"] = 1.5
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        run_experiment(
            run_id="continuous",
            objective=objective,
            objective_name="continuous",
            specs=specs,
            algorithm="random",
            max_trials=6,
            seed=17,
            checkpoint_path=checkpoint,
        )


def test_grid_requires_explicit_float_step_and_rejects_large_enumerations() -> None:
    def objective(params: dict[str, object]) -> ObjectiveResult:
        return ObjectiveResult(score=float(params["rate"]))

    with pytest.raises(ValueError, match="explicit step"):
        run_experiment(
            run_id="continuous-grid",
            objective=objective,
            objective_name="continuous-grid",
            specs=[ParamSpec("rate", "float", low=0.0, high=1.0)],
            algorithm="grid",
            max_trials=2,
        )
    with pytest.raises(ValueError, match="more than 10,000"):
        run_experiment(
            run_id="big-grid",
            objective=objective,
            objective_name="big-grid",
            specs=[ParamSpec("rate", "int", low=0, high=10_001)],
            algorithm="grid",
            max_trials=1,
        )


def test_timing_is_observed_separately_and_pause_resumes_at_trial_boundary(tmp_path: Path) -> None:
    specs = [ParamSpec("x", "int", low=1, high=6)]

    def objective(params: dict[str, object]) -> ObjectiveResult:
        return ObjectiveResult(score=float(params["x"]))

    checkpoint = tmp_path / "paused.json"
    paused = run_experiment(
        run_id="pause",
        objective=objective,
        objective_name="pause",
        specs=specs,
        algorithm="grid",
        max_trials=6,
        checkpoint_path=checkpoint,
        control=lambda trial, _trials: "pause" if trial.number == 2 else "continue",
    )
    assert paused.status == "paused"
    assert len(paused.trials) == 3
    assert all(trial.duration_seconds is not None and trial.duration_seconds >= 0 for trial in paused.trials)
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_payload["schema_version"] == 2
    assert all(row["duration_seconds"] is None for row in checkpoint_payload["trials"])
    observations = json.loads((tmp_path / "observations.json").read_text(encoding="utf-8"))
    assert len(observations["trials"]) == 3
    resumed = run_experiment(
        run_id="pause",
        objective=objective,
        objective_name="pause",
        specs=specs,
        algorithm="grid",
        max_trials=6,
        checkpoint_path=checkpoint,
    )
    assert resumed.status == "completed"
    assert [trial.params for trial in resumed.trials] == [{"x": value} for value in range(1, 7)]
