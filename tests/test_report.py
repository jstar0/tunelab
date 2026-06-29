from pathlib import Path

from tunelab.objective import SurfaceObjective
from tunelab.report import write_report
from tunelab.runner import run_experiment


def test_report_contains_machine_and_human_artifacts(tmp_path: Path) -> None:
    objective = SurfaceObjective()
    result = run_experiment(
        run_id="report",
        objective=objective,
        objective_name=objective.name,
        specs=objective.specs,
        algorithm="random",
        max_trials=4,
        seed=2,
    )
    artifacts = write_report(result, tmp_path / "report")
    assert all(Path(path).exists() for path in artifacts.values())
    document = (tmp_path / "report" / "report.html").read_text(encoding="utf-8")
    assert "TuneLab experiment report" in document
    assert "Best parameters" in document
    assert "<svg" in document
