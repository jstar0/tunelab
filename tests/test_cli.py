import json
from pathlib import Path

from tunelab.cli import main


def test_demo_cli_writes_four_strategy_index(tmp_path: Path, capsys) -> None:
    assert main(["demo", "--trials", "4", "--output-dir", str(tmp_path / "demo")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["runs"]) == 4
    assert (tmp_path / "demo" / "benchmark.html").exists()


def test_demo_artifacts_are_byte_identical_in_distinct_output_directories(tmp_path: Path, capsys) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert main(["demo", "--trials", "4", "--output-dir", str(first)]) == 0
    capsys.readouterr()
    assert main(["demo", "--trials", "4", "--output-dir", str(second)]) == 0

    def tree(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.name not in {"checkpoint.json", "observations.json"}
        }

    assert tree(first) == tree(second)


def test_sweep_cli_writes_hash_bound_artifacts(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "sweep"
    assert main(["sweep", "--seeds", "17,29", "--trials", "2", "--output-dir", str(output_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seeds"] == [17, 29]
    assert payload["algorithms"] == ["grid", "random", "tpe", "genetic"]
    assert all((output_dir / name).exists() for name in ("sweep.json", "sweep.csv", "report.md", "manifest.json"))


def test_recovery_demo_leaves_a_resumed_run_and_observations(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "recovery"
    assert main(["recovery-demo", "--trials", "6", "--fail-after", "2", "--output-dir", str(output_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["interruption_recorded"] is True
    assert payload["resumed_status"] == "completed"
    assert payload["trials"] == 6
    assert (output_dir / "checkpoint.json").exists()
    assert (output_dir / "observations.json").exists()
