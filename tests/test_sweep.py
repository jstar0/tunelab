import json
from pathlib import Path

import pytest

from tunelab.sweep import parse_seed_list, seed_sweep_payload, write_sweep


def test_parse_seed_list_preserves_order_and_rejects_duplicates() -> None:
    assert parse_seed_list("17, 29,41") == (17, 29, 41)
    with pytest.raises(ValueError, match="unique"):
        parse_seed_list("17,17")
    with pytest.raises(ValueError, match="comma-separated"):
        parse_seed_list("17,,41")


def test_seed_sweep_payload_is_deterministic_and_complete() -> None:
    first = seed_sweep_payload((17, 29), trials=2, folds=3)
    second = seed_sweep_payload((17, 29), trials=2, folds=3)
    assert first == second
    assert first["algorithms"] == ["grid", "random", "tpe", "genetic"]
    assert len(first["runs"]) == 8
    assert first["dataSeed"] == 2026
    assert first["boundary"]["fixedDataSeed"] is True
    assert set(first["aggregate"]) == set(first["algorithms"])
    assert all(run["completedTrials"] == 2 for run in first["runs"])


def test_sweep_exposes_a_less_saturated_wine_contract_and_keeps_iris_available() -> None:
    wine = seed_sweep_payload((17,), objective_name="wine", trials=1, folds=3)
    iris = seed_sweep_payload((17,), objective_name="iris", trials=1, folds=3)

    assert wine["objective"] == "wine-random-forest"
    assert wine["boundary"]["publicData"] == "sklearn-wine"
    assert iris["objective"] == "iris-random-forest"
    assert iris["boundary"]["publicData"] == "sklearn-iris"


def test_sweep_records_a_fixed_data_seed_separately_from_search_seeds() -> None:
    payload = seed_sweep_payload((17, 29), data_seed=91, trials=1, folds=3)

    assert payload["dataSeed"] == 91
    assert [run["seed"] for run in payload["runs"]] == [17, 17, 17, 17, 29, 29, 29, 29]


def test_seed_sweep_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="trials"):
        seed_sweep_payload((17,), trials=0)
    with pytest.raises(ValueError, match="folds"):
        seed_sweep_payload((17,), folds=1)
    with pytest.raises(ValueError, match="at least one"):
        seed_sweep_payload(())


def test_write_sweep_artifacts_are_byte_identical(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_sweep(first_dir, (17, 29), trials=2, folds=3)
    write_sweep(second_dir, (17, 29), trials=2, folds=3)
    names = {"sweep.json", "sweep.csv", "report.md", "manifest.json"}
    assert {path.name for path in first_dir.iterdir()} == names
    for name in names:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    payload = json.loads((first_dir / "sweep.json").read_text(encoding="utf-8"))
    assert payload["boundary"]["timingsExcluded"] is True
