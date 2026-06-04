from __future__ import annotations

import time

from fastapi.testclient import TestClient

from tunelab.app.main import create_app
from tunelab.models import ObjectiveResult
from tunelab.objective import SurfaceObjective


def _terminal(client: TestClient, job_id: str) -> dict:
    for _ in range(150):
        payload = client.get(f"/api/v1/jobs/{job_id}").json()
        if payload["status"] in {"succeeded", "failed", "cancelled", "paused"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("TuneLab job did not reach a terminal state")


def test_local_api_runs_a_real_experiment_and_serves_registered_result(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "runtime")) as client:
        objectives = client.get("/api/v1/objectives")
        assert objectives.status_code == 200
        assert {row["id"] for row in objectives.json()["objectives"]} == {"surface", "iris", "wine"}

        created = client.post(
            "/api/v1/experiment-sets",
            json={
                "name": "surface smoke",
                "objective": "surface",
                "algorithms": ["random"],
                "budget": 3,
                "search_seed": 17,
                "objective_seed": 17,
                "folds": 3,
            },
        )
        assert created.status_code == 202
        job_id = created.json()["jobs"][0]["job_id"]
        terminal = _terminal(client, job_id)
        assert terminal["status"] == "succeeded"
        assert terminal["progress"] == {"done": 3, "total": 3, "unit": "trials"}

        rows = client.get(f"/api/v1/runs/{job_id}/trials?limit=2")
        assert rows.status_code == 200
        assert rows.json()["total"] == 3
        assert len(rows.json()["rows"]) == 2
        assert all(row["timing_status"] == "recorded" for row in rows.json()["rows"])
        assert all(row["duration_seconds"] > 0 for row in rows.json()["rows"])

        result = client.get(f"/api/v1/runs/{job_id}/result")
        assert result.status_code == 200
        assert result.json()["duration_seconds"] is None
        exported = client.post(f"/api/v1/runs/{job_id}/exports")
        assert exported.status_code == 201
        downloaded = client.get(exported.json()["download_url"])
        assert downloaded.status_code == 200
        assert downloaded.json()["run_id"] == job_id
        report_export = client.post(f"/api/v1/runs/{job_id}/exports", json={"kind": "report"})
        assert report_export.status_code == 201
        assert "TuneLab experiment report" in client.get(report_export.json()["download_url"]).text
        config_export = client.post(f"/api/v1/runs/{job_id}/exports", json={"kind": "config"})
        assert config_export.status_code == 201
        assert client.get(config_export.json()["download_url"]).json()["budget"] == 3

        events = client.get(f"/api/v1/jobs/{job_id}/events").text
        assert "event: queued" in events
        assert "event: succeeded" in events
        resumed_events = client.get(
            f"/api/v1/jobs/{job_id}/events", headers={"Last-Event-ID": "1"}
        ).text
        assert "event: queued" not in resumed_events
        assert "event: succeeded" in resumed_events

        comparison = client.post("/api/v1/comparisons", json={"run_ids": [job_id, job_id]})
        assert comparison.status_code == 422


def test_api_rejects_a_partial_objective_space(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "runtime")) as client:
        response = client.post(
            "/api/v1/experiment-sets",
            json={
                "name": "bad wine space",
                "objective": "wine",
                "algorithms": ["grid"],
                "budget": 2,
                "space": [{"name": "max_depth", "kind": "int", "low": 2, "high": 12, "step": 2}],
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"]["code"] == "INVALID_EXPERIMENT"


def test_comparison_reports_seed_aggregate_and_tutorial_failures(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "runtime")) as client:
        created = client.post(
            "/api/v1/experiment-sets",
            json={
                "name": "surface failure tutorial",
                "objective": "surface",
                "algorithms": ["grid", "random"],
                "budget": 8,
                "search_seed": 17,
                "tutorial_failure_mode": True,
            },
        )
        assert created.status_code == 202
        job_ids = [item["job_id"] for item in created.json()["jobs"]]
        assert all(_terminal(client, job_id)["status"] == "succeeded" for job_id in job_ids)
        failed_rows = client.get(f"/api/v1/runs/{job_ids[0]}/trials?limit=200").json()["rows"]
        assert any(row["state"] == "failed" and "Tutorial failure" in row["error"] for row in failed_rows)
        comparison = client.post("/api/v1/comparisons", json={"run_ids": job_ids})
        assert comparison.status_code == 200
        aggregate = comparison.json()["aggregate"]
        assert {item["algorithm"] for item in aggregate} == {"grid", "random"}
        assert all(item["count"] == 1 and item["population_std_best_score"] == 0.0 for item in aggregate)


def test_api_pauses_at_a_real_trial_boundary_and_resumes(tmp_path, monkeypatch) -> None:
    def slow_surface(_self, params):
        time.sleep(0.05)
        return ObjectiveResult(score=float(params["depth"]))

    monkeypatch.setattr(SurfaceObjective, "__call__", slow_surface)
    with TestClient(create_app(tmp_path / "runtime")) as client:
        created = client.post(
            "/api/v1/experiment-sets",
            json={
                "name": "pause smoke",
                "objective": "surface",
                "algorithms": ["random"],
                "budget": 5,
            },
        )
        job_id = created.json()["jobs"][0]["job_id"]
        for _ in range(100):
            if client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running":
                break
            time.sleep(0.01)
        paused = client.post(f"/api/v1/jobs/{job_id}/pause")
        assert paused.status_code == 200
        terminal = _terminal(client, job_id)
        assert terminal["status"] == "paused"
        assert 1 <= terminal["progress"]["done"] < 5
        resumed = client.post(f"/api/v1/jobs/{job_id}/resume")
        assert resumed.status_code == 200
        assert _terminal(client, job_id)["status"] == "succeeded"
