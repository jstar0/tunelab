"""Local HTTP API for real TuneLab experiment lifecycles."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..models import ParamSpec
from ..objective import IrisObjective, SurfaceObjective, WineObjective
from .jobs import JobSnapshot, TuneJobs


class ParamPayload(BaseModel):
    name: str
    kind: Literal["int", "float", "categorical", "bool"]
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    choices: list[Any] = Field(default_factory=list)
    log_scale: bool = False

    def to_spec(self) -> ParamSpec:
        return ParamSpec(
            self.name,
            self.kind,
            low=self.low,
            high=self.high,
            step=self.step,
            choices=tuple(self.choices),
            log_scale=self.log_scale,
        )


class ExperimentSetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    objective: Literal["surface", "iris", "wine"] = "wine"
    algorithms: list[Literal["grid", "random", "tpe", "genetic"]] = Field(min_length=1, max_length=4)
    budget: int = Field(ge=1, le=500)
    search_seed: int = 7
    objective_seed: int = 17
    folds: int = Field(default=5, ge=2, le=10)
    space: list[ParamPayload] | None = None
    tutorial_failure_mode: bool = False


class ComparisonRequest(BaseModel):
    run_ids: list[str] = Field(min_length=2, max_length=4)


class ExportRequest(BaseModel):
    kind: Literal["canonical", "observations", "trials", "report", "config"] = "canonical"


def error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, {"error": {"code": code, "message": message, "retryable": False}})


def _objective(config: dict[str, Any]) -> tuple[Any, list[ParamSpec], str]:
    name = config["objective"]
    if name == "surface":
        objective: Any = SurfaceObjective(
            fail_on_activation="gelu" if config.get("tutorial_failure_mode") else None
        )
    elif name == "iris":
        objective = IrisObjective(seed=int(config["objective_seed"]), folds=int(config["folds"]))
    else:
        objective = WineObjective(seed=int(config["objective_seed"]), folds=int(config["folds"]))
    specs = [ParamSpec(**item) for item in config.get("space", [])] or list(objective.specs)
    expected = {spec.name for spec in objective.specs}
    if {spec.name for spec in specs} != expected:
        raise ValueError("Search space must declare every parameter accepted by the selected objective")
    return objective, specs, objective.name


def _job_payload(snapshot: JobSnapshot) -> dict[str, Any]:
    capabilities = {
        "pause": snapshot.status == "running",
        "resume": snapshot.status in {"paused", "interrupted"},
        "cancel": snapshot.status in {"queued", "running", "paused", "interrupted"},
    }
    return {
        "job_id": snapshot.job_id,
        "experiment_set_id": snapshot.experiment_set_id,
        "status": snapshot.status,
        "revision": snapshot.revision,
        "config": snapshot.config,
        "capabilities": capabilities,
        "progress": {"done": snapshot.done, "total": snapshot.total, "unit": "trials"},
        "error": snapshot.error,
    }


def create_app(root: Path | None = None) -> FastAPI:
    runtime = root or Path("runtime/tunelab")
    jobs = TuneJobs(runtime, _objective)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await jobs.start()
        try:
            yield
        finally:
            await jobs.stop()

    app = FastAPI(title="TuneLab", version="0.1.0", lifespan=lifespan)

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/objectives")
    def objectives() -> dict[str, Any]:
        payload = []
        for key, objective in (("surface", SurfaceObjective()), ("iris", IrisObjective()), ("wine", WineObjective())):
            payload.append(
                {
                    "id": key,
                    "name": objective.name,
                    "default_space": [spec.as_dict() for spec in objective.specs],
                    "default_folds": None if key == "surface" else 5,
                    "description": (
                        "Mechanism-only synthetic surface"
                        if key == "surface"
                        else f"Public {key.title()} RandomForest CV"
                    ),
                }
            )
        return {"objectives": payload}

    @app.post("/api/v1/experiment-sets", status_code=202)
    async def create_experiment_set(request: ExperimentSetRequest) -> dict[str, Any]:
        if len(set(request.algorithms)) != len(request.algorithms):
            raise error(422, "DUPLICATE_ALGORITHM", "Choose each search strategy at most once.")
        try:
            specs = [item.to_spec() for item in request.space] if request.space else []
            base = request.model_dump()
            base["space"] = [spec.as_dict() for spec in specs]
            _objective(base)
        except ValueError as exc:
            raise error(422, "INVALID_EXPERIMENT", str(exc)) from exc
        experiment_set_id = f"set_{uuid.uuid4().hex}"
        created = []
        for algorithm in request.algorithms:
            config = {**base, "algorithm": algorithm}
            output_dir = runtime / "workspaces" / experiment_set_id / algorithm
            snapshot = jobs.store.create(experiment_set_id, config, output_dir)
            try:
                await jobs.enqueue(snapshot.job_id)
            except asyncio.QueueFull as exc:
                jobs.store.request(snapshot.job_id, "cancel")
                raise error(503, "QUEUE_FULL", "At most five experiments may wait for the local worker.") from exc
            created.append(_job_payload(snapshot))
        return {"experiment_set_id": experiment_set_id, "jobs": created}

    @app.get("/api/v1/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": [_job_payload(snapshot) for snapshot in jobs.store.list()]}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return _job_payload(jobs.store.get(job_id))
        except KeyError as exc:
            raise error(404, "JOB_NOT_FOUND", "Experiment run does not exist.") from exc

    @app.get("/api/v1/runs/{job_id}/trials")
    def trials(
        job_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=200),
    ) -> dict[str, Any]:
        try:
            directory = jobs.store.output_dir(job_id)
        except KeyError as exc:
            raise error(404, "JOB_NOT_FOUND", "Experiment run does not exist.") from exc
        checkpoint = directory / "checkpoint.json"
        if not checkpoint.exists():
            return {"job_id": job_id, "rows": [], "total": 0}
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        rows = payload.get("trials", [])
        observed: dict[int, float] = {}
        observations_path = directory / "observations.json"
        if observations_path.exists():
            observation_payload = json.loads(observations_path.read_text(encoding="utf-8"))
            observed = {
                int(row["number"]): float(row["duration_seconds"])
                for row in observation_payload.get("trials", [])
                if isinstance(row, dict)
                and type(row.get("number")) is int
                and isinstance(row.get("duration_seconds"), (int, float))
            }
        display_rows = []
        for row in rows:
            number = int(row["number"])
            display_rows.append(
                {
                    **row,
                    "duration_seconds": observed.get(number),
                    "timing_status": "recorded" if number in observed else "unrecorded",
                }
            )
        return {"job_id": job_id, "rows": display_rows[offset : offset + limit], "total": len(rows)}

    @app.get("/api/v1/runs/{job_id}/result")
    def result(job_id: str) -> dict[str, Any]:
        try:
            directory = jobs.store.output_dir(job_id)
            snapshot = jobs.store.get(job_id)
        except KeyError as exc:
            raise error(404, "JOB_NOT_FOUND", "Experiment run does not exist.") from exc
        run_path = directory / "run.json"
        if snapshot.status != "succeeded" or not run_path.exists():
            raise error(409, "RESULT_NOT_READY", "This experiment has not completed.")
        return json.loads(run_path.read_text(encoding="utf-8"))

    @app.post("/api/v1/comparisons")
    def compare(request: ComparisonRequest) -> dict[str, Any]:
        if len(set(request.run_ids)) != len(request.run_ids):
            raise error(422, "DUPLICATE_RUN", "Choose each run at most once.")
        snapshots: list[JobSnapshot] = []
        for job_id in request.run_ids:
            try:
                snapshots.append(jobs.store.get(job_id))
            except KeyError as exc:
                raise error(404, "JOB_NOT_FOUND", "One selected experiment does not exist.") from exc
        reference = snapshots[0].config
        compared_fields = (
            "objective",
            "objective_seed",
            "folds",
            "space",
            "budget",
            "tutorial_failure_mode",
        )
        reasons = [
            field
            for field in compared_fields
            if any(snapshot.config.get(field) != reference.get(field) for snapshot in snapshots[1:])
        ]
        rows = []
        for snapshot in snapshots:
            directory = jobs.store.output_dir(snapshot.job_id)
            run_path = directory / "run.json"
            payload = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else None
            rows.append(
                {
                    "job_id": snapshot.job_id,
                    "algorithm": snapshot.config["algorithm"],
                    "search_seed": snapshot.config["search_seed"],
                    "status": snapshot.status,
                    "budget": snapshot.total,
                    "completed_trials": snapshot.done,
                    "failed_trials": 0 if payload is None else payload["failed_trials"],
                    "best_score": None if payload is None else payload["best_score"],
                    "convergence": [] if payload is None else payload["convergence"],
                    "config": {field: snapshot.config.get(field) for field in compared_fields},
                }
            )
        grouped: dict[str, list[float]] = {}
        for row in rows:
            if row["best_score"] is not None:
                grouped.setdefault(str(row["algorithm"]), []).append(float(row["best_score"]))
        aggregate = [
            {
                "algorithm": algorithm,
                "count": len(scores),
                "mean_best_score": round(fmean(scores), 8),
                "population_std_best_score": round(pstdev(scores), 8) if len(scores) > 1 else 0.0,
            }
            for algorithm, scores in sorted(grouped.items())
        ]
        return {
            "comparable": not reasons,
            "reasons": reasons,
            "runs": rows,
            "aggregate": aggregate,
            "note": "Same-budget comparison" if not reasons else "Runs are shown side by side without a winner claim.",
        }

    @app.post("/api/v1/jobs/{job_id}/{action}")
    async def control(job_id: str, action: Literal["pause", "resume", "cancel"]) -> dict[str, Any]:
        try:
            snapshot = jobs.store.request(job_id, action)
        except KeyError as exc:
            raise error(404, "JOB_NOT_FOUND", "Experiment run does not exist.") from exc
        except ValueError as exc:
            raise error(409, "INVALID_JOB_STATE", str(exc)) from exc
        if action == "resume":
            try:
                await jobs.enqueue(job_id)
            except asyncio.QueueFull as exc:
                raise error(503, "QUEUE_FULL", "The local worker queue is full.") from exc
        return _job_payload(snapshot)

    @app.get("/api/v1/jobs/{job_id}/events")
    async def events(
        job_id: str,
        after: int = 0,
        last_event_id: int | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            jobs.store.get(job_id)
        except KeyError as exc:
            raise error(404, "JOB_NOT_FOUND", "Experiment run does not exist.") from exc

        async def stream() -> AsyncIterator[str]:
            cursor = max(after, last_event_id or 0)
            while True:
                for event in jobs.store.events(job_id, cursor):
                    cursor = event["seq"]
                    yield f"id: {cursor}\nevent: {event['kind']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                if jobs.store.get(job_id).status in {"succeeded", "failed", "cancelled", "paused", "interrupted"}:
                    return
                await asyncio.sleep(0.15)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/v1/runs/{job_id}/exports", status_code=201)
    def export(job_id: str, request: ExportRequest | None = None) -> dict[str, str]:
        try:
            snapshot = jobs.store.get(job_id)
            directory = jobs.store.output_dir(job_id)
        except KeyError as exc:
            raise error(404, "JOB_NOT_FOUND", "Experiment run does not exist.") from exc
        if snapshot.status != "succeeded":
            raise error(409, "RESULT_NOT_READY", "Only completed experiments can be exported.")
        files = {
            "canonical": ("run.json", "application/json"),
            "observations": ("observations.json", "application/json"),
            "trials": ("trials.csv", "text/csv; charset=utf-8"),
            "report": ("report.html", "text/html; charset=utf-8"),
            "config": ("config.json", "application/json"),
        }
        kind = request.kind if request is not None else "canonical"
        filename, media_type = files[kind]
        if kind == "config" and not (directory / filename).exists():
            (directory / filename).write_text(
                json.dumps(snapshot.config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        try:
            artifact_id = jobs.store.register_artifact(
                job_id,
                directory / filename,
                f"tunelab-{job_id[-8:]}-{filename}",
                media_type,
            )
        except FileNotFoundError as exc:
            raise error(409, "ARTIFACT_NOT_READY", "The selected artifact is not available yet.") from exc
        return {"artifact_id": artifact_id, "download_url": f"/api/v1/artifacts/{artifact_id}/download"}

    @app.get("/api/v1/artifacts/{artifact_id}/download")
    def download(artifact_id: str) -> FileResponse:
        try:
            artifact = jobs.store.artifact(artifact_id)
        except KeyError as exc:
            raise error(404, "ARTIFACT_NOT_FOUND", "Export artifact does not exist.") from exc
        return FileResponse(artifact["path"], media_type=artifact["media_type"], filename=artifact["filename"])

    web_dist = Path(__file__).resolve().parents[3] / "web" / "dist"
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="workbench")
    return app
