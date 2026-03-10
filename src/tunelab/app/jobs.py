"""Persistent one-worker job orchestration for the TuneLab workbench."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

from ..models import RunResult, Trial
from ..report import write_report
from ..runner import run_experiment


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    job_id: str
    experiment_set_id: str
    status: str
    done: int
    total: int
    error: str | None
    revision: int
    config: dict[str, Any]


class JobStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    experiment_set_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    desired_action TEXT,
                    done INTEGER NOT NULL,
                    total INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    media_type TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "UPDATE jobs SET status = 'interrupted', desired_action = NULL, "
                "revision = revision + 1, updated_at = ? "
                "WHERE status IN ('queued', 'running', 'pausing', 'cancelling')",
                (time(),),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _event(self, connection: sqlite3.Connection, job_id: str, kind: str, payload: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO job_events(job_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (job_id, kind, json.dumps(payload, ensure_ascii=False, sort_keys=True), time()),
        )

    def create(self, experiment_set_id: str, config: dict[str, Any], output_dir: Path) -> JobSnapshot:
        job_id = f"job_{uuid.uuid4().hex}"
        now = time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, 'queued', NULL, 0, ?, 1, ?, ?, NULL, ?, ?)",
                (
                    job_id,
                    experiment_set_id,
                    int(config["budget"]),
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                    str(output_dir),
                    now,
                    now,
                ),
            )
            self._event(connection, job_id, "queued", {"done": 0, "total": int(config["budget"])})
        return self.get(job_id)

    def get(self, job_id: str) -> JobSnapshot:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return JobSnapshot(
            job_id=row["job_id"],
            experiment_set_id=row["experiment_set_id"],
            status=row["status"],
            done=int(row["done"]),
            total=int(row["total"]),
            error=row["error"],
            revision=int(row["revision"]),
            config=json.loads(row["config_json"]),
        )

    def list(self) -> list[JobSnapshot]:
        with self._connect() as connection:
            ids = [row["job_id"] for row in connection.execute("SELECT job_id FROM jobs ORDER BY created_at DESC")]
        return [self.get(job_id) for job_id in ids]

    def output_dir(self, job_id: str) -> Path:
        with self._connect() as connection:
            row = connection.execute("SELECT output_dir FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return Path(row["output_dir"])

    def events(self, job_id: str, after: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT seq, kind, payload_json FROM job_events WHERE job_id = ? AND seq > ? ORDER BY seq",
                (job_id, after),
            ).fetchall()
        return [
            {"seq": int(row["seq"]), "kind": row["kind"], "payload": json.loads(row["payload_json"])}
            for row in rows
        ]

    def register_artifact(self, job_id: str, path: Path, filename: str, media_type: str) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_id = f"artifact_{uuid.uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)",
                (artifact_id, job_id, str(path), filename, media_type),
            )
        return artifact_id

    def artifact(self, artifact_id: str) -> dict[str, str]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return {key: str(row[key]) for key in ("job_id", "path", "filename", "media_type")}

    def set_running(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'running', desired_action = NULL, revision = revision + 1, "
                "updated_at = ? WHERE job_id = ?",
                (time(), job_id),
            )
            self._event(connection, job_id, "running", {})

    def progress(self, job_id: str, trial: Trial) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT desired_action, total FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            connection.execute(
                "UPDATE jobs SET done = ?, updated_at = ? WHERE job_id = ?",
                (trial.number + 1, time(), job_id),
            )
            self._event(
                connection,
                job_id,
                "trial",
                {"done": trial.number + 1, "total": int(row["total"]), "state": trial.state, "number": trial.number},
            )
            return row["desired_action"]

    def finish(self, job_id: str, result: RunResult) -> None:
        kind = {"completed": "succeeded", "paused": "paused", "cancelled": "cancelled"}.get(result.status, "failed")
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, desired_action = NULL, done = ?, revision = revision + 1, "
                "updated_at = ? WHERE job_id = ?",
                (kind, len(result.trials), time(), job_id),
            )
            self._event(connection, job_id, kind, {"done": len(result.trials), "total": result.max_trials})

    def fail(self, job_id: str, message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'failed', desired_action = NULL, error = ?, revision = revision + 1, "
                "updated_at = ? WHERE job_id = ?",
                (message, time(), job_id),
            )
            self._event(connection, job_id, "failed", {"message": message})

    def request(self, job_id: str, action: str) -> JobSnapshot:
        snapshot = self.get(job_id)
        if action == "pause" and snapshot.status == "running":
            next_status, desired, event = "pausing", "pause", "pausing"
        elif action == "resume" and snapshot.status in {"paused", "interrupted"}:
            next_status, desired, event = "queued", None, "queued"
        elif action == "cancel" and snapshot.status == "running":
            next_status, desired, event = "cancelling", "cancel", "cancelling"
        elif action == "cancel" and snapshot.status in {"queued", "paused", "interrupted"}:
            next_status, desired, event = "cancelled", None, "cancelled"
        else:
            raise ValueError(f"Cannot {action} a {snapshot.status} job")
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, desired_action = ?, revision = revision + 1, "
                "updated_at = ? WHERE job_id = ?",
                (next_status, desired, time(), job_id),
            )
            self._event(connection, job_id, event, {})
        return self.get(job_id)


ObjectiveFactory = Callable[[dict[str, Any]], tuple[Callable[[dict[str, Any]], Any], list[Any], str]]


class TuneJobs:
    """Executes one persisted experiment at a time without blocking the API loop."""

    def __init__(self, root: Path, objective_factory: ObjectiveFactory) -> None:
        self.root = root
        self.store = JobStore(root / "state.sqlite")
        self.objective_factory = objective_factory
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=5)
        self.worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.worker = asyncio.create_task(self._work())

    async def stop(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass

    async def enqueue(self, job_id: str) -> None:
        self.queue.put_nowait(job_id)

    async def _work(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self._run(job_id)
            finally:
                self.queue.task_done()

    async def _run(self, job_id: str) -> None:
        snapshot = self.store.get(job_id)
        if snapshot.status != "queued":
            return
        self.store.set_running(job_id)
        config = snapshot.config
        output = self.store.output_dir(job_id)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            objective, specs, objective_name = self.objective_factory(config)

            def control(trial: Trial, _trials: tuple[Trial, ...]) -> str | None:
                return self.store.progress(job_id, trial)

            result = await asyncio.to_thread(
                run_experiment,
                run_id=job_id,
                objective=objective,
                objective_name=objective_name,
                specs=specs,
                algorithm=config["algorithm"],
                max_trials=int(config["budget"]),
                seed=int(config["search_seed"]),
                checkpoint_path=output / "checkpoint.json",
                control=control,
            )
        except Exception as exc:
            self.store.fail(job_id, f"{type(exc).__name__}: {exc}")
            return
        if result.status == "completed":
            try:
                write_report(result, output)
            except Exception as exc:
                self.store.fail(job_id, f"ArtifactError: {type(exc).__name__}: {exc}")
                return
        self.store.finish(job_id, result)
