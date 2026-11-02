"""Record real TuneLab API responses for the static replay build.

Starts the real local service on a temporary runtime directory, creates the
experiment sets below through the same endpoints the workbench uses, waits for
every run to finish and writes the unmodified responses to ``web/src/replay/``.

    uv run python scripts/record_replay.py
"""

from __future__ import annotations

import itertools
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "src" / "replay"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


PORT = _free_port()
BASE = f"http://127.0.0.1:{PORT}/api/v1"
ALGORITHMS = ["grid", "random", "tpe", "genetic"]


def _set(label: str, objective: str, seed: int, folds: int, **extra: object) -> dict:
    return {
        "name": f"{label} · 48 trials",
        "objective": objective,
        "budget": 48,
        "search_seed": seed,
        "objective_seed": 17,
        "folds": folds,
        **extra,
    }


# Surface uses folds=3 because that is what the workbench sends after switching to it.
# Each set is run with every strategy so that any strategy subset chosen in the
# workbench maps onto recorded runs.
SETS = [
    _set("Wine", "wine", 7, 5),
    _set("Wine", "wine", 11, 5),
    _set("Wine", "wine", 13, 5),
    _set("Iris", "iris", 7, 5),
    _set("Surface", "surface", 7, 3),
    _set("Surface", "surface", 7, 3, tutorial_failure_mode=True),
]
TERMINAL = {"succeeded", "failed", "cancelled"}


def call(method: str, path: str, body: dict | None = None, raw: bool = False):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    return payload if raw else json.loads(payload)


def wait_for_server(process: subprocess.Popen) -> None:
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("service exited early")
        try:
            call("GET", "/health")
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.2)
    raise RuntimeError("service did not start")


def parse_events(raw: bytes) -> list[dict]:
    events = []
    for block in raw.decode().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" in fields:
            events.append({"id": int(fields["id"]), "event": fields["event"], "data": json.loads(fields["data"])})
    return events


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix="tunelab-replay-"))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tunelab.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--runtime-dir",
            str(runtime),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(process)
        objectives = call("GET", "/objectives")
        records: list[dict] = []
        for config in SETS:
            created = call("POST", "/experiment-sets", {**config, "algorithms": ALGORITHMS})
            for job in created["jobs"]:
                while True:
                    state = call("GET", f"/jobs/{job['job_id']}")
                    if state["status"] in TERMINAL:
                        break
                    time.sleep(0.5)
                if state["status"] != "succeeded":
                    raise RuntimeError(f"{job['job_id']} ended as {state['status']}: {state['error']}")
                trials = call("GET", f"/runs/{job['job_id']}/trials?limit=200")
                events = parse_events(call("GET", f"/jobs/{job['job_id']}/events", raw=True))
                exports = {}
                for kind in ("report", "trials", "config"):
                    artifact = call("POST", f"/runs/{job['job_id']}/exports", {"kind": kind})
                    exports[kind] = call("GET", artifact["download_url"].removeprefix("/api/v1"), raw=True).decode()
                records.append({"job": state, "trials": trials, "events": events, "exports": exports})
                print(
                    f"recorded {state['config']['objective']} {state['config']['algorithm']} "
                    f"seed {state['config']['search_seed']}",
                    flush=True,
                )
        # The service lists newest first; keep the recorded order stable instead.
        listed = {row["job_id"]: row for row in call("GET", "/runs")["runs"]}
        for record in records:
            assert listed[record["job"]["job_id"]] == record["job"]

        # Real comparison responses, split into per-run rows and per-selection summaries.
        groups: dict[tuple, list[str]] = {}
        for record in records:
            cfg = record["job"]["config"]
            groups.setdefault((cfg["objective"], cfg.get("tutorial_failure_mode")), []).append(record["job"]["job_id"])
        ids = [record["job"]["job_id"] for record in records]
        selections = {frozenset(pair) for pair in itertools.combinations(ids, 2)}
        for members in groups.values():
            for size in (3, 4):
                selections.update(frozenset(c) for c in itertools.combinations(members, size))
            selections.update(frozenset(c) for c in itertools.combinations(members, 2))
        run_rows: dict[str, dict] = {}
        summaries: dict[str, dict] = {}
        for selection in sorted(selections, key=sorted):
            order = [i for i in ids if i in selection]
            response = call("POST", "/comparisons", {"run_ids": order})
            for row in response["runs"]:
                previous = run_rows.setdefault(row["job_id"], row)
                assert previous == row, "comparison run row differs between responses"
            summaries[",".join(sorted(selection))] = {
                key: response[key] for key in ("comparable", "reasons", "note", "aggregate")
            }
        print(f"recorded {len(summaries)} comparison selections", flush=True)

        meta = {
            "source": "scripts/record_replay.py against the local TuneLab service",
            "recorded_at": time.strftime("%Y-%m-%d", time.gmtime()),
        }
        (OUT / "recording.json").write_text(
            json.dumps(
                {
                    "meta": meta,
                    "objectives": objectives,
                    "runs": [{k: r[k] for k in ("job", "trials", "events")} for r in records],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (OUT / "exports.json").write_text(
            json.dumps({r["job"]["job_id"]: r["exports"] for r in records}, ensure_ascii=False, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        (OUT / "comparisons.json").write_text(
            json.dumps({"rows": run_rows, "selections": summaries}, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    finally:
        process.terminate()
        process.wait(timeout=15)
    return 0


if __name__ == "__main__":
    sys.exit(main())
