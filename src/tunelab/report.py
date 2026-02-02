from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .models import RunResult


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _svg_convergence(result: RunResult) -> str:
    points: list[tuple[int, float]] = []
    best: float | None = None
    for trial in result.trials:
        if trial.score is None:
            continue
        best = trial.score if best is None else max(best, trial.score)
        points.append((trial.number, best))
    if not points:
        return (
            '<svg viewBox="0 0 640 220" role="img" aria-label="No completed trials">'
            '<text x="20" y="40">No completed trials</text></svg>'
        )
    width, height, margin = 640, 220, 34
    min_value = min(value for _index, value in points)
    max_value = max(value for _index, value in points)
    spread = max(max_value - min_value, 1e-6)
    max_index = max(points[-1][0], 1)
    coords = [
        (
            margin + index / max_index * (width - 2 * margin),
            height - margin - (value - min_value) / spread * (height - 2 * margin),
        )
        for index, value in points
    ]
    path = " ".join(
        ("M" if position == 0 else "L") + f"{x:.1f},{y:.1f}"
        for position, (x, y) in enumerate(coords)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Best score by trial">'
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#cbd5e1"/>'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#cbd5e1"/>'
        f'<path d="{path}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<text x="{margin}" y="18" fill="#475569" font-size="12">best score</text>'
        f'<text x="{width-margin-80}" y="{height-8}" fill="#475569" font-size="12">trial</text>'
        "</svg>"
    )


def write_report(result: RunResult, output_dir: Path) -> dict[str, str]:
    """Write canonical artifacts without machine-local observation timing."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    payload["duration_seconds"] = None
    payload["mean_trial_seconds"] = None
    for trial in payload["trials"]:
        trial["duration_seconds"] = None
    _atomic_text(output_dir / "run.json", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    trial_path = output_dir / "trials.csv"
    columns = ["trial", "state", "score", "duration_seconds", "params", "metrics", "error"]
    with trial_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for trial in result.trials:
            writer.writerow(
                {
                    "trial": trial.number,
                    "state": trial.state,
                    "score": "" if trial.score is None else f"{trial.score:.8f}",
                    "duration_seconds": "unrecorded",
                    "params": json.dumps(trial.params, ensure_ascii=False, sort_keys=True),
                    "metrics": json.dumps(trial.metrics, ensure_ascii=False, sort_keys=True),
                    "error": trial.error or "",
                }
            )

    best = result.best_trial
    best_text = f"{result.best_score:.4f}" if result.best_score is not None else "—"
    best_params = html.escape(json.dumps(best.params if best else {}, ensure_ascii=False, sort_keys=True, indent=2))
    rows_parts: list[str] = []
    for trial in result.trials:
        score_text = "—" if trial.score is None else f"{trial.score:.4f}"
        rows_parts.append(
            "<tr>"
            f"<td>{trial.number}</td><td>{html.escape(trial.state)}</td>"
            f"<td>{score_text}</td>"
            f"<td><code>{html.escape(json.dumps(trial.params, ensure_ascii=False, sort_keys=True))}</code></td>"
            f"<td>{html.escape(trial.error or '')}</td>"
            "</tr>"
        )
    rows = "".join(rows_parts)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TuneLab · {html.escape(result.run_id)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin:0; background:#f6f8fc; color:#172033; }} main {{ max-width:1080px; margin:auto; padding:38px 22px 64px; }}
h1 {{ margin:0 0 8px; letter-spacing:-.025em; }} h2 {{ margin-top:34px; }} .lede,.meta {{ color:#526071; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:24px 0; }}
.card {{ background:white; border:1px solid #dfe6f1; border-radius:14px; padding:16px; }}
.label {{ color:#667085; font-size:.8rem; }} .value {{ font-size:1.75rem; font-weight:750; margin-top:5px; }}
.panel {{ background:white; border:1px solid #dfe6f1; border-radius:14px; padding:16px; overflow:auto; }}
svg {{ width:100%; max-height:220px; }} table {{ width:100%; border-collapse:collapse; font-size:.88rem; }} th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid #edf1f6; vertical-align:top; }} th {{ background:#f1f5fa; color:#475467; }} code {{ white-space:pre-wrap; word-break:break-word; }}
.note {{ background:#fff8e7; border:1px solid #f3d58f; border-radius:12px; padding:13px 15px; }} footer {{ color:#7b8494; font-size:.8rem; margin-top:38px; }}
</style></head><body><main>
<h1>TuneLab experiment report</h1>
<p class="lede">A deterministic, resumable comparison of transparent search strategies. Scores come from the configured objective; inspect the run manifest before comparing experiments.</p>
<p class="meta">Run <code>{html.escape(result.run_id)}</code> · objective <code>{html.escape(result.objective)}</code> · algorithm <code>{html.escape(result.algorithm)}</code> · seed {result.seed}</p>
<section class="cards"><div class="card"><div class="label">Best score</div><div class="value">{best_text}</div></div>
<div class="card"><div class="label">Completed</div><div class="value">{len(result.completed_trials)}/{len(result.trials)}</div></div>
<div class="card"><div class="label">Failed</div><div class="value">{len(result.failed_trials)}</div></div>
<div class="card"><div class="label">Timing</div><div class="value">observations</div></div></section>
<h2>Convergence</h2><div class="panel">{_svg_convergence(result)}</div>
<h2>Best parameters</h2><div class="panel"><code>{best_params}</code></div>
<h2>Trials</h2><div class="panel"><table><thead><tr><th>#</th><th>State</th><th>Score</th><th>Parameters</th><th>Error</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="note">A benchmark score is meaningful only with its dataset, seed, split, metric, and code version. This report does not claim that one strategy wins outside the displayed objective.</p>
<footer>Generated by TuneLab · independent implementation</footer>
</main></body></html>"""
    html_path = output_dir / "report.html"
    _atomic_text(html_path, document)
    return {"run": str(output_dir / "run.json"), "trials": str(trial_path), "html": str(html_path)}


def write_benchmark_index(results: list[RunResult], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        rows.append(
            {
                "algorithm": result.algorithm,
                "best_score": result.best_score,
                "completed_trials": len(result.completed_trials),
                "failed_trials": len(result.failed_trials),
                "duration_seconds": None,
            }
        )
    _atomic_text(output_dir / "benchmark.json", json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    csv_path = output_dir / "benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["algorithm"])
        writer.writeheader()
        writer.writerows(rows)
    table_parts: list[str] = []
    for row in rows:
        score = row["best_score"]
        score_text = "—" if score is None else f"{float(score):.4f}"
        table_parts.append(
            f"<tr><td>{html.escape(str(row['algorithm']))}</td>"
            f"<td>{score_text}</td><td>{row['completed_trials']}</td>"
            f"<td>{row['failed_trials']}</td><td>observations</td></tr>"
        )
    table = "".join(table_parts)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TuneLab benchmark</title><style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#172033}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e5e7eb;text-align:left}}th{{background:#f1f5f9}}</style></head><body><h1>TuneLab strategy benchmark</h1><p>Same objective, seed, and trial budget; compare the full manifests before drawing conclusions.</p><table><thead><tr><th>Algorithm</th><th>Best score</th><th>Completed</th><th>Failed</th><th>Duration</th></tr></thead><tbody>{table}</tbody></table></body></html>"""
    html_path = output_dir / "benchmark.html"
    _atomic_text(html_path, page)
    return {"json": str(output_dir / "benchmark.json"), "csv": str(csv_path), "html": str(html_path)}
