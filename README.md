# TuneLab

TuneLab turns a search space and an objective function into a reproducible experiment: every trial is recorded, checkpoints are atomic, interrupted runs resume, and the result is readable as JSON, CSV, or a self-contained HTML report.

It is deliberately small and transparent. The package is useful when you need to compare search strategies without hiding the dataset, seed, metric, or failed evaluations behind a dashboard.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run tunelab demo --output-dir outputs/demo
open outputs/demo/benchmark.html
```

The demo evaluates the same deterministic synthetic objective with grid, random, TPE-like, and genetic search. It writes one report per strategy plus `benchmark.json`, `benchmark.csv`, and `benchmark.html`. Be clear about what it shows: the surface is clamped at 1.0, and at the default 48 trials **all four strategies reach exactly 1.0**. The demo proves the orchestration runs end to end and reproduces byte for byte. It does not rank the strategies, and nothing here could.

Run a single experiment:

```bash
uv run tunelab run \
  --objective wine \
  --algorithm tpe \
  --trials 12 \
  --seed 7 \
  --output-dir outputs/wine-tpe
```

The command creates `checkpoint.json` by default. Re-running the same command resumes from that checkpoint and writes `run.json`, `trials.csv`, and `report.html`.

To exercise the recovery path intentionally:

```bash
uv run tunelab recovery-demo --trials 8 --fail-after 3 \
  --output-dir outputs/recovery-demo
```

The command intentionally stops after the third new trial, then resumes the
same random sequence from its atomic checkpoint. It leaves `checkpoint.json`,
`observations.json`, canonical result, trial CSV and HTML report together so a
reviewer can inspect recovery rather than trust a sentence about it.
