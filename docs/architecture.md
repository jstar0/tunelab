# TuneLab architecture

TuneLab has four intentionally boring layers:

1. `models.py` defines serializable value objects (`ParamSpec`, `ObjectiveResult`, `Trial`, and `RunResult`). Validation happens at the boundary so reports never need to guess what a trial means.
2. `space.py` turns a list of parameter declarations into grid candidates or seeded proposals. Adaptive proposals consume prior completed trials but never mutate them.
3. `runner.py` owns the experiment loop. It derives a deterministic random stream from `(seed, trial_number)`, evaluates one candidate, appends either a completed or failed `Trial`, and atomically replaces the checkpoint after each evaluation.
4. `report.py` renders the canonical `RunResult` into machine-readable artifacts and a static HTML page. Wall-clock observations are deliberately kept out of these canonical files.
5. `app/jobs.py` owns the local SQLite job/event log and one-worker queue. `app/main.py` exposes the task, trial, comparison and artifact-ID API; `web/` is a React client that never submits server filesystem paths.

## Resume semantics

The v2 checkpoint stores a schema version, a SHA-256 digest of the run configuration, the declared search space, and canonical trial rows. A resumed invocation must use the same run ID, objective name, algorithm, seed, budget, and search space. Parameter restoration validates type, bounds, step, log domain and choices directly; it does not materialize a whole grid for continuous random/TPE-like/genetic runs. A per-trial random stream means a resumed run proposes the same next candidate as an uninterrupted run even when an adaptive sampler uses a variable number of random draws.

Actual evaluation duration and start/end timestamps go to `observations.json`. A missing observation is shown as `未记录`; a legacy v1 zero is never reinterpreted as a measured zero. The worker writes the observation and canonical checkpoint around every completed trial, so a missing observation after an interruption is visible without invalidating the experiment facts.

## Failure semantics

An exception raised by an objective is converted into a failed trial with its type and message. The loop continues so a flaky candidate is visible rather than silently retried. The Surface-only failed-trial tutorial explicitly configures GELU candidates to raise, and records that fact in the frozen run configuration; it is a visible UI mechanism exercise, never a benchmark. Pause and cancel requests are read after each persisted trial, so the interface never claims to stop a currently executing RandomForest fit. `InjectedFailure` is only a demonstration hook: it is raised after the checkpoint is written and is never used by the search core.

## Comparison semantics

The benchmark index puts strategies side by side, but it does not rank them universally. The comparison API checks objective, objective/data seed, CV folds, full space, budget and tutorial mode before it permits a same-budget reading. A mismatch stays visible as a reason and the runs remain side by side without a winner claim. When selected completed runs share an algorithm, the API returns their mean best score and population standard deviation together with the number of seeds. The HTML report repeats the comparison warning because a single best score is easy to over-interpret.
