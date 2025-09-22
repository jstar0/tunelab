# Design decisions

This document exists to be interrogated. Each entry names a decision that is
visible in the code, the alternative it rejected, and the cost it accepted. The
final section lists the weaknesses a reviewer should find, so nobody has to
discover them by reading between the lines.

## 1. What each strategy actually is

Naming is where a tuning project earns or loses trust, so this table is precise
about the implementation in [`src/tunelab/space.py`](../src/tunelab/space.py)
and [`src/tunelab/runner.py`](../src/tunelab/runner.py).

| Name in the CLI | What the code does | What it is **not** |
| --- | --- | --- |
| `grid` | `itertools.product` over each spec's `grid_values()`, consumed in order, skipping signatures already evaluated. | Not shuffled, not adaptive. |
| `random` | Independent draws per parameter: uniform over choices, uniform or log-uniform over ranges, snapped to `step`. | Not quasi-random; no Sobol or Latin hypercube. |
| `tpe` | Random for the first four trials, then: sort history by score, take the top 25% (minimum 2) as the good set, pick one member as an anchor, and propose a Gaussian neighbour per parameter with sigma = 0.18 x range (log-space for log-scaled parameters). | **Not a Tree-structured Parzen Estimator.** There is no density model of good and bad regions, no `l(x)/g(x)` ratio, and no expected-improvement acquisition. It is a good-set-anchored local sampler. The docstring says "inspired by TPE" for exactly this reason. |
| `genetic` | Random for the first four trials, then: take the top third of completed trials as parents (minimum 2), pick two at random, uniform crossover with a per-parameter coin flip, then mutate each parameter with probability 0.25 by resampling from its prior. | Not generational: there are no discrete generations and no explicit elitism step. The population is "all completed trials", so it is a steady-state GA. |

Rejected alternative: depend on Optuna or scikit-optimize and get real TPE and
real Bayesian optimisation. That would make the project a wrapper. The point
here is an inspectable orchestration layer, so every proposal policy is small
enough to read and to seed deterministically.

## 2. No configuration is ever evaluated twice

`_propose` computes a `params_signature` (sorted `(name, repr(value))` pairs)
for each candidate and retries up to 128 times to find an unseen one. When the
space is exhausted it returns `None`, which the runner treats as normal
completion rather than an error.

This makes the budget mean *unique configurations*, which is what makes a
cross-strategy comparison meaningful. It also flatters random search relative
to a naive implementation, because random search here never spends budget on a
repeat.

Cost accepted: on a small discrete space, a strategy can finish before
`max_trials`, so run records may show fewer trials than requested.

## 3. Two random generators, not one

The runner uses a separate `Random` for proposals from the one used inside an
objective. An adaptive strategy consumes a variable number of draws depending
on history, so sharing a single stream would make the objective's randomness
depend on the search path and destroy reproducibility.

## 4. A checkpoint is bound to a configuration digest, including the objective's internals

`_config_digest` hashes the run configuration: run id, objective name,
**objective fingerprint**, algorithm, seed, budget, and search space.
`_load_checkpoint` refuses a file whose `schema_version != 1` or whose digest
differs.

The fingerprint matters more than it looks. `objective_name` is a constant
string (`"iris-random-forest"`), and the cross-validation fold count lives
inside the objective, not in the search space. Before objectives exposed a
fingerprint, a checkpoint taken at 3 folds resumed without complaint under 9
folds, producing one run whose early trials were 3-fold scores and whose later
trials were 9-fold scores, compared against each other to pick a best trial.
`IrisObjective.fingerprint` is now `name:seed=...:folds=...`, and the runner
reads it with `getattr(objective, "fingerprint", objective_name)` so a plain
callable objective still works.

Because `max_trials` is inside the digest, a run's budget cannot be extended by
resuming: changing `--trials` raises rather than continuing. That is a
deliberate trade of flexibility for a strict contract.

Rejected alternative: resume on run id. Ids get reused during debugging;
digests cannot silently drift.

Proof: `test_checkpoint_rejects_a_changed_objective_configuration` and
`test_objective_without_a_fingerprint_still_works` in
[`tests/test_runner.py`](../tests/test_runner.py).

## 5. Checkpoint writes are atomic, and the rest are not

`_atomic_write_json` writes to a pid-suffixed temporary file in the destination
directory and `replace`s it. An interrupted write leaves the previous complete
checkpoint.

Be precise about the scope, because the word "atomic" is easy to over-apply
here. Atomic: `checkpoint.json`, `run.json`, `report.html`, `benchmark.json`,
`benchmark.html`. **Not** atomic: `trials.csv`, `sweep.json`, `sweep.csv`,
`report.md`, `manifest.json`. And `report.py`'s temporary name omits the pid
suffix that `runner.py` uses, so two concurrent writers into one directory can
collide there.

Nothing `fsync`s, so a hard power loss can still lose the most recent write.
The rename is atomic; durability is not guaranteed.

## 6. Trials record failure as data

A trial carries a `state` and an optional `score`. Every aggregation filters on
`state == "completed" and score is not None`, so a failing objective produces a
recorded failed trial and a visible success rate rather than a gap in a chart.

## 7. The objective boundary is a plain callable

`Objective = Callable[[dict[str, Any]], ObjectiveResult]`. A user objective
needs no base class and no framework, and `ObjectiveResult` carries both the
scalar the search optimises and a free-form metrics dict that flows into the
report.

## 8. Three objectives with different jobs

- `SurfaceObjective` is a closed-form deterministic surface used by tests and
  demos so the suite runs in milliseconds.
- `IrisObjective` is 3-fold stratified cross-validated RandomForest accuracy on
  scikit-learn's bundled Iris data, with `random_state` fixed on both the model
  and the splitter.
- `WineObjective` uses the same visible RandomForest search space and CV rule on
  scikit-learn's public Wine data. It is the default benchmark and sweep
  contract because it has a less saturated score surface at the same budget;
  Iris remains available as a compact reference.

Every run remains byte-reproducible. The sweep fixes the objective's data/model
seed at 2026 by default and varies only `run_experiment`'s search seed. That
makes all strategies see the same split and model randomness for a given
parameter assignment. The artifact records both `seeds` and `dataSeed`, so a
reader can tell search variation from a deliberate data-split change.

## 9. The sweep reports a stability descriptor, not a ranking

`sweep` runs the same experiment under seeds 17/29/41 and reports, per
algorithm, mean best score, population sigma, min, max, and success rate. The
generated report carries the sentence "stability descriptor over one public
dataset, not a universal algorithm ranking", and the claim string is part of the
artifact.

Rejected alternative: declare a winner. Three seeds on one small dataset cannot
support that, and there is no significance test here.

## 10. A checkpoint binds its trial sequence, not only its configuration

The config digest protects the requested run, but a matching digest alone does not make the file trustworthy: it can still contain a hand-edited trial state, `NaN` duration, duplicated candidate, skipped number, or parameters that no longer belong to the declared space. `Trial.from_dict` now validates state, score/error consistency, finite non-negative duration and finite metrics. `_load_checkpoint` additionally requires trial numbers to be exactly `0..n-1`, parameter keys to match the current specs, each parameter signature to be in the deterministic grid contract, and every signature to be unique.

Rejected alternative: load whatever JSON has the right digest and let later aggregation ignore unfamiliar states. That can turn a corrupt resume into a shorter or differently seeded experiment while preserving a plausible report.

Cost accepted: a checkpoint is now intentionally tied to TuneLab's discrete search-space contract. A future continuous or schema-migrating runner needs an explicit checkpoint version and validator rather than permissive coercion.

Proof: the checkpoint-tampering tests in [`tests/test_runner.py`](../tests/test_runner.py) mutate state, duration, numbers, parameters, and duplicate candidates; all fail before resuming.

## 11. Hash-bound artifacts exclude wall-clock measurements

Trial and run durations are stored as deterministic zeroes, and checkpoint
metadata has no write timestamp. The numerical objective, seeds, search-space
contract and failure records remain in the artifact; machine scheduling and
clock speed do not. This lets the same demo reproduce byte-for-byte in another
output directory.

Rejected alternative: preserve `perf_counter` output in a file labelled as a
reproducible benchmark. That makes a content hash vary on every run while
pretending a microsecond timing is comparable across machines.

## Known weaknesses

An interviewer should get these from the document, not from digging.

1. **`tpe` is not TPE.** See the table in section 1. The name is a familiar
   label for a policy that is much simpler than the algorithm it evokes. If a
   reviewer expects Parzen density estimation and expected improvement, they
   will not find it.
2. **The synthetic and Iris objectives are score-saturated.** Enumerating the
   full space: the 450 Iris grid points yield only **four distinct scores**, and
   **154 tie at the maximum 0.96**, so eight random draws find the maximum with
   roughly 96% probability. The synthetic surface is clamped to 1.0, and at the
   demo's 48 trials all four strategies return exactly 1.0. The Wine default is
   less saturated (16 distinct seed-17 grid scores; 26 maximum-score ties), but
   it is not a substitute for diverse real workloads.
3. **The default Wine sweep still does not establish a winner.** Seeds 17/29/41,
   8 trials, 3 folds:

   | algorithm | mean best | population sigma |
   | --- | ---: | ---: |
| grid | 0.94369115 | 0.00000000 |
| random | 0.98305085 | 0.00000000 |
| tpe | 0.98119900 | 0.00261892 |
| genetic | 0.98305085 | 0.00000000 |

   With a fixed data split, the rows now describe search variation rather than
   mixed search-and-split variation. They still cover one small dataset and one
   split; grid starts with a lexicographic prefix rather than a space-covering
   sample. These rows are evidence about this contract, not an algorithm ranking.
4. **Three seeds, no significance test.** The sweep reports mean and
   *population* sigma (not sample), min, max. No confidence interval, no paired
   test, no multiple-comparison correction. `successRate` counts runs that
   produced any score, not per-trial success.
5. **The default sweep uses only one data split.** The data/model seed is now
   fixed while search seeds vary, so cross-strategy differences no longer mix
   search randomness with split randomness. That also means the default reports
   behavior on one stratified split only; it is not a robustness study over
   resampled data. A caller can choose `--data-seed`, and the artifact records
   it, but no nested or repeated-CV protocol is implemented.
6. **All four strategies search the same finite grid.** Every numeric spec in
   both bundled objectives sets `step`, and `sample_value` snaps float draws
   back onto the step while integer draws come straight from `grid_values()`.
   Verified: 2000 random draws on each objective produced zero off-grid points.
   So "random search" here is discrete-uniform over the grid, and grid under a
   small budget is a lexicographic prefix scan, not a space-covering search.
7. **The three stochastic strategies emit identical first trials.** The RNG is
   reseeded per trial index, and both `tpe` and `genetic` fall back to
   `sample_params` during their four-trial warm-up, so at a given seed random,
   `tpe`, and `genetic` produce byte-identical trials 0 through 3. At the sweep's
   default budget of 8 that leaves each adaptive strategy exactly four adaptive
   proposals.
8. **Budget is unique configurations, not wall-clock or objective cost.** Two
   strategies with the same trial budget can do very different amounts of work
   if the objective's cost varies with the parameters, and nothing here
   normalises for that.
9. **`tpe` and `genetic` share a magic-number policy.** Warm-up of 4, good-set
   quantile 0.25, sigma factor 0.18, parent fraction one third, mutation rate
   0.25, categorical stickiness 0.75. None of these is tuned or ablated; they
   are readable defaults.
10. **No parallelism.** Trials run sequentially in one process (`n_jobs=1` on the
   Iris objective as well). The orchestration is about reproducibility and
   resume, not throughput.
11. **No pruning or early stopping.** Every proposed trial runs to completion,
   so the project does not implement the feature that makes real tuning
   frameworks cheap.
12. **Durability and atomicity gaps.** See section 5.
13. **Adapter contract errors abort rather than resume.** An exception raised by
    a correctly shaped objective is recorded as a failed trial; that supports a
    flaky provider or an invalid candidate. A return value that is not an
    `ObjectiveResult` now raises before the trial is appended or checkpointed.
    This makes a bad adapter visible, but it also means callers must repair it
    before any later candidates can run.
14. **`bool` is not a distinct parameter kind.** It validates specially but is
    dispatched everywhere by `if spec.choices`, identically to categorical, and
    no test or objective ever searches one. `checkpoint_metadata` is dead code
    with no caller.
15. **The 128-attempt duplicate retry is a heuristic.** On a nearly exhausted
    discrete space a strategy can stop early even though unseen configurations
    remain, because 128 random draws happened to miss them.

## Scope

The bundled objectives use a closed-form synthetic surface plus scikit-learn's
public Iris and Wine datasets. There is no client data, private dataset, or
production model in this repository, and no number produced here is a claim
about model quality on any real task.
