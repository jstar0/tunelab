import { FormEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type Objective = {
  id: "surface" | "iris" | "wine";
  name: string;
  description: string;
  default_space: Param[];
  default_folds: number | null;
};
type Param = {
  name: string;
  kind: string;
  low?: number;
  high?: number;
  step?: number;
  choices?: unknown[];
  log_scale?: boolean;
};
type Job = {
  job_id: string;
  experiment_set_id: string;
  status: string;
  revision: number;
  config: Config & { algorithm: string };
  capabilities: { pause: boolean; resume: boolean; cancel: boolean };
  progress: { done: number; total: number; unit: string };
  error?: string | null;
};
type Trial = {
  number: number;
  params: Record<string, unknown>;
  score: number | null;
  metrics: Record<string, number>;
  state: "completed" | "failed";
  error: string | null;
  duration_seconds: number | null;
  timing_status: "recorded" | "unrecorded";
};
type Config = {
  name: string;
  objective: "surface" | "iris" | "wine";
  algorithms: string[];
  budget: number;
  search_seed: number;
  objective_seed: number;
  folds: number;
  space?: Param[];
  tutorial_failure_mode?: boolean;
};
type Comparison = {
  comparable: boolean;
  reasons: string[];
  note: string;
  runs: Array<{
    job_id: string;
    algorithm: string;
    search_seed: number;
    status: string;
    budget: number;
    completed_trials: number;
    failed_trials: number;
    best_score: number | null;
  }>;
  aggregate: Array<{
    algorithm: string;
    count: number;
    mean_best_score: number;
    population_std_best_score: number;
  }>;
};

const algorithmNames: Record<string, string> = {
  grid: "Grid",
  random: "Random",
  tpe: "TPE-like",
  genetic: "Genetic",
};
const algorithmColor: Record<string, string> = {
  grid: "#9aa8b4",
  random: "#47c7c1",
  tpe: "#7aa2f7",
  genetic: "#c78be8",
};
const defaultName = (objective: Config["objective"], budget: number) =>
  `${objective === "wine" ? "Wine" : objective === "iris" ? "Iris" : "Surface"} · ${budget} trials`;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, init);
  const body = await response.json();
  if (!response.ok) throw new Error(body?.detail?.error?.message || "请求失败");
  return body as T;
}

function LineChart({ trials, color }: { trials: Trial[]; color: string }) {
  const points = trials.filter((trial) => trial.score !== null);
  if (!points.length)
    return <div className="empty-chart">等待第一个完成的 trial…</div>;
  let best = -Infinity;
  const values = points.map((trial) => {
    best = Math.max(best, Number(trial.score));
    return { x: trial.number, y: best };
  });
  const low = Math.min(...values.map((item) => item.y));
  const high = Math.max(...values.map((item) => item.y));
  const span = Math.max(high - low, 0.002);
  const width = 680;
  const height = 260;
  const pad = 34;
  const maxX = Math.max(values.at(-1)?.x || 1, 1);
  const path = values
    .map((item, index) => {
      const x = pad + (item.x / maxX) * (width - pad * 2);
      const y = height - pad - ((item.y - low) / span) * (height - pad * 2);
      return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      className="chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="截至每个 trial 的最佳验证分数"
    >
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} />
      <path d={path} stroke={color} fill="none" strokeWidth="3" />
      <text x={pad} y="18">
        best CV score
      </text>
      <text x={width - 78} y={height - 8}>
        trial
      </text>
    </svg>
  );
}

function SpaceEditor({
  params,
  onChange,
}: {
  params: Param[];
  onChange: (
    name: string,
    field: keyof Param,
    value: string | number | unknown[],
  ) => void;
}) {
  return (
    <div className="space editor">
      {params.map((param) => (
        <fieldset key={param.name}>
          <legend>{param.name}</legend>
          <label>
            类型
            <select
              value={param.kind}
              onChange={(event) =>
                onChange(param.name, "kind", event.target.value)
              }
            >
              <option value="int">int</option>
              <option value="float">float</option>
              <option value="categorical">categorical</option>
              <option value="bool">bool</option>
            </select>
          </label>
          {param.kind === "categorical" || param.kind === "bool" ? (
            <label>
              choices
              <input
                value={(param.choices || []).join(", ")}
                onChange={(event) =>
                  onChange(
                    param.name,
                    "choices",
                    event.target.value
                      .split(",")
                      .map((value) => value.trim())
                      .filter(Boolean)
                      .map((value) =>
                        param.kind === "bool"
                          ? value.toLowerCase() === "true"
                          : value,
                      ),
                  )
                }
              />
            </label>
          ) : (
            <div className="numeric-fields">
              <label>
                low
                <input
                  type="number"
                  step={param.kind === "float" ? "any" : 1}
                  value={param.low ?? ""}
                  onChange={(event) =>
                    onChange(param.name, "low", Number(event.target.value))
                  }
                />
              </label>
              <label>
                high
                <input
                  type="number"
                  step={param.kind === "float" ? "any" : 1}
                  value={param.high ?? ""}
                  onChange={(event) =>
                    onChange(param.name, "high", Number(event.target.value))
                  }
                />
              </label>
              <label>
                step
                <input
                  type="number"
                  min="0"
                  step={param.kind === "float" ? "any" : 1}
                  value={param.step ?? ""}
                  onChange={(event) =>
                    onChange(param.name, "step", Number(event.target.value))
                  }
                />
              </label>
            </div>
          )}
        </fieldset>
      ))}
    </div>
  );
}

function App() {
  const [objectives, setObjectives] = useState<Objective[]>([]);
  const [runs, setRuns] = useState<Job[]>([]);
  const [active, setActive] = useState<Job>();
  const [trials, setTrials] = useState<Trial[]>([]);
  const [page, setPage] = useState<"experiments" | "new" | "compare">(
    "experiments",
  );
  const [message, setMessage] = useState(
    "选择一个实验，或新建一个真实的公开数据实验。",
  );
  const [config, setConfig] = useState<Config>({
    name: "Wine · 48 trials",
    objective: "wine",
    algorithms: ["random", "tpe"],
    budget: 48,
    search_seed: 7,
    objective_seed: 17,
    folds: 5,
  });
  const [selectedTrial, setSelectedTrial] = useState<Trial>();
  const [trialFilter, setTrialFilter] = useState<"all" | "failed">("all");
  const [comparison, setComparison] = useState<Comparison>();
  const [comparisonIds, setComparisonIds] = useState<string[]>([]);

  const refreshRuns = async () => {
    const response = await api<{ runs: Job[] }>("/runs");
    setRuns(response.runs);
    if (active)
      setActive(response.runs.find((run) => run.job_id === active.job_id));
  };
  const openRun = async (run: Job) => {
    setActive(run);
    setPage("experiments");
    const response = await api<{ rows: Trial[] }>(
      `/runs/${run.job_id}/trials?limit=200`,
    );
    setTrials(response.rows);
    setSelectedTrial(undefined);
    setMessage(
      `正在查看 ${algorithmNames[run.config.algorithm] || run.config.algorithm} · ${run.config.objective}。`,
    );
  };
  useEffect(() => {
    api<{ objectives: Objective[] }>("/objectives").then((value) =>
      setObjectives(value.objectives),
    );
    refreshRuns().catch((error) => setMessage(error.message));
  }, []);
  useEffect(() => {
    if (
      !active ||
      ["succeeded", "failed", "cancelled", "paused", "interrupted"].includes(
        active.status,
      )
    )
      return;
    const refresh = () => {
      refreshRuns().catch(() => undefined);
      api<{ rows: Trial[] }>(`/runs/${active.job_id}/trials?limit=200`)
        .then((value) => setTrials(value.rows))
        .catch(() => undefined);
    };
    const events = new EventSource(`/api/v1/jobs/${active.job_id}/events`);
    [
      "queued",
      "running",
      "trial",
      "pausing",
      "paused",
      "cancelling",
      "cancelled",
      "succeeded",
      "failed",
    ].forEach((kind) => events.addEventListener(kind, refresh));
    refresh();
    return () => events.close();
  }, [active?.job_id, active?.status]);

  const currentObjective = objectives.find(
    (item) => item.id === config.objective,
  );
  const editableSpace = config.space || currentObjective?.default_space || [];
  const best = useMemo(
    () =>
      trials
        .filter((trial) => trial.score !== null)
        .reduce<
          Trial | undefined
        >((winner, trial) => (!winner || Number(trial.score) > Number(winner.score) ? trial : winner), undefined),
    [trials],
  );
  const comparableCandidates = runs.filter(
    (run) =>
      active &&
      run.config.objective === active.config.objective &&
      run.config.objective_seed === active.config.objective_seed &&
      run.config.folds === active.config.folds &&
      run.config.budget === active.config.budget &&
      JSON.stringify(run.config.space || []) ===
        JSON.stringify(active.config.space || []),
  );
  useEffect(() => {
    if (page === "compare" && active && comparisonIds.length < 2) {
      setComparisonIds(
        comparableCandidates.slice(0, 4).map((run) => run.job_id),
      );
    }
  }, [page, active?.job_id, runs.length]);
  useEffect(() => {
    if (page !== "compare" || comparisonIds.length < 2) {
      setComparison(undefined);
      return;
    }
    api<Comparison>("/comparisons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_ids: comparisonIds }),
    })
      .then(setComparison)
      .catch((error: Error) => setMessage(error.message));
  }, [page, comparisonIds]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    try {
      const result = await api<{ jobs: Job[] }>("/experiment-sets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      await refreshRuns();
      await openRun(result.jobs[0]);
      setMessage(
        `已创建 ${result.jobs.length} 个真实实验；候选序列和每条观察都会持久化。`,
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法创建实验");
    }
  };
  const control = async (action: "pause" | "resume" | "cancel") => {
    if (!active) return;
    try {
      const next = await api<Job>(`/jobs/${active.job_id}/${action}`, {
        method: "POST",
      });
      setActive(next);
      await refreshRuns();
      setMessage(
        action === "pause"
          ? "会在当前 trial 结束后暂停。"
          : action === "cancel"
            ? "会在当前 trial 结束后取消。"
            : "已排队继续实验。",
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "状态未改变");
    }
  };
  const exportArtifact = async (
    kind: "canonical" | "observations" | "trials" | "report" | "config",
  ) => {
    if (!active) return;
    try {
      const artifact = await api<{ download_url: string }>(
        `/runs/${active.job_id}/exports`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ kind }),
        },
      );
      const link = document.createElement("a");
      link.href = artifact.download_url;
      link.download = "";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setMessage(
        `${kind === "observations" ? "观测" : kind === "trials" ? "trial 明细" : "canonical"}导出已生成。`,
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "导出失败");
    }
  };
  const changeObjective = (objective: Config["objective"]) => {
    const item = objectives.find((entry) => entry.id === objective);
    setConfig((current) => ({
      ...current,
      objective,
      name: defaultName(objective, current.budget),
      folds: item?.default_folds || 3,
      space: item?.default_space,
      tutorial_failure_mode: false,
    }));
  };
  const toggleAlgorithm = (algorithm: string) =>
    setConfig((current) => ({
      ...current,
      algorithms: current.algorithms.includes(algorithm)
        ? current.algorithms.filter((value) => value !== algorithm)
        : [...current.algorithms, algorithm],
    }));
  const updateSpace = (
    name: string,
    field: keyof Param,
    value: string | number | unknown[],
  ) => {
    setConfig((current) => ({
      ...current,
      space: editableSpace.map((param) =>
        param.name === name ? { ...param, [field]: value } : param,
      ),
    }));
  };
  const observedSeconds = trials.reduce(
    (total, trial) => total + (trial.duration_seconds || 0),
    0,
  );
  const visibleTrials =
    trialFilter === "failed"
      ? trials.filter((trial) => trial.state === "failed")
      : trials;
  useEffect(() => {
    if (!selectedTrial) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelectedTrial(undefined);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [selectedTrial]);
  const toggleComparison = (jobId: string) =>
    setComparisonIds((current) =>
      current.includes(jobId)
        ? current.filter((item) => item !== jobId)
        : current.length < 4
          ? [...current, jobId]
          : current,
    );

  return (
    <main>
      <aside>
        <p className="brand">TUNELAB / LOCAL</p>
        <h1>调参实验</h1>
        <p>让搜索策略、数据协议与每次 trial 的证据在同一个工作台里。</p>
        <nav>
          <button
            className={page === "experiments" ? "active" : ""}
            onClick={() => setPage("experiments")}
          >
            01　实验
          </button>
          <button
            className={page === "new" ? "active" : ""}
            onClick={() => setPage("new")}
          >
            02　新建
          </button>
          <button
            className={page === "compare" ? "active" : ""}
            onClick={() => setPage("compare")}
          >
            03　比较
          </button>
        </nav>
        <small>本地执行 · 公开内置数据 · 不上传代码</small>
      </aside>
      <section className="shell">
        <header>
          <span>
            实验 /{" "}
            {page === "new" ? "新建" : page === "compare" ? "比较" : "运行"}
          </span>
          <span>{message}</span>
        </header>
        {page === "new" && (
          <form className="new-layout" onSubmit={submit}>
            <section className="card">
              <p className="eyebrow">OBJECTIVE</p>
              <h2>选择实验协议</h2>
              <div className="objective-list">
                {objectives.map((item) => (
                  <button
                    type="button"
                    className={config.objective === item.id ? "chosen" : ""}
                    key={item.id}
                    onClick={() => changeObjective(item.id)}
                  >
                    <b>
                      {item.id === "wine"
                        ? "Wine RandomForest"
                        : item.id === "iris"
                          ? "Iris RandomForest"
                          : "Surface tutorial"}
                    </b>
                    <small>{item.description}</small>
                  </button>
                ))}
              </div>
              <p className="hint">
                CV 分数不是独立测试集性能。Surface
                只用于机制演示，不能用于算法排名。
              </p>
              {config.objective === "surface" && (
                <label className="tutorial-toggle">
                  <input
                    type="checkbox"
                    checked={Boolean(config.tutorial_failure_mode)}
                    onChange={(event) =>
                      setConfig({
                        ...config,
                        tutorial_failure_mode: event.target.checked,
                      })
                    }
                  />
                  <span>
                    产生可复现的 failed trial 教程（GELU 候选会明确报错）
                  </span>
                </label>
              )}
            </section>
            <section className="card">
              <p className="eyebrow">SEARCH SPACE</p>
              <h2>搜索空间编辑</h2>
              <SpaceEditor params={editableSpace} onChange={updateSpace} />
              <p className="hint">
                完整空间会随 run
                冻结。想改参数或预算，请复制为新实验，不会悄悄续跑旧
                checkpoint。
              </p>
            </section>
            <section className="card budget">
              <p className="eyebrow">BUDGET</p>
              <h2>策略与预算</h2>
              <label>
                实验名称
                <input
                  value={config.name}
                  onChange={(event) =>
                    setConfig({ ...config, name: event.target.value })
                  }
                />
              </label>
              <label>
                trial 预算{" "}
                <input
                  type="number"
                  min="1"
                  max="500"
                  value={config.budget}
                  onChange={(event) => {
                    const budget = Number(event.target.value);
                    setConfig((current) => ({
                      ...current,
                      budget,
                      name:
                        current.name ===
                        defaultName(current.objective, current.budget)
                          ? defaultName(current.objective, budget)
                          : current.name,
                    }));
                  }}
                />
              </label>
              <label>
                搜索 seed{" "}
                <input
                  type="number"
                  value={config.search_seed}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      search_seed: Number(event.target.value),
                    })
                  }
                />
              </label>
              {config.objective !== "surface" && (
                <label>
                  CV folds{" "}
                  <input
                    type="number"
                    min="2"
                    max="10"
                    value={config.folds}
                    onChange={(event) =>
                      setConfig({
                        ...config,
                        folds: Number(event.target.value),
                      })
                    }
                  />
                </label>
              )}
              <div className="algorithms">
                {Object.keys(algorithmNames).map((algorithm) => (
                  <label key={algorithm}>
                    <input
                      type="checkbox"
                      checked={config.algorithms.includes(algorithm)}
                      onChange={() => toggleAlgorithm(algorithm)}
                    />
                    <span>{algorithmNames[algorithm]}</span>
                  </label>
                ))}
              </div>
              <button className="primary" disabled={!config.algorithms.length}>
                创建 {config.algorithms.length || ""} 个实验 →
              </button>
            </section>
          </form>
        )}
        {page === "experiments" && (
          <div className="workbench">
            <section className="run-list">
              <div className="list-heading">
                <h2>实验</h2>
                <button onClick={() => setPage("new")}>新建</button>
              </div>
              {runs.length ? (
                runs.map((run) => (
                  <button
                    key={run.job_id}
                    className={run.job_id === active?.job_id ? "selected" : ""}
                    onClick={() => openRun(run)}
                  >
                    <span className={`status ${run.status}`}>{run.status}</span>
                    <b>
                      {algorithmNames[run.config.algorithm] ||
                        run.config.algorithm}
                    </b>
                    <small>
                      {run.config.objective} · {run.progress.done}/
                      {run.progress.total} trials
                    </small>
                  </button>
                ))
              ) : (
                <p className="empty">
                  还没有实验。创建 Wine 对比，或用 Surface 看暂停/恢复机制。
                </p>
              )}
            </section>
            <section className="run-main">
              {active ? (
                <>
                  <div className="run-heading">
                    <div>
                      <p className="eyebrow">
                        {active.config.objective.toUpperCase()} /{" "}
                        {active.config.algorithm.toUpperCase()}
                      </p>
                      <h2>{active.config.name}</h2>
                      <p>
                        {active.progress.done} / {active.progress.total} 个
                        trial · 状态{" "}
                        <span className={`status ${active.status}`}>
                          {active.status}
                        </span>
                      </p>
                    </div>
                    <div className="controls">
                      {active.capabilities.pause && (
                        <button onClick={() => control("pause")}>
                          当前 trial 后暂停
                        </button>
                      )}
                      {active.capabilities.resume && (
                        <button onClick={() => control("resume")}>继续</button>
                      )}
                      {active.capabilities.cancel && (
                        <button
                          className="danger"
                          onClick={() => control("cancel")}
                        >
                          当前 trial 后取消
                        </button>
                      )}
                      {active.status === "succeeded" && (
                        <>
                          <button onClick={() => exportArtifact("report")}>
                            HTML report
                          </button>
                          <button onClick={() => exportArtifact("trials")}>
                            CSV trial 明细
                          </button>
                          <button onClick={() => exportArtifact("config")}>
                            配置 JSON
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="metrics">
                    <div>
                      <span>最佳 CV</span>
                      <b>{best?.score?.toFixed(4) || "—"}</b>
                    </div>
                    <div>
                      <span>完成</span>
                      <b>
                        {
                          trials.filter((trial) => trial.state === "completed")
                            .length
                        }
                      </b>
                    </div>
                    <div>
                      <span>失败</span>
                      <b>
                        {
                          trials.filter((trial) => trial.state === "failed")
                            .length
                        }
                      </b>
                    </div>
                    <div>
                      <span>已记录计算耗时</span>
                      <b>
                        {observedSeconds
                          ? `${observedSeconds.toFixed(1)}s`
                          : "未记录"}
                      </b>
                    </div>
                    <div>
                      <span>协议</span>
                      <b>
                        {active.config.objective === "surface"
                          ? "tutorial"
                          : `${active.config.folds}-fold CV`}
                      </b>
                    </div>
                  </div>
                  <section className="chart-card">
                    <div className="section-title">
                      <h3>收敛</h3>
                      <small>
                        y = 截至当前 trial 的最佳验证分数；失败不填作 0。
                      </small>
                    </div>
                    <LineChart
                      trials={trials}
                      color={
                        algorithmColor[active.config.algorithm] || "#47c7c1"
                      }
                    />
                  </section>
                  <section className="trials">
                    <div className="section-title">
                      <h3>Trial 明细</h3>
                      <div className="trial-tools">
                        <label>
                          显示
                          <select
                            value={trialFilter}
                            onChange={(event) =>
                              setTrialFilter(
                                event.target.value as "all" | "failed",
                              )
                            }
                          >
                            <option value="all">全部 ({trials.length})</option>
                            <option value="failed">
                              仅失败 ({trials.filter((trial) => trial.state === "failed").length})
                            </option>
                          </select>
                        </label>
                        <small>点行或按 Enter 查看参数、CV 指标和错误。</small>
                      </div>
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>状态</th>
                            <th>score</th>
                            <th>CV</th>
                            <th>耗时</th>
                            <th>参数</th>
                          </tr>
                        </thead>
                        <tbody>
                          {visibleTrials.map((trial) => (
                            <tr
                              key={trial.number}
                              tabIndex={0}
                              onClick={() => setSelectedTrial(trial)}
                              onKeyDown={(event) =>
                                event.key === "Enter" && setSelectedTrial(trial)
                              }
                            >
                              <td>{trial.number}</td>
                              <td>
                                <span className={`status ${trial.state}`}>
                                  {trial.state}
                                </span>
                              </td>
                              <td>{trial.score?.toFixed(4) || "—"}</td>
                              <td>
                                {trial.metrics.cv_std !== undefined
                                  ? `${trial.metrics.cv_mean?.toFixed(3)} ± ${trial.metrics.cv_std.toFixed(3)}`
                                  : "—"}
                              </td>
                              <td>
                                {trial.duration_seconds === null
                                  ? "未记录"
                                  : `${trial.duration_seconds.toFixed(2)}s`}
                              </td>
                              <td>
                                <code>{JSON.stringify(trial.params)}</code>
                              </td>
                            </tr>
                          ))}
                          {!visibleTrials.length && (
                            <tr>
                              <td colSpan={6} className="empty-row">
                                没有失败的 trial；此运行没有可筛出的错误记录。
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </>
              ) : (
                <div className="blank">
                  <h2>从真实实验开始</h2>
                  <p>
                    新建一组 Wine 或 Iris 运行；每个策略都有自己的 immutable
                    checkpoint 和观测文件。
                  </p>
                  <button className="primary" onClick={() => setPage("new")}>
                    新建实验 →
                  </button>
                </div>
              )}
            </section>
            {selectedTrial && (
              <aside className="drawer">
                <button
                  className="close"
                  onClick={() => setSelectedTrial(undefined)}
                  aria-label="关闭 trial 详情"
                >
                  ×
                </button>
                <p className="eyebrow">TRIAL {selectedTrial.number}</p>
                <h3>
                  {selectedTrial.state === "failed"
                    ? "这次 trial 未产生分数"
                    : "参数与验证指标"}
                </h3>
                <pre>{JSON.stringify(selectedTrial.params, null, 2)}</pre>
                {selectedTrial.score !== null && (
                  <p>
                    score <b>{selectedTrial.score.toFixed(6)}</b>
                  </p>
                )}
                <p>
                  计算耗时{" "}
                  <b>
                    {selectedTrial.duration_seconds === null
                      ? "未记录"
                      : `${selectedTrial.duration_seconds.toFixed(3)}s`}
                  </b>
                </p>
                <pre>{JSON.stringify(selectedTrial.metrics, null, 2)}</pre>
                {selectedTrial.error && (
                  <p className="error">{selectedTrial.error}</p>
                )}
                <button
                  onClick={() =>
                    navigator.clipboard?.writeText(
                      JSON.stringify(selectedTrial.params, null, 2),
                    )
                  }
                >
                  复制参数 JSON
                </button>
              </aside>
            )}
          </div>
        )}
        {page === "compare" && (
          <section className="compare">
            <p className="eyebrow">COMPARISON</p>
            <h2>比较实验</h2>
            <p className="hint">
              最多选四条 run。系统会先检查它们是否满足同预算比较条件。
            </p>
            <div className="compare-picker">
              {runs.map((run) => (
                <label key={run.job_id}>
                  <input
                    type="checkbox"
                    checked={comparisonIds.includes(run.job_id)}
                    onChange={() => toggleComparison(run.job_id)}
                  />
                  <span>
                    {algorithmNames[run.config.algorithm]} ·{" "}
                    {run.config.objective} · {run.progress.done}/
                    {run.progress.total}
                  </span>
                </label>
              ))}
            </div>
            {comparison ? (
              <>
                <p
                  className={
                    comparison.comparable ? "notice-good" : "notice-warn"
                  }
                >
                  {comparison.comparable
                    ? "目标、数据种子、CV、空间与预算一致：可以做同预算比较。"
                    : `以下条件不同：${comparison.reasons.join("、")}。仅并列展示，不标胜者。`}
                </p>
                <div className="comparison-table">
                  <div className="comparison-head">
                    <span>策略</span>
                    <span>搜索 seed</span>
                    <span>状态</span>
                    <span>完成</span>
                    <span>最佳分数</span>
                  </div>
                  {comparison.runs.map((run) => (
                    <button
                      key={run.job_id}
                      onClick={() => {
                        const selected = runs.find(
                          (item) => item.job_id === run.job_id,
                        );
                        if (selected) openRun(selected);
                      }}
                    >
                      <span style={{ color: algorithmColor[run.algorithm] }}>
                        {algorithmNames[run.algorithm]}
                      </span>
                      <span>{run.search_seed}</span>
                      <span className={`status ${run.status}`}>
                        {run.status}
                      </span>
                      <span>
                        {run.completed_trials}/{run.budget}
                      </span>
                      <span>
                        {run.best_score === null
                          ? "未完成"
                          : run.best_score.toFixed(4)}
                      </span>
                    </button>
                  ))}
                </div>
                {comparison.aggregate.length > 0 && (
                  <section className="aggregate" aria-label="多 seed 汇总">
                    <h3>多 seed 汇总</h3>
                    <p>
                      每种策略仅汇总当前勾选、已完成 run 的最佳 CV；population σ
                      是描述统计，不是置信区间。
                    </p>
                    <div className="aggregate-grid">
                      {comparison.aggregate.map((item) => (
                        <div key={item.algorithm}>
                          <span style={{ color: algorithmColor[item.algorithm] }}>
                            {algorithmNames[item.algorithm] || item.algorithm}
                          </span>
                          <b>n = {item.count}</b>
                          <small>mean best CV {item.mean_best_score.toFixed(4)}</small>
                          <small>population σ {item.population_std_best_score.toFixed(4)}</small>
                        </div>
                      ))}
                    </div>
                  </section>
                )}
                <p className="hint">
                  当前版本不把少量单 seed 运行叫做“胜者”。多 seed 汇总显示 mean
                  与 population σ，但不是置信区间。
                </p>
              </>
            ) : (
              <div className="blank">
                <p>选择至少两条 run 后，系统会显示可比性原因和实际汇总。</p>
              </div>
            )}
          </section>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
