// Replay layer: answers the workbench's /api/v1 calls from responses that were
// recorded from the real service (scripts/record_replay.py). Nothing is invented
// here: anything that was not recorded is refused with an explanation.

type Job = {
  job_id: string;
  experiment_set_id: string;
  status: string;
  revision: number;
  config: Record<string, any>;
  capabilities: { pause: boolean; resume: boolean; cancel: boolean };
  progress: { done: number; total: number; unit: string };
  error: string | null;
};
type RecordedEvent = { id: number; event: string; data: unknown };
type RecordedRun = {
  job: Job;
  trials: { job_id: string; rows: unknown[]; total: number };
  events: RecordedEvent[];
};
type Recording = {
  objectives: { objectives: { id: string; default_space: unknown[] }[] };
  runs: RecordedRun[];
};
type Comparisons = {
  rows: Record<string, unknown>;
  selections: Record<string, Record<string, unknown>>;
};

const NO_CONTROLS = { pause: false, resume: false, cancel: false };
const PACE_MS = 70;

let recording: Promise<Recording> | undefined;
let comparisons: Promise<Comparisons> | undefined;
let exportsData: Promise<Record<string, Record<string, string>>> | undefined;
const loadRecording = () => (recording ??= import("./recording.json").then((m) => m.default as unknown as Recording));
const loadComparisons = () => (comparisons ??= import("./comparisons.json").then((m) => m.default as unknown as Comparisons));
const loadExports = () => (exportsData ??= import("./exports.json").then((m) => m.default as unknown as Record<string, Record<string, string>>));

type Live = { status: string; done: number; revision: number };
const live = new Map<string, Live>();
const listeners = new Map<string, Set<(kind: string, data: unknown, id: number) => void>>();
const queue: string[] = [];
let pumping = false;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function failure(status: number, code: string, message: string) {
  return new Response(JSON.stringify({ detail: { error: { code, message, retryable: false } } }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
const ok = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

function view(run: RecordedRun): Job {
  const state = live.get(run.job.job_id);
  if (!state) return { ...run.job, capabilities: NO_CONTROLS };
  return {
    ...run.job,
    status: state.status,
    revision: run.job.revision + state.revision,
    capabilities: NO_CONTROLS,
    progress: { ...run.job.progress, done: state.done },
  };
}

function emit(jobId: string, kind: string, data: unknown, id: number) {
  listeners.get(jobId)?.forEach((listener) => listener(kind, data, id));
}

async function pump() {
  if (pumping) return;
  pumping = true;
  const data = await loadRecording();
  while (queue.length) {
    const jobId = queue.shift()!;
    const run = data.runs.find((item) => item.job.job_id === jobId);
    const state = live.get(jobId);
    if (!run || !state) continue;
    for (const event of run.events) {
      if (event.event === "queued") continue;
      await sleep(PACE_MS);
      state.revision += 1;
      if (event.event === "running") state.status = "running";
      if (event.event === "trial") state.done = (event.data as { payload: { done: number } }).payload.done;
      if (event.event === "succeeded") {
        state.status = "succeeded";
        state.done = run.job.progress.total;
      }
      emit(jobId, event.event, event.data, event.id);
    }
  }
  pumping = false;
}

function describe(runs: RecordedRun[]) {
  const seen = new Map<string, Set<number>>();
  for (const { job } of runs) {
    const label = `${job.config.objective}${job.config.tutorial_failure_mode ? "（含失败教程）" : ""}`;
    const seeds = seen.get(label) ?? new Set<number>();
    seeds.add(job.config.search_seed);
    seen.set(label, seeds);
  }
  return [...seen].map(([label, seeds]) => `${label} · 48 trials · search seed ${[...seeds].join("/")}`).join("；");
}

async function createSet(body: Record<string, any>) {
  const data = await loadRecording();
  const defaultSpace = data.objectives.objectives.find((item) => item.id === body.objective)?.default_space;
  const space = body.space ?? [];
  const isDefaultSpace = space.length === 0 || JSON.stringify(space) === JSON.stringify(defaultSpace);
  const matches = (body.algorithms as string[]).map((algorithm) =>
    isDefaultSpace
      ? data.runs.find(({ job: { config: c } }) =>
          c.algorithm === algorithm &&
          c.objective === body.objective &&
          c.budget === body.budget &&
          c.search_seed === body.search_seed &&
          c.objective_seed === body.objective_seed &&
          (body.objective === "surface" || c.folds === body.folds) &&
          Boolean(c.tutorial_failure_mode) === Boolean(body.tutorial_failure_mode),
        )
      : undefined,
  );
  if (!body.algorithms?.length || matches.some((match) => !match)) {
    return failure(
      422,
      "REPLAY_NOT_RECORDED",
      `回放版不能现场运行搜索：你选的这组配置没有录制。可复现的已录制配置：${describe(data.runs)}（四种策略均已录制，搜索空间使用默认值）。`,
    );
  }
  const found = matches as RecordedRun[];
  const setId = found[0].job.experiment_set_id;
  const started: Job[] = [];
  for (const run of found) {
    const id = run.job.job_id;
    const current = live.get(id);
    if (current && (current.status === "queued" || current.status === "running")) {
      started.push(view(run));
      continue;
    }
    live.set(id, { status: "queued", done: 0, revision: (current?.revision ?? 0) + 1 });
    queue.push(id);
    started.push(view(run));
  }
  void pump();
  return ok({ experiment_set_id: setId, jobs: started }, 202);
}

async function exportArtifact(jobId: string, kind: string) {
  const runs = (await loadRecording()).runs;
  const run = runs.find((item) => item.job.job_id === jobId);
  if (!run) return failure(404, "JOB_NOT_FOUND", "回放数据里没有这条实验。");
  const state = live.get(jobId);
  if (state && state.status !== "succeeded") return failure(409, "RESULT_NOT_READY", "回放尚未结束，结束后再导出。");
  const types: Record<string, string> = {
    report: "text/html;charset=utf-8",
    trials: "text/csv;charset=utf-8",
    config: "application/json",
  };
  const names: Record<string, string> = { report: "report.html", trials: "trials.csv", config: "config.json" };
  if (!types[kind]) return failure(409, "REPLAY_UNAVAILABLE", "回放版只提供 HTML report、CSV trial 明细和配置 JSON 三种已录制的导出。");
  const text = (await loadExports())[jobId]?.[kind];
  if (text === undefined) return failure(409, "REPLAY_UNAVAILABLE", "这份导出没有录制。");
  const url = URL.createObjectURL(new Blob([text], { type: types[kind] }));
  return ok({ artifact_id: `replay_${jobId}_${kind}`, download_url: url, filename: `tunelab-${jobId.slice(-8)}-${names[kind]}` }, 201);
}

async function route(method: string, url: URL, body: any): Promise<Response> {
  const path = url.pathname.replace(/^.*\/api\/v1/, "");
  const data = await loadRecording();
  const find = (id: string) => data.runs.find((item) => item.job.job_id === id);
  let match: RegExpMatchArray | null;
  if (method === "GET" && path === "/health") return ok({ status: "ok", mode: "replay" });
  if (method === "GET" && path === "/objectives") return ok(data.objectives);
  if (method === "GET" && path === "/runs") return ok({ runs: data.runs.map(view) });
  if (method === "GET" && (match = path.match(/^\/jobs\/([^/]+)$/))) {
    const run = find(match[1]);
    return run ? ok(view(run)) : failure(404, "JOB_NOT_FOUND", "回放数据里没有这条实验。");
  }
  if (method === "GET" && (match = path.match(/^\/runs\/([^/]+)\/trials$/))) {
    const run = find(match[1]);
    if (!run) return failure(404, "JOB_NOT_FOUND", "回放数据里没有这条实验。");
    const visible = live.get(match[1])?.done ?? run.trials.rows.length;
    const rows = run.trials.rows.slice(0, visible);
    const offset = Number(url.searchParams.get("offset") ?? 0);
    const limit = Number(url.searchParams.get("limit") ?? 100);
    return ok({ job_id: match[1], rows: rows.slice(offset, offset + limit), total: rows.length });
  }
  if (method === "POST" && path === "/experiment-sets") return createSet(body ?? {});
  if (method === "POST" && path === "/comparisons") {
    const ids: string[] = body?.run_ids ?? [];
    const pack = await loadComparisons();
    const summary = pack.selections[[...ids].sort().join(",")];
    if (!summary || new Set(ids).size !== ids.length) {
      return failure(
        404,
        "REPLAY_NOT_RECORDED",
        "回放版没有录制这个组合的比较结果。同一目标下任意 2–4 条 run，或任意两条 run 的比较均已录制。",
      );
    }
    return ok({ ...summary, runs: ids.map((id) => pack.rows[id]) });
  }
  if (method === "POST" && (match = path.match(/^\/jobs\/([^/]+)\/(pause|resume|cancel)$/))) {
    return failure(409, "REPLAY_UNAVAILABLE", "回放版没有真实运行的任务，不能暂停、继续或取消。");
  }
  if (method === "POST" && (match = path.match(/^\/runs\/([^/]+)\/exports$/))) return exportArtifact(match[1], body?.kind ?? "canonical");
  return failure(501, "REPLAY_UNAVAILABLE", "这项操作在回放版里不可用：它需要真实运行的后端服务。");
}

class ReplayEventSource extends EventTarget {
  readonly url: string;
  readyState = 1;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onopen: ((event: Event) => void) | null = null;
  private off: (() => void) | undefined;

  constructor(url: string) {
    super();
    this.url = url;
    const jobId = url.match(/\/jobs\/([^/]+)\/events/)?.[1];
    if (!jobId) return;
    const listener = (kind: string, data: unknown, id: number) => {
      const message = new MessageEvent(kind, { data: JSON.stringify(data), lastEventId: String(id) });
      this.dispatchEvent(message);
    };
    const set = listeners.get(jobId) ?? new Set();
    set.add(listener);
    listeners.set(jobId, set);
    this.off = () => set.delete(listener);
  }
  close() {
    this.readyState = 2;
    this.off?.();
  }
}

export function install() {
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const url = new URL(raw, window.location.href);
    if (!url.pathname.includes("/api/v1/")) return realFetch(input, init);
    const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
    let body: unknown;
    try {
      body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    } catch {
      body = undefined;
    }
    return route(method, url, body);
  };
  (window as unknown as { EventSource: unknown }).EventSource = ReplayEventSource;
}
