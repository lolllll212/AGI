// A typed client for the agent API. Every URL is relative to the page, so the console works
// at the server's root, behind a reverse proxy on a sub-path, and through the dev-server proxy.

export type Mode = "reflex" | "react" | "deliberate";

export interface Health {
  status: "ok" | "degraded";
  model: boolean;
  model_circuit: string;
  uptime_s: number;
  in_flight: number;
  tools: number;
  auth_required: boolean;
}

export interface Info {
  version: string;
  model: string | null;
  demo: boolean;
  examples: string[];
  caller: string;
  sandbox: string;
  metrics: boolean;
}

export interface Step {
  thought: string;
  action: string | null;
  args: Record<string, unknown>;
  observation: string;
  ok: boolean;
}

export interface RunResult {
  task: string;
  answer: string | null;
  mode: Mode;
  system: 1 | 2;
  route: { reason: string } | null;
  success: boolean | null;
  grounded: boolean | null;
  blocked: boolean;
  stop_reason: string;
  steps: Step[];
  tool_calls: number;
  tool_errors: number;
  llm_calls: number;
  repairs: number;
  compactions: number;
  escalated: boolean;
  trace_id: string | null;
  seconds: number;
  error: string | null;
  flags: string[];
}

export interface RunSummary {
  trace_id: string | null;
  task: string;
  mode: Mode;
  success: boolean | null;
  blocked: boolean;
  stop_reason: string;
  seconds: number;
  tool_calls: number;
  llm_calls: number;
  session: string | null;
  at: number;
}

export interface TraceEvent {
  trace_id: string;
  span_id: string;
  parent_id: string | null;
  kind: string;
  name: string;
  start: number;
  end: number;
  status: "ok" | "error";
  attributes: Record<string, unknown>;
}

export interface ToolInfo {
  name: string;
  description: string;
  risk: string;
  parameters: Record<string, { type?: string; description?: string; [key: string]: unknown }>;
}

export interface Stats {
  engine: Record<string, number>;
  sessions: number;
  sessions_in_memory: number;
  routing: [string, number, number][];
  routing_model: { outcomes: number; skill: number; trusted: boolean; strongest: [string, number][] } | null;
  memory: Record<string, unknown> | null;
  drift: unknown;
  trace_export: Record<string, unknown>[];
  drift_alarms: unknown[];
  http: {
    tls: boolean;
    requests: Record<string, number>;
    rejected: Record<string, number>;
    active_connections: number;
    refused_connections: number;
  };
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public kind: string,
    message: string,
  ) {
    super(message);
  }
}

const TOKEN_KEY = "agi.token";

export function loadToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveToken(token: string): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* private browsing: keep the token in memory only */
  }
}

export class Api {
  constructor(
    private token: string = "",
    private onUnauthorized: () => void = () => {},
  ) {}

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = {};
    if (this.token) headers["Authorization"] = `Bearer ${this.token}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    let response: Response;
    try {
      response = await fetch(new URL(path, document.baseURI), {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch {
      throw new ApiError(0, "network", "The server cannot be reached.");
    }
    let data: unknown = null;
    try {
      data = await response.json();
    } catch {
      /* not JSON: handled below */
    }
    if (!response.ok) {
      const error = (data as { error?: { type?: string; message?: string } } | null)?.error;
      if (response.status === 401) this.onUnauthorized();
      throw new ApiError(response.status, error?.type ?? "error", error?.message ?? `HTTP ${response.status}`);
    }
    return data as T;
  }

  health = () => this.request<Health>("GET", "health");
  info = () => this.request<Info>("GET", "v1/info");
  tools = () => this.request<{ tools: ToolInfo[] }>("GET", "v1/tools").then((r) => r.tools);
  stats = () => this.request<Stats>("GET", "v1/stats");
  runs = () => this.request<{ runs: RunSummary[] }>("GET", "v1/runs").then((r) => r.runs);
  trace = (id: string) =>
    this.request<{ events: TraceEvent[] }>("GET", `v1/traces/${encodeURIComponent(id)}`).then((r) => r.events);
  run = (task: string, mode: Mode | null, session: string | null) =>
    this.request<RunResult>("POST", "v1/run", { task, mode, session, steps: true });
  endSession = (id: string) => this.request<unknown>("DELETE", `v1/sessions/${encodeURIComponent(id)}`);
}
