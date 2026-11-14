import { Badge } from "../components/Badge";
import { Loading, Notice } from "../components/Notice";
import { useAsync } from "../hooks/useAsync";
import type { Api, RunSummary, TraceEvent } from "../services/api";
import { clock, pretty, seconds } from "../utils/format";

function outcome(run: RunSummary): { text: string; tone: "good" | "bad" | "neutral" } {
  if (run.blocked) return { text: "blocked", tone: "bad" };
  if (run.success === false) return { text: run.stop_reason || "failed", tone: "bad" };
  return { text: "answered", tone: "good" };
}

function Trace({ api, id }: { api: Api; id: string }) {
  const trace = useAsync(() => api.trace(id), [api, id]);
  if (trace.loading) return <Loading what="the trace" />;
  if (trace.error) return <Notice tone="error">{trace.error}</Notice>;
  const events = [...(trace.data ?? [])].sort((a, b) => a.start - b.start);
  if (events.length === 0) return <p className="muted">This trace has no events.</p>;
  const origin = events[0].start;
  const depth = depths(events);
  return (
    <ol className="trace">
      {events.map((event) => (
        <li key={event.span_id} style={{ marginLeft: `${Math.min(depth.get(event.span_id) ?? 0, 6) * 18}px` }}>
          <details>
            <summary>
              <Badge tone={event.status === "error" ? "bad" : "accent"}>{event.kind}</Badge>
              <span className="trace-name">{event.name}</span>
              <span className="muted small">
                +{seconds(event.start - origin)} · {seconds(event.end - event.start)}
              </span>
            </summary>
            <pre>{pretty(event.attributes)}</pre>
          </details>
        </li>
      ))}
    </ol>
  );
}

function depths(events: TraceEvent[]): Map<string, number> {
  const parent = new Map(events.map((event) => [event.span_id, event.parent_id]));
  const depth = new Map<string, number>();
  for (const event of events) {
    let level = 0;
    let cursor = event.parent_id;
    while (cursor && parent.has(cursor) && level < 20) {
      level += 1;
      cursor = parent.get(cursor) ?? null;
    }
    depth.set(event.span_id, level);
  }
  return depth;
}

export function RunsPage({ api, selected }: { api: Api; selected?: string }) {
  const runs = useAsync(() => api.runs(), [api], 5000);
  return (
    <section>
      <header className="page-head">
        <h1>Runs</h1>
        <button type="button" onClick={runs.reload}>
          Refresh
        </button>
      </header>
      <div className="split">
        <div className="card list">
          {runs.loading && <Loading what="runs" />}
          {runs.error && <Notice tone="error">{runs.error}</Notice>}
          {runs.data?.length === 0 && (
            <p className="muted">No runs yet. Runs you start from the chat or through the API with this token appear here.</p>
          )}
          <ul>
            {runs.data?.map((run, index) => {
              const state = outcome(run);
              const active = run.trace_id !== null && run.trace_id === selected;
              return (
                <li key={`${run.trace_id}-${index}`}>
                  <a
                    className={`run-row${active ? " active" : ""}`}
                    href={run.trace_id ? `#/runs/${encodeURIComponent(run.trace_id)}` : "#/runs"}
                    aria-current={active ? "true" : undefined}
                  >
                    <span className="run-task">{run.task}</span>
                    <span className="meta">
                      <Badge tone={state.tone}>{state.text}</Badge>
                      <Badge tone="accent">{run.mode}</Badge>
                      <Badge>{seconds(run.seconds)}</Badge>
                      <span className="muted small">{clock(run.at)}</span>
                    </span>
                  </a>
                </li>
              );
            })}
          </ul>
        </div>
        <div className="card detail">
          {selected ? (
            <>
              <h2>
                Trace <code>{selected}</code>
              </h2>
              <Trace api={api} id={selected} />
            </>
          ) : (
            <p className="muted">Choose a run to see every step it took: routing, model calls, tool calls, checks.</p>
          )}
        </div>
      </div>
    </section>
  );
}
