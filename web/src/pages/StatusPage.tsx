import { Loading, Notice } from "../components/Notice";
import { useAsync } from "../hooks/useAsync";
import type { Api, Health, Info } from "../services/api";
import { label, percent, pretty, uptime } from "../utils/format";

function Figure({ name, value }: { name: string; value: string | number }) {
  return (
    <div className="figure">
      <span className="figure-value">{value}</span>
      <span className="figure-name">{name}</span>
    </div>
  );
}

export function StatusPage({ api, info, health }: { api: Api; info: Info | null; health: Health | null }) {
  const stats = useAsync(() => api.stats(), [api], 5000);
  const data = stats.data;
  const runs = data?.engine.runs ?? 0;
  return (
    <section>
      <header className="page-head">
        <h1>Status</h1>
        <span className="muted small">Refreshes every 5 seconds</span>
      </header>
      {stats.loading && <Loading what="status" />}
      {stats.error && <Notice tone="error">{stats.error}</Notice>}

      <div className="card">
        <h2>Service</h2>
        <div className="figures">
          <Figure name="Version" value={info?.version ?? "-"} />
          <Figure name="Model" value={info?.model ?? (health?.model ? "configured" : "none")} />
          <Figure name="Model circuit" value={health?.model_circuit ?? "-"} />
          <Figure name="Sandbox" value={info?.sandbox ?? "-"} />
          <Figure name="Uptime" value={health ? uptime(health.uptime_s) : "-"} />
          <Figure name="Signed in as" value={info?.caller ?? "-"} />
        </div>
      </div>

      {data && (
        <>
          <div className="card">
            <h2>Engine</h2>
            <div className="figures">
              <Figure name="Runs" value={runs} />
              <Figure name="Answered" value={runs ? percent((data.engine.successes ?? 0) / runs) : "-"} />
              {Object.entries(data.engine)
                .filter(([name]) => name !== "runs" && name !== "successes")
                .map(([name, value]) => (
                  <Figure key={name} name={label(name)} value={value} />
                ))}
              <Figure name="Conversations" value={data.sessions} />
            </div>
          </div>

          <div className="card">
            <h2>Learned routing</h2>
            {data.routing_model && data.routing_model.outcomes > 0 && (
              <p className="muted small">
                Across kinds of task, a model has learned from {data.routing_model.outcomes} outcomes.{" "}
                {data.routing_model.trusted
                  ? `Its forecasts have removed ${percent(data.routing_model.skill)} of the error of guessing the average, so the router consults it.`
                  : "Its forecasts have not yet beaten guessing the average, so the router ignores it."}
                {data.routing_model.trusted && data.routing_model.strongest.length > 0 && (
                  <>
                    {" "}
                    It leans most on:{" "}
                    {data.routing_model.strongest
                      .slice(0, 5)
                      .map(([name, weight]) => `${name.replace("word:", "")} (${weight > 0 ? "harder" : "easier"})`)
                      .join(", ")}
                    .
                  </>
                )}
              </p>
            )}
            {data.routing.length === 0 ? (
              <p className="muted">
                Nothing learned yet. After enough runs of one kind, the router notes how often the quick path failed and
                starts planning first where it usually does.
              </p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th scope="col">Kind of task</th>
                    <th scope="col">Quick path fails</th>
                    <th scope="col">Runs seen</th>
                  </tr>
                </thead>
                <tbody>
                  {data.routing.map(([profile, risk, seen]) => (
                    <tr key={profile}>
                      <td>{profile}</td>
                      <td>{percent(risk)}</td>
                      <td>{seen}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="card">
            <h2>HTTP</h2>
            <div className="figures">
              <Figure name="TLS" value={data.http.tls ? "on" : "off"} />
              <Figure name="Open connections" value={data.http.active_connections} />
              <Figure name="Refused connections" value={data.http.refused_connections} />
              <Figure name="Requests" value={Object.values(data.http.requests).reduce((a, b) => a + b, 0)} />
              <Figure name="Rejected" value={Object.values(data.http.rejected).reduce((a, b) => a + b, 0)} />
            </div>
            {info?.metrics && (
              <p className="muted small">
                Prometheus can scrape the same numbers and more from <code>/metrics</code> with this token.
              </p>
            )}
          </div>

          {data.memory && (
            <div className="card">
              <h2>Memory</h2>
              <pre>{pretty(data.memory)}</pre>
            </div>
          )}
          {data.drift_alarms.length > 0 && (
            <div className="card">
              <h2>Drift alarms</h2>
              <pre>{pretty(data.drift_alarms)}</pre>
            </div>
          )}
        </>
      )}
    </section>
  );
}
