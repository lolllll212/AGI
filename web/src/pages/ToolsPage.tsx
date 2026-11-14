import { Badge, type Tone } from "../components/Badge";
import { Loading, Notice } from "../components/Notice";
import { useAsync } from "../hooks/useAsync";
import type { Api } from "../services/api";

const RISK: Record<string, Tone> = { safe: "good", read: "good", write: "warn", network: "warn" };
const RISK_HELP: Record<string, string> = {
  safe: "Reads or computes; changes nothing",
  write: "Can change files in the workspace",
  network: "Can reach the network",
};

export function ToolsPage({ api }: { api: Api }) {
  const tools = useAsync(() => api.tools(), [api]);
  return (
    <section>
      <header className="page-head">
        <h1>Tools</h1>
      </header>
      <p className="muted">What the agent can use on this server. It picks the few that fit each task.</p>
      {tools.loading && <Loading what="tools" />}
      {tools.error && <Notice tone="error">{tools.error}</Notice>}
      <div className="grid">
        {tools.data?.map((tool) => (
          <article className="card tool" key={tool.name}>
            <h2>
              <code>{tool.name}</code> <Badge tone={RISK[tool.risk] ?? "neutral"} title={RISK_HELP[tool.risk]}>
                {tool.risk}
              </Badge>
            </h2>
            <p>{tool.description}</p>
            {Object.keys(tool.parameters).length > 0 && (
              <table>
                <tbody>
                  {Object.entries(tool.parameters).map(([name, schema]) => (
                    <tr key={name}>
                      <th scope="row">
                        <code>{name}</code>
                      </th>
                      <td className="muted small">{String(schema.type ?? "")}</td>
                      <td className="small">{String(schema.description ?? "")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
