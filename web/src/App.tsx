import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "./components/Badge";
import { Notice } from "./components/Notice";
import { TokenGate } from "./components/TokenGate";
import { useAsync } from "./hooks/useAsync";
import { useHashRoute } from "./hooks/useHashRoute";
import { ChatPage } from "./pages/ChatPage";
import { RunsPage } from "./pages/RunsPage";
import { StatusPage } from "./pages/StatusPage";
import { ToolsPage } from "./pages/ToolsPage";
import { Api, loadToken, saveToken } from "./services/api";

const PAGES = [
  { id: "chat", name: "Chat" },
  { id: "runs", name: "Runs" },
  { id: "tools", name: "Tools" },
  { id: "status", name: "Status" },
];

export function App() {
  const [token, setToken] = useState(loadToken);
  const [expired, setExpired] = useState(false);
  const route = useHashRoute();
  const page = PAGES.some((p) => p.id === route[0]) ? route[0] : "chat";

  const signOut = useCallback((wasRejected: boolean) => {
    saveToken("");
    setToken("");
    setExpired(wasRejected);
  }, []);
  const api = useMemo(() => new Api(token, () => signOut(true)), [token, signOut]);

  const health = useAsync(() => api.health(), [api], 10000);
  const needsToken = health.data?.auth_required === true && !token;
  const info = useAsync(() => (needsToken || !health.data ? Promise.resolve(null) : api.info()), [api, needsToken, !health.data]);

  useEffect(() => {
    document.title = `${PAGES.find((p) => p.id === page)?.name ?? "Chat"} · agi console`;
  }, [page]);

  if (health.loading && !health.data) return <main className="gate muted">Connecting…</main>;
  if (!health.data) {
    return (
      <main className="gate">
        <div className="card gate-card">
          <h1>agi console</h1>
          <Notice tone="error">The server cannot be reached{health.error ? `: ${health.error}` : "."}</Notice>
          <button type="button" onClick={health.reload}>
            Try again
          </button>
        </div>
      </main>
    );
  }
  if (needsToken) {
    return (
      <TokenGate
        reason={expired ? "The token is no longer accepted. Enter it again." : undefined}
        onToken={(value) => {
          saveToken(value);
          setExpired(false);
          setToken(value);
        }}
      />
    );
  }

  const degraded = health.data.status !== "ok";
  return (
    <div className="shell">
      <nav className="side" aria-label="Sections">
        <a className="brand" href="#/chat">
          agi <span className="muted">console</span>
        </a>
        <ul>
          {PAGES.map((item) => (
            <li key={item.id}>
              <a href={`#/${item.id}`} className={item.id === page ? "active" : ""} aria-current={item.id === page ? "page" : undefined}>
                {item.name}
              </a>
            </li>
          ))}
        </ul>
        <div className="side-foot">
          <p className="small">
            <span className={`dot ${degraded ? "dot-bad" : "dot-good"}`} aria-hidden="true" />
            {degraded ? "Model unavailable" : "Running"}
          </p>
          <p className="small muted">{info.data?.model ?? (health.data.model ? "" : "No model configured")}</p>
          {info.data && <p className="small muted">v{info.data.version}</p>}
          {health.data.auth_required && (
            <button type="button" className="link" onClick={() => signOut(false)}>
              Sign out
            </button>
          )}
        </div>
      </nav>
      <main className="main">
        {info.data?.demo && (
          <Notice tone="warn">
            <Badge tone="warn">Demonstration</Badge> Answers come from scripts that replay the benchmark tasks, not from a
            language model. Start the server with <code>--model</code> to use a real one.
          </Notice>
        )}
        {degraded && (
          <Notice tone="error">
            The model has failed repeatedly, so calls to it are paused for a short while. Tasks that need no model still work.
          </Notice>
        )}
        {/* The chat stays mounted so a conversation survives a look at another page. */}
        <div hidden={page !== "chat"}>
          <ChatPage api={api} info={info.data} />
        </div>
        {page === "runs" && <RunsPage api={api} selected={route[1]} />}
        {page === "tools" && <ToolsPage api={api} />}
        {page === "status" && <StatusPage api={api} info={info.data} health={health.data} />}
      </main>
    </div>
  );
}
