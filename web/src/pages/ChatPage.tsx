import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { Notice } from "../components/Notice";
import { RunMeta } from "../components/RunMeta";
import { Steps } from "../components/Steps";
import type { Api, Info, Mode, RunResult } from "../services/api";
import { newSessionId } from "../utils/format";

interface Turn {
  id: number;
  task: string;
  result?: RunResult;
  error?: string;
}

const MODES: { value: Mode | ""; label: string }[] = [
  { value: "", label: "Automatic" },
  { value: "reflex", label: "Reflex" },
  { value: "react", label: "Reason-act" },
  { value: "deliberate", label: "Deliberate" },
];

export function ChatPage({ api, info }: { api: Api; info: Info | null }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<Mode | "">("");
  const [busy, setBusy] = useState(false);
  const [session, setSession] = useState(newSessionId);
  const counter = useRef(0);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView?.({ block: "end" });
  }, [turns]);

  async function ask(task: string) {
    const text = task.trim();
    if (!text || busy) return;
    const id = ++counter.current;
    setTurns((all) => [...all, { id, task: text }]);
    setDraft("");
    setBusy(true);
    try {
      const result = await api.run(text, mode || null, session);
      setTurns((all) => all.map((turn) => (turn.id === id ? { ...turn, result } : turn)));
    } catch (problem) {
      const error = problem instanceof Error ? problem.message : String(problem);
      setTurns((all) => all.map((turn) => (turn.id === id ? { ...turn, error } : turn)));
    } finally {
      setBusy(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void ask(draft);
  }

  function onKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void ask(draft);
    }
  }

  function startOver() {
    if (turns.some((turn) => turn.result)) api.endSession(session).catch(() => undefined);
    setTurns([]);
    setSession(newSessionId());
  }

  const examples = info?.examples ?? [];

  return (
    <section className="chat">
      <header className="page-head">
        <h1>Chat</h1>
        <button type="button" onClick={startOver} disabled={busy || turns.length === 0}>
          New conversation
        </button>
      </header>

      <div className="conversation" aria-live="polite">
        {turns.length === 0 && (
          <div className="empty">
            <p>
              Give the agent a task. It decides whether to answer directly, work through it with tools, or plan first,
              and every answer shows which of those it did.
            </p>
            {examples.length > 0 && (
              <>
                <p className="muted small">{info?.demo ? "The demonstration model can do these:" : "For example:"}</p>
                <ul className="examples">
                  {examples.slice(0, info?.demo ? examples.length : 6).map((example) => (
                    <li key={example}>
                      <button type="button" className="example" onClick={() => void ask(example)} disabled={busy}>
                        {example}
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
        {turns.map((turn) => (
          <article key={turn.id} className="turn">
            <div className="bubble bubble-user">{turn.task}</div>
            {!turn.result && !turn.error && <div className="bubble bubble-agent pending">Working…</div>}
            {turn.error && <Notice tone="error">{turn.error}</Notice>}
            {turn.result && (
              <div className="bubble bubble-agent">
                {turn.result.answer ? (
                  <p className="answer">{turn.result.answer}</p>
                ) : (
                  <p className="answer muted">
                    No answer ({turn.result.stop_reason || "stopped"}
                    {turn.result.error ? `: ${turn.result.error}` : ""}).
                  </p>
                )}
                <RunMeta result={turn.result} />
                {turn.result.route?.reason && <p className="muted small">Why this mode: {turn.result.route.reason}</p>}
                <Steps steps={turn.result.steps} />
              </div>
            )}
          </article>
        ))}
        <div ref={end} />
      </div>

      <form className="composer" onSubmit={submit}>
        <label className="sr-only" htmlFor="task">
          Task
        </label>
        <textarea
          id="task"
          rows={2}
          placeholder="Ask something, or describe a task…"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onKey}
          maxLength={20000}
        />
        <div className="composer-row">
          <label htmlFor="mode" className="muted small">
            Mode
          </label>
          <select id="mode" value={mode} onChange={(event) => setMode(event.target.value as Mode | "")}>
            {MODES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <span className="muted small grow">Enter sends, Shift+Enter adds a line</span>
          <button className="primary" type="submit" disabled={busy || !draft.trim()}>
            {busy ? "Working…" : "Send"}
          </button>
        </div>
      </form>
    </section>
  );
}
