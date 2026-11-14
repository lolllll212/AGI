import { FormEvent, useState } from "react";
import { Api, ApiError } from "../services/api";
import { Notice } from "./Notice";

/** Asks for the API token and checks it before letting the console load. */
export function TokenGate({ onToken, reason }: { onToken: (token: string) => void; reason?: string }) {
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(reason ?? null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const token = value.trim();
    if (!token) return;
    setBusy(true);
    setError(null);
    try {
      await new Api(token).info();
      onToken(token);
    } catch (problem) {
      const status = problem instanceof ApiError ? problem.status : 0;
      setError(
        status === 401
          ? "That token was not accepted."
          : status === 429
            ? "Too many attempts. Wait a few seconds and try again."
            : problem instanceof Error
              ? problem.message
              : "Something went wrong.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="gate">
      <form className="card gate-card" onSubmit={submit}>
        <h1>agi console</h1>
        <p className="muted">
          This server needs an API token. It is the value of <code>AGI_API_TOKEN</code> (or your own key from the
          keys file) on the machine that runs it.
        </p>
        <label htmlFor="token">API token</label>
        <input
          id="token"
          type="password"
          autoComplete="off"
          autoFocus
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
        {error && <Notice tone="error">{error}</Notice>}
        <button className="primary" type="submit" disabled={busy || !value.trim()}>
          {busy ? "Checking…" : "Open console"}
        </button>
        <p className="muted small">The token stays in this browser tab and is forgotten when the tab closes.</p>
      </form>
    </main>
  );
}
