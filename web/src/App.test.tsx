// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

type Handler = (init: RequestInit | undefined) => { status?: number; body: unknown };

/** A stand-in server: answers each "METHOD path" from a table and records what was asked. */
function fakeServer(routes: Record<string, Handler>) {
  const calls: { key: string; headers: Record<string, string>; body: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: URL | string, init?: RequestInit) => {
      const url = new URL(String(input));
      const key = `${init?.method ?? "GET"} ${url.pathname}`;
      calls.push({
        key,
        headers: (init?.headers ?? {}) as Record<string, string>,
        body: init?.body ? JSON.parse(String(init.body)) : undefined,
      });
      const handler = routes[key];
      const { status = 200, body } = handler
        ? handler(init)
        : { status: 404, body: { error: { type: "not_found", message: `no route ${key}` } } };
      return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
    }),
  );
  return calls;
}

const health = (auth: boolean) => ({
  body: { status: "ok", model: true, model_circuit: "closed", uptime_s: 5, in_flight: 0, tools: 10, auth_required: auth },
});
const info = {
  body: { version: "0.6.0", model: "test-model", demo: false, examples: ["What is 2 + 2?"], caller: "default", sandbox: "Sandbox", metrics: true },
};
const result = {
  task: "What is 2 + 2?", answer: "2 + 2 = 4", mode: "reflex", system: 1, route: { reason: "plain arithmetic" },
  success: true, grounded: true, blocked: false, stop_reason: "answer",
  steps: [{ thought: "", action: "calculator", args: { expression: "2 + 2" }, observation: "4", ok: true }],
  tool_calls: 1, tool_errors: 0, llm_calls: 0, repairs: 0, compactions: 0, escalated: false,
  trace_id: "abc123", seconds: 0.004, error: null, flags: [],
};

beforeEach(() => {
  sessionStorage.clear();
  window.location.hash = "";
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("the console", () => {
  it("asks for a token when the server needs one, and sends it afterwards", async () => {
    const calls = fakeServer({
      "GET /health": () => health(true),
      "GET /v1/info": (init) =>
        (init?.headers as Record<string, string>)?.Authorization === "Bearer right-token"
          ? info
          : { status: 401, body: { error: { type: "unauthorized", message: "missing or wrong bearer token" } } },
    });
    render(<App />);
    const field = await screen.findByLabelText("API token");

    fireEvent.change(field, { target: { value: "wrong-token" } });
    fireEvent.click(screen.getByRole("button", { name: "Open console" }));
    expect((await screen.findByRole("alert")).textContent).toContain("was not accepted");

    fireEvent.change(field, { target: { value: "right-token" } });
    fireEvent.click(screen.getByRole("button", { name: "Open console" }));
    await screen.findByRole("heading", { name: "Chat" });
    expect(sessionStorage.getItem("agi.token")).toBe("right-token");
    await waitFor(() => expect(screen.getByText("test-model")).toBeTruthy());
    expect(calls.filter((c) => c.key === "GET /v1/info").slice(-1)[0]?.headers.Authorization).toBe("Bearer right-token");
  });

  it("runs an example task and shows the answer, how it was answered, and its steps", async () => {
    const calls = fakeServer({
      "GET /health": () => health(false),
      "GET /v1/info": () => info,
      "POST /v1/run": () => ({ body: result }),
    });
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "What is 2 + 2?" }));

    expect(await screen.findByText("2 + 2 = 4")).toBeTruthy();
    expect(screen.getByText("reflex")).toBeTruthy();
    expect(screen.getByText("1 tool call")).toBeTruthy();
    expect(screen.getByText("Why this mode: plain arithmetic")).toBeTruthy();
    expect(screen.getByRole("link", { name: "View trace" }).getAttribute("href")).toBe("#/runs/abc123");
    expect(screen.getByText("1 step")).toBeTruthy();

    const sent = calls.find((c) => c.key === "POST /v1/run")?.body as { task: string; mode: null; session: string; steps: boolean };
    expect(sent.task).toBe("What is 2 + 2?");
    expect(sent.mode).toBeNull();
    expect(sent.steps).toBe(true);
    expect(sent.session).toMatch(/^web-[0-9a-f]{18}$/);
  });

  it("points out a name in the answer that nothing the agent saw supports", async () => {
    fakeServer({
      "GET /health": () => health(false),
      "GET /v1/info": () => info,
      "POST /v1/run": () => ({ body: { ...result, answer: "4, as Vikram Rao expected", flags: ["unverified_name:Vikram Rao"] } }),
    });
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "What is 2 + 2?" }));
    expect(await screen.findByText("unverified: Vikram Rao")).toBeTruthy();
  });

  it("shows a failed request as an error instead of an answer", async () => {
    fakeServer({
      "GET /health": () => health(false),
      "GET /v1/info": () => info,
      "POST /v1/run": () => ({ status: 429, body: { error: { type: "rate_limited", message: "too many requests; slow down" } } }),
    });
    render(<App />);
    fireEvent.change(await screen.findByLabelText("Task"), { target: { value: "anything" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect((await screen.findByRole("alert")).textContent).toBe("too many requests; slow down");
  });

  it("says so plainly when the model is a scripted demonstration", async () => {
    fakeServer({
      "GET /health": () => health(false),
      "GET /v1/info": () => ({ body: { ...info.body, demo: true, model: "demonstration (scripted, offline)" } }),
    });
    render(<App />);
    expect(await screen.findByText(/Answers come from scripts/)).toBeTruthy();
  });

  it("returns to the token form when the server stops accepting the token", async () => {
    sessionStorage.setItem("agi.token", "old-token");
    fakeServer({
      "GET /health": () => health(true),
      "GET /v1/info": () => ({ status: 401, body: { error: { type: "unauthorized", message: "missing or wrong bearer token" } } }),
    });
    render(<App />);
    expect(await screen.findByLabelText("API token")).toBeTruthy();
    expect(sessionStorage.getItem("agi.token")).toBeNull();
    expect(screen.getByRole("alert").textContent).toContain("no longer accepted");
  });
});
