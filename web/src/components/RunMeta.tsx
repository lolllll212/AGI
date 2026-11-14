import type { RunResult } from "../services/api";
import { seconds } from "../utils/format";
import { Badge } from "./Badge";

const MODE_HELP: Record<string, string> = {
  reflex: "Answered directly, without the model reasoning step by step",
  react: "Reason-act loop: think, use a tool, look at the result, repeat",
  deliberate: "Planned first, then carried out and checked the plan",
};

/** The facts about a finished run: how it was answered, how long it took, and anything worth a warning. */
export function RunMeta({ result }: { result: RunResult }) {
  const unverified = result.flags.filter((flag) => flag.startsWith("unverified_name:")).map((flag) => flag.slice(16));
  return (
    <div className="meta">
      <Badge tone="accent" title={MODE_HELP[result.mode]}>
        {result.mode}
      </Badge>
      <Badge>{seconds(result.seconds)}</Badge>
      <Badge>
        {result.tool_calls} tool {result.tool_calls === 1 ? "call" : "calls"}
      </Badge>
      <Badge>
        {result.llm_calls} model {result.llm_calls === 1 ? "call" : "calls"}
      </Badge>
      {result.escalated && <Badge tone="warn" title="A faster mode failed, so a slower one took over">escalated</Badge>}
      {result.repairs > 0 && <Badge tone="warn" title="Failed tool calls that were repaired">{result.repairs} repaired</Badge>}
      {result.tool_errors > 0 && <Badge tone="warn">{result.tool_errors} tool errors</Badge>}
      {result.compactions > 0 && (
        <Badge title="Earlier steps were condensed to stay inside the context window">{result.compactions} compactions</Badge>
      )}
      {result.blocked && <Badge tone="bad">blocked by a guardrail</Badge>}
      {result.grounded === false && (
        <Badge tone="bad" title="The answer contains figures that no tool result supports">
          unsupported figures
        </Badge>
      )}
      {unverified.length > 0 && (
        <Badge tone="warn" title="The answer names these, and nothing the agent looked at mentions them">
          unverified: {unverified.join(", ")}
        </Badge>
      )}
      {result.success === false && !result.blocked && <Badge tone="bad">{result.stop_reason || "failed"}</Badge>}
      {result.trace_id && (
        <a className="meta-link" href={`#/runs/${encodeURIComponent(result.trace_id)}`}>
          View trace
        </a>
      )}
    </div>
  );
}
