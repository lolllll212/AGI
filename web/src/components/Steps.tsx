import type { Step } from "../services/api";
import { pretty } from "../utils/format";

/** What the agent thought, did and saw, one step at a time. */
export function Steps({ steps }: { steps: Step[] }) {
  if (steps.length === 0) return null;
  return (
    <details className="steps">
      <summary>
        {steps.length} {steps.length === 1 ? "step" : "steps"}
      </summary>
      <ol>
        {steps.map((step, index) => (
          <li key={index} className={step.ok ? "" : "step-failed"}>
            {step.thought && <p className="step-thought">{step.thought}</p>}
            {step.action && (
              <p className="step-action">
                <code>{step.action}</code>
                {Object.keys(step.args).length > 0 && <code className="step-args">{JSON.stringify(step.args)}</code>}
              </p>
            )}
            {step.observation && <pre className="step-observation">{pretty(step.observation)}</pre>}
          </li>
        ))}
      </ol>
    </details>
  );
}
