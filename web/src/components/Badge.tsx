import type { ReactNode } from "react";

export type Tone = "neutral" | "good" | "bad" | "warn" | "accent";

export function Badge({ tone = "neutral", title, children }: { tone?: Tone; title?: string; children: ReactNode }) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {children}
    </span>
  );
}
