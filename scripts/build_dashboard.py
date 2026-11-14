#!/usr/bin/env python3
"""Write monitoring/grafana/dashboards/agi-overview.json.

The dashboard is kept as this short description rather than edited by hand, so a metric that is
renamed in the server is renamed here in one place. Run it after changing the panels:

    python scripts/build_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "monitoring" / "grafana" / "dashboards" / "agi-overview.json"
PROMETHEUS = {"type": "prometheus", "uid": "prometheus"}
LOKI = {"type": "loki", "uid": "loki"}
STAT = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "colorMode": "value", "graphMode": "none"}
SERIES = {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}


class Board:
    def __init__(self) -> None:
        self.panels: list[dict] = []
        self.y = 0

    def add(self, title: str, kind: str, queries: list[str], x: int, width: int, height: int = 8, unit: str | None = None,
            description: str = "", names: list[str] | None = None, datasource: dict = PROMETHEUS,
            options: dict | None = None) -> None:
        targets = []
        for index, query in enumerate(queries):
            target = {"refId": chr(65 + index), "expr": query, "datasource": datasource}
            if names:
                target["legendFormat"] = names[index]
            targets.append(target)
        self.panels.append({
            "id": len(self.panels) + 1, "title": title, "type": kind, "description": description, "datasource": datasource,
            "gridPos": {"x": x, "y": self.y, "w": width, "h": height}, "targets": targets,
            "fieldConfig": {"defaults": {"unit": unit} if unit else {}, "overrides": []},
            "options": options if options is not None else (STAT if kind == "stat" else SERIES)})

    def next_row(self, height: int = 8) -> None:
        self.y += height


def build() -> dict:
    board = Board()
    board.add("Tasks per minute", "stat", ["sum(rate(agi_runs_total[5m])) * 60"], 0, 4, 4)
    board.add("Answered", "stat",
              ['sum(rate(agi_runs_total{outcome="success"}[15m])) / clamp_min(sum(rate(agi_runs_total[15m])), 0.001)'],
              4, 4, 4, unit="percentunit", description="Share of tasks that ended with an answer over the last 15 minutes.")
    board.add("Typical time per task", "stat",
              ["histogram_quantile(0.5, sum by (le) (rate(agi_run_duration_seconds_bucket[15m])))"],
              8, 4, 4, unit="s", description="Median over the last 15 minutes.")
    board.add("In progress", "stat", ["agi_runs_in_flight"], 12, 4, 4)
    board.add("Model circuit", "stat", ["agi_model_circuit_open"], 16, 4, 4,
              description="0 = closed (healthy). 1 = open: calls to the model are paused after repeated failures.")
    board.add("Conversations", "stat", ["agi_sessions"], 20, 4, 4)
    board.next_row(4)

    board.add("Tasks by mode", "timeseries", ["sum by (mode) (rate(agi_runs_total[5m])) * 60"], 0, 12, unit="short",
              description="Tasks per minute, by the mode that answered: reflex, react (reason-act) or deliberate.",
              names=["{{mode}}"])
    board.add("Tasks by outcome", "timeseries", ["sum by (outcome) (rate(agi_runs_total[5m])) * 60"], 12, 12, unit="short",
              description="Tasks per minute.", names=["{{outcome}}"])
    board.next_row()

    board.add("Time per task", "timeseries",
              ["histogram_quantile(0.5, sum by (le) (rate(agi_run_duration_seconds_bucket[5m])))",
               "histogram_quantile(0.95, sum by (le) (rate(agi_run_duration_seconds_bucket[5m])))"],
              0, 12, unit="s", names=["median", "95th percentile"])
    board.add("Model and tool calls", "timeseries",
              ["rate(agi_llm_calls_total[5m]) * 60", "rate(agi_tool_calls_total[5m]) * 60",
               "rate(agi_tool_errors_total[5m]) * 60", "rate(agi_repairs_total[5m]) * 60"],
              12, 12, unit="short", description="Per minute.",
              names=["model calls", "tool calls", "tool errors", "repaired calls"])
    board.next_row()

    board.add("Escalations, compactions, unsupported answers", "timeseries",
              ["rate(agi_escalations_total[5m]) * 60", "rate(agi_context_compactions_total[5m]) * 60",
               "rate(agi_ungrounded_answers_total[5m]) * 60"],
              0, 12, unit="short", description="Per minute.",
              names=["escalations", "context compactions", "answers with unsupported figures"])
    board.add("HTTP responses by status", "timeseries", ["sum by (status) (rate(agi_http_requests_total[5m]))"],
              12, 12, unit="reqps", names=["{{status}}"])
    board.next_row()

    board.add("HTTP response time by route (95th percentile)", "timeseries",
              ["histogram_quantile(0.95, sum by (le, route) (rate(agi_http_request_duration_seconds_bucket[5m])))"],
              0, 12, unit="s", names=["{{route}}"])
    board.add("Refused and rejected", "timeseries",
              ["sum by (reason) (rate(agi_http_rejected_total[5m]))", "rate(agi_http_connections_refused_total[5m])",
               "agi_http_connections_active"],
              12, 12, unit="short", names=["rejected: {{reason}}", "connections refused", "open connections"])
    board.next_row()

    board.add("Trace spans exported", "timeseries", ["sum by (result) (rate(agi_trace_spans_exported_total[5m]))"],
              0, 8, unit="short", names=["{{result}}"])
    board.add("Drift alarms", "stat", ["agi_drift_alarms"], 8, 4)
    board.add("Uptime", "stat", ["agi_uptime_seconds"], 12, 4, unit="s")
    board.add("Tools", "stat", ["agi_tools"], 16, 4)
    board.add("Version", "stat", ["agi_build_info"], 20, 4, names=["{{version}}"],
              options={**STAT, "colorMode": "none", "textMode": "name"})
    board.next_row()

    board.add("Requests that failed (logs)", "logs", ['{service="api"} | json | status >= 400'], 0, 24, 10, datasource=LOKI,
              options={"showTime": True, "wrapLogMessage": True, "sortOrder": "Descending", "enableLogDetails": True})

    return {"uid": "agi-overview", "title": "Agent overview", "tags": ["agi"], "timezone": "browser", "schemaVersion": 39,
            "version": 1, "refresh": "30s", "time": {"from": "now-6h", "to": "now"}, "editable": True, "graphTooltip": 1,
            "panels": board.panels, "templating": {"list": []}, "annotations": {"list": []}}


def render() -> str:
    return json.dumps(build(), indent=2) + "\n"


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(build()['panels'])} panels)")
