"""A small benchmark of real-world tasks for a tool-using agent.

Each task comes with fixture files, a ``check`` that compares the agent's answer
with a truth computed independently in plain Python, and a *reference
solution* - a sequence of tool calls known to solve it. Replaying the
references (:func:`reference_llm`) proves the suite is solvable with the
built-in tools; running a real model through :func:`run_suite` measures the
model.

One task is deliberately unanswerable from the data. The right answer there is
to say so; inventing a figure fails it.
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..core.react import ReActAgent, Run, numbers_in

SALES = """region,product,units,amount
North,Widget,12,"1,200.50"
South,Widget,7,700.00
East,Gadget,3,"2,450.00"
North,Gadget,2,"1,630.25"
South,Gizmo,15,975.40
East,Widget,9,899.10
North,Gizmo,4,260.00
South,Gadget,1,815.75
East,Gizmo,6,390.00
Central,Widget,20,"1,990.00"
"""
CUSTOMERS = """customer_id,name,city
1,Asha,Pune
2,Bilal,Patna
3,Chitra,Pune
4,Dev,Delhi
5,Esha,Patna
"""
ORDERS = """order_id,customer_id,amount
100,1,250.00
101,2,120.50
102,3,"1,075.25"
103,1,99.75
104,4,640.00
105,5,310.00
106,3,45.00
"""
LOG = """2025-03-01 09:00:01 INFO  service started
2025-03-01 09:02:13 ERROR database timeout after 30s
2025-03-01 09:02:14 WARN  retrying request 1/4
2025-03-01 09:05:40 ERROR upstream returned 502
2025-03-01 09:07:02 INFO  cache warmed (1200 keys)
2025-03-01 09:11:27 ERROR database timeout after 30s
2025-03-01 09:11:29 WARN  retrying request 2/4
2025-03-01 09:15:55 ERROR disk quota exceeded on /var/data
2025-03-01 09:20:00 INFO  health check ok
2025-03-01 09:31:08 ERROR database timeout after 30s
2025-03-01 09:40:12 ERROR upstream returned 502
2025-03-01 09:45:00 INFO  health check ok
"""
CONFIG = {"service": {"name": "billing-api", "timeout_seconds": 45, "retries": 4, "port": 8443},
          "limits": {"requests_per_minute": 600, "burst": 50}}
FIXTURES = {"sales.csv": SALES, "customers.csv": CUSTOMERS, "orders.csv": ORDERS, "app.log": LOG,
            "config.json": json.dumps(CONFIG, indent=2) + "\n"}


def write_fixtures(directory: str | Path, extra: dict[str, str] | None = None) -> Path:
    """Write the shared fixture files, plus a task's own ``extra`` files, into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in {**FIXTURES, **(extra or {})}.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return directory


# -- truth, computed without any agent tool ---------------------------------- #
def _rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _money(text: str) -> float:
    return float(text.replace(",", ""))


_BY_REGION: dict[str, float] = {}
for _row in _rows(SALES):
    _BY_REGION[_row["region"]] = _BY_REGION.get(_row["region"], 0.0) + _money(_row["amount"])
TOP_REGION, TOP_TOTAL = max(_BY_REGION.items(), key=lambda kv: kv[1])
_PUNE = {r["customer_id"] for r in _rows(CUSTOMERS) if r["city"] == "Pune"}
PUNE_TOTAL = sum(_money(r["amount"]) for r in _rows(ORDERS) if r["customer_id"] in _PUNE)
_ERRORS = [line.split("ERROR", 1)[1].strip() for line in LOG.splitlines() if " ERROR " in line]
ERROR_COUNT, (TOP_ERROR, TOP_ERROR_COUNT) = len(_ERRORS), Counter(_ERRORS).most_common(1)[0]
COMPOUND = round(12500 * 1.065 ** 8, 2)
DAYS = (_dt.date(2025, 3, 1) - _dt.date(2024, 2, 10)).days
AVERAGE_UNITS = sum(int(r["units"]) for r in _rows(SALES)) / len(_rows(SALES))
WORST_WAIT = CONFIG["service"]["timeout_seconds"] * (CONFIG["service"]["retries"] + 1)


def has_number(text: str, value: float, tolerance: float = 0.01) -> bool:
    """Does ``text`` state ``value`` (thousands separators and rounding tolerated)?"""
    return any(abs(found - value) <= tolerance for found, _ in numbers_in(text))


@dataclass
class EvalTask:
    name: str
    prompt: str
    check: Callable[[str, Path], bool]
    reference: list[tuple[str, dict]] = field(default_factory=list)
    conclude: Callable[[list[str]], str] = lambda observations: observations[-1]
    skill: str = ""
    files: dict[str, str] = field(default_factory=dict)     # extra workspace files this task needs


def _last_number(text: str) -> str:
    return re.findall(r"-?\d+(?:\.\d+)?", text)[-1]


def _report_ok(answer: str, workspace: Path) -> bool:
    report = workspace / "report.txt"
    return report.is_file() and re.search(
        rf"average_units\s*=\s*{AVERAGE_UNITS:.2f}\b", report.read_text()) is not None


def _admits_missing(answer: str, workspace: Path) -> bool:
    lowered = answer.lower()
    admits = re.search(r"cannot|can't|unable|not (available|possible|present|contain|include|"
                       r"recorded|in the)|no (west|date|such|data|record|information)|does(n't| not)|"
                       r"missing|impossible", lowered) is not None
    return admits and not any(v > 31 and d == 2 for v, d in numbers_in(answer))


def _top_error(observations: list[str]) -> str:
    messages = [re.sub(r"^.*?ERROR\s+", "", line) for line in observations[-1].splitlines()
                if "ERROR" in line]
    top, count = Counter(messages).most_common(1)[0]
    return f"There are {len(messages)} ERROR lines; the most frequent is '{top}' ({count} times)."


SUITE: list[EvalTask] = [
    EvalTask("compound_interest",
             "A deposit of 12,500 earns 6.5% interest compounded annually. What is it worth after "
             "8 years? Give the amount to 2 decimal places.",
             lambda a, w: has_number(a, COMPOUND),
             [("calculator", {"expression": "round(12500 * 1.065 ** 8, 2)"})],
             lambda o: f"After 8 years the deposit is worth {o[-1]}.", "arithmetic"),
    EvalTask("top_region",
             "In sales.csv, which region has the highest total sales amount, and what is that total?",
             lambda a, w: TOP_REGION.lower() in a.lower() and has_number(a, TOP_TOTAL),
             [("query_data", {"file": "sales.csv", "sql": "select region, round(sum(amount), 2) as total "
                              "from data group by region order by total desc limit 1"})],
             lambda o: "{} has the highest total sales amount: {}.".format(
                 *[c.strip() for c in o[-1].splitlines()[-1].split("|")]), "data analysis"),
    EvalTask("pune_orders",
             "Using orders.csv and customers.csv, what is the total order amount placed by customers "
             "who live in Pune?",
             lambda a, w: has_number(a, PUNE_TOTAL),
             [("query_data", {"file": "orders.csv, customers.csv", "sql":
                              "select round(sum(o.amount), 2) as total from data o join customers c "
                              "on o.customer_id = c.customer_id where c.city = 'Pune'"})],
             lambda o: f"Customers in Pune placed orders worth {_last_number(o[-1])} in total.",
             "multi-file join"),
    EvalTask("log_errors",
             "How many ERROR lines are there in app.log, and which error message occurs most often?",
             lambda a, w: has_number(a, ERROR_COUNT, 0) and TOP_ERROR.lower() in a.lower(),
             [("search_files", {"pattern": " ERROR ", "path": "app.log"})], _top_error, "log analysis"),
    EvalTask("days_between",
             "How many days are there from 2024-02-10 to 2025-03-01?",
             lambda a, w: has_number(a, DAYS, 0),
             [("python", {"code": "import datetime\nresult = (datetime.date(2025, 3, 1) - "
                                  "datetime.date(2024, 2, 10)).days"})],
             lambda o: f"There are {_last_number(o[-1])} days between the two dates.", "date arithmetic"),
    EvalTask("worst_case_wait",
             "config.json describes a service that waits `timeout_seconds` for each attempt and makes "
             "`retries` further attempts after the first one fails. What is the worst-case total wait "
             "in seconds?",
             lambda a, w: has_number(a, WORST_WAIT, 0),
             [("read_file", {"path": "config.json"}),
              ("calculator", {"expression": f"{CONFIG['service']['timeout_seconds']} * "
                                            f"({CONFIG['service']['retries']} + 1)"})],
             lambda o: f"The worst-case total wait is {o[-1]} seconds.", "read then compute"),
    EvalTask("write_report",
             "Compute the average number of units per row in sales.csv and save it in a new file "
             "report.txt containing exactly one line of the form average_units=<value with 2 decimals>.",
             _report_ok,
             [("query_data", {"file": "sales.csv", "sql": "select avg(units) as average from data"}),
              ("write_file", {"path": "report.txt", "content": f"average_units={AVERAGE_UNITS:.2f}\n"})],
             lambda o: f"Saved average_units={AVERAGE_UNITS:.2f} to report.txt.", "act on the world"),
    EvalTask("unanswerable",
             "According to sales.csv, what was the total sales amount for the West region in March 2019?",
             _admits_missing,
             [("query_data", {"file": "sales.csv"}),
              ("query_data", {"file": "sales.csv", "sql": "select distinct region from data"})],
             lambda o: "This cannot be answered from sales.csv: there is no West region in the data "
                       "and the file has no date column.", "knowing what is not known"),
]


def reference_llm(task: EvalTask) -> Callable[[str], str]:
    """A stand-in "model" that replays a task's reference solution.

    It reads the real observations back out of the prompt, so the final answer
    is built from what the tools actually returned.
    """
    def reply(prompt: str) -> str:
        pad = prompt.rsplit("\n\nTask: ", 1)[-1]
        observations = [o.strip() for o in
                        re.findall(r"^Observation: (.*?)(?=^Thought:|^Action:|\Z)", pad, re.S | re.M)]
        if len(observations) < len(task.reference):
            tool, args = task.reference[len(observations)]
            return f"Thought: use {tool}.\nAction: {tool}\nAction Input: {json.dumps(args)}"
        return f"Thought: the observations answer the task.\nFinal Answer: {task.conclude(observations)}"
    return reply


@dataclass
class EvalResult:
    task: str
    success: bool
    run: Run

    def row(self) -> str:
        run = self.run
        grounded = {True: "yes", False: "NO", None: "-"}[run.grounded]
        return (f"{'PASS' if self.success else 'FAIL'}  {self.task:<18} {run.tool_calls:>2} calls  "
                f"{run.tool_errors} errors  grounded: {grounded:<3}  {run.stop_reason}")


@dataclass
class EvalReport:
    results: list[EvalResult]

    @property
    def passed(self) -> int:
        return sum(r.success for r in self.results)

    @property
    def success_rate(self) -> float:
        return self.passed / max(1, len(self.results))

    @property
    def runs(self) -> list[Run]:
        return [r.run for r in self.results]

    def __str__(self) -> str:
        calls = sum(r.run.tool_calls for r in self.results)
        return ("\n".join(r.row() for r in self.results)
                + f"\n{self.passed}/{len(self.results)} tasks passed, {calls} tool calls in total")


def run_suite(make_agent: Callable[[Path, EvalTask], ReActAgent],
              tasks: Sequence[EvalTask] = tuple(SUITE), workdir: str | Path | None = None) -> EvalReport:
    """Run each task in a fresh workspace. ``make_agent(workspace, task)`` builds the agent."""
    results = []
    for task in tasks:
        with tempfile.TemporaryDirectory(dir=workdir) as scratch:
            workspace = write_fixtures(Path(scratch) / "workspace", task.files)
            agent = make_agent(workspace, task)
            run = agent.run(task.prompt, check=lambda answer: task.check(answer, workspace))
            results.append(EvalResult(task.name, bool(run.success), run))
    return EvalReport(results)
