"""Scripted language models: plain functions from a prompt to the text a model would return."""
from __future__ import annotations

import re

SALES_QUERY = ('Thought: query the file.\nAction: query_data\nAction Input: {"file": "sales.csv", "sql": '
               '"SELECT ROUND(SUM(amount), 2) FROM data WHERE region = \'%s\'"}')


def sales_model(prompt: str) -> str:
    """Answers "total amount for <region>" from sales.csv with one query; declines anything else."""
    task = prompt.rsplit("\n\nTask: ", 1)[-1]
    region = next((name for name in ("North", "South", "East") if name in task.split("\n")[0]), None)
    if region is None:
        return "Final Answer: I can only report regional totals."
    if "Observation:" not in task:
        return SALES_QUERY % region
    figure = float(re.findall(r"\d+\.\d+", task.rsplit("Observation:", 1)[1])[-1])
    return f"Thought: the query returned the total.\nFinal Answer: The {region} total is {figure:,.2f}."
