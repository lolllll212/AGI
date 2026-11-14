"""An example plugin: two small numeric tools.

Start the server with ``--plugins plugins`` and the agent gets them alongside the standard tools.
A plugin is any folder with a ``plugin.py`` that defines ``tools()``.
"""
from __future__ import annotations

import statistics

from agi.domain.entities.tool import Param, Tool, ToolError


def describe_numbers(numbers: list) -> dict:
    try:
        values = [float(n) for n in numbers]
    except (TypeError, ValueError):
        raise ToolError("numbers must be a list of numbers") from None
    if not values:
        raise ToolError("numbers is empty")
    return {"count": len(values), "sum": sum(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def prime_factors(n: int) -> list[int]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ToolError("n must be a whole number of at least 2")
    if n > 10 ** 12:
        raise ToolError("n is too large for this tool (limit: 10^12)")
    factors, divisor = [], 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1 if divisor == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def tools() -> list[Tool]:
    return [
        Tool("describe_numbers", "Count, sum, mean, median, minimum, maximum and standard deviation of a list of numbers.",
             describe_numbers, {"numbers": Param("array", "the numbers to summarise", items="number")}),
        Tool("prime_factors", "The prime factors of a whole number, smallest first (for example 84 -> 2, 2, 3, 7).",
             prime_factors, {"n": Param("integer", "a whole number from 2 to 10^12")}),
    ]
