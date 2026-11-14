"""A small formal verifier for propositional reasoning.

Language models produce plausible-sounding inferences; this module checks
whether they actually *follow*. Formulas use ``~`` (not), ``&`` (and), ``|``
(or), ``->`` (implies) and ``<->`` (iff), with ``not/and/or`` accepted too.

    >>> entails(["rain -> wet", "rain"], "wet").valid
    True
    >>> entails(["rain -> wet", "wet"], "rain").counterexample   # affirming the consequent
    {'rain': False, 'wet': True}

Implementation: recursive-descent parser -> Tseitin CNF -> DPLL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

_TOKEN = re.compile(r"\s*(<->|->|[()~!&|]|[A-Za-z_][A-Za-z0-9_]*)")
_WORDS = {"not": "~", "and": "&", "or": "|", "implies": "->", "iff": "<->", "!": "~"}


class LogicSyntaxError(ValueError):
    pass


def parse(text: str):
    """Parse a formula into nested tuples: ``("imp", ("var", "a"), ("var", "b"))``."""
    tokens, position = [], 0
    text = text.strip()
    while position < len(text):
        match = _TOKEN.match(text, position)
        if not match:
            raise LogicSyntaxError(f"unexpected character at {position}: {text[position:position+10]!r}")
        tokens.append(_WORDS.get(match.group(1).lower(), match.group(1)))
        position = match.end()
    index = 0

    def peek():
        return tokens[index] if index < len(tokens) else None

    def take(expected=None):
        nonlocal index
        token = peek()
        if token is None or (expected and token != expected):
            raise LogicSyntaxError(f"expected {expected or 'a term'} but found {token!r} in {text!r}")
        index += 1
        return token

    def iff():
        left = implication()
        while peek() == "<->":
            take()
            left = ("iff", left, implication())
        return left

    def implication():
        left = disjunction()
        if peek() == "->":
            take()
            return ("imp", left, implication())      # right-associative
        return left

    def disjunction():
        left = conjunction()
        while peek() == "|":
            take()
            left = ("or", left, conjunction())
        return left

    def conjunction():
        left = negation()
        while peek() == "&":
            take()
            left = ("and", left, negation())
        return left

    def negation():
        if peek() == "~":
            take()
            return ("not", negation())
        if peek() == "(":
            take()
            inner = iff()
            take(")")
            return inner
        token = take()
        if not re.match(r"[A-Za-z_]", token):
            raise LogicSyntaxError(f"unexpected {token!r} in {text!r}")
        if token.lower() in ("true", "false"):
            return ("const", token.lower() == "true")
        return ("var", token)

    tree = iff()
    if index != len(tokens):
        raise LogicSyntaxError(f"unexpected {peek()!r} in {text!r}")
    return tree


class _CNF:
    """Tseitin encoder: one fresh literal per sub-formula, linear-size clauses."""

    def __init__(self):
        self.names: dict[str, int] = {}
        self.count = 0
        self.clauses: list[list[int]] = []

    def fresh(self) -> int:
        self.count += 1
        return self.count

    def literal(self, node) -> int:
        kind = node[0]
        if kind == "var":
            if node[1] not in self.names:
                self.names[node[1]] = self.fresh()
            return self.names[node[1]]
        if kind == "const":
            x = self.fresh()
            self.clauses.append([x if node[1] else -x])
            return x
        if kind == "not":
            return -self.literal(node[1])
        a, b, x = self.literal(node[1]), self.literal(node[2]), self.fresh()
        if kind == "and":
            self.clauses += [[-x, a], [-x, b], [x, -a, -b]]
        elif kind == "or":
            self.clauses += [[-x, a, b], [x, -a], [x, -b]]
        elif kind == "imp":
            self.clauses += [[-x, -a, b], [x, a], [x, -b]]
        else:  # iff
            self.clauses += [[-x, -a, b], [-x, a, -b], [x, a, b], [x, -a, -b]]
        return x

    def require(self, node, truth: bool = True) -> None:
        literal = self.literal(node)
        self.clauses.append([literal if truth else -literal])


def _dpll(clauses: list[list[int]], model: dict[int, bool]) -> dict[int, bool] | None:
    while True:                                   # unit propagation
        simplified, unit = [], None
        for clause in clauses:
            kept, satisfied = [], False
            for literal in clause:
                value = model.get(abs(literal))
                if value is None:
                    kept.append(literal)
                elif value == (literal > 0):
                    satisfied = True
                    break
            if satisfied:
                continue
            if not kept:
                return None                       # conflict
            if len(kept) == 1 and unit is None:
                unit = kept[0]
            simplified.append(kept)
        clauses = simplified
        if unit is None:
            break
        model = {**model, abs(unit): unit > 0}
    if not clauses:
        return model
    literal = min(clauses, key=len)[0]
    for choice in (literal > 0, literal <= 0):
        found = _dpll(clauses, {**model, abs(literal): choice})
        if found is not None:
            return found
    return None


def satisfiable(formulas: Iterable[str], negated: Iterable[str] = ()) -> dict[str, bool] | None:
    """A model making every formula true (and every ``negated`` one false), or ``None``."""
    cnf = _CNF()
    for formula in formulas:
        cnf.require(parse(formula), True)
    for formula in negated:
        cnf.require(parse(formula), False)
    model = _dpll(cnf.clauses, {})
    if model is None:
        return None
    return {name: model.get(number, False) for name, number in sorted(cnf.names.items())}


@dataclass(frozen=True)
class Proof:
    valid: bool
    counterexample: dict | None = None      # a world where premises hold and the claim fails

    def __bool__(self) -> bool:
        return self.valid


def entails(premises: Sequence[str], conclusion: str) -> Proof:
    """Does ``conclusion`` hold in every world where all ``premises`` hold?"""
    world = satisfiable(premises, negated=[conclusion])
    return Proof(world is None, world)


def consistent(premises: Sequence[str]) -> bool:
    return satisfiable(premises) is not None


@dataclass(frozen=True)
class ArgumentCheck:
    valid: bool
    failed_step: int | None = None          # index into ``steps``
    reason: str = ""
    counterexample: dict | None = None


def check_argument(premises: Sequence[str], steps: Sequence[str]) -> ArgumentCheck:
    """Verify a chain of reasoning step by step.

    Each step must follow from the premises plus the steps before it. The
    first step that does not is returned with a concrete counterexample.
    """
    if not consistent(premises):
        return ArgumentCheck(False, None, "the premises contradict each other")
    known = list(premises)
    for index, step in enumerate(steps):
        proof = entails(known, step)
        if not proof.valid:
            return ArgumentCheck(False, index, f"step {index + 1} ('{step}') does not follow",
                                 proof.counterexample)
        known.append(step)
    return ArgumentCheck(True)
