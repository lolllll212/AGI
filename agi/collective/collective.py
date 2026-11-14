"""A group of agents that grows its own toolbox, one verified tool at a time."""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..sandbox import SandboxResult
from .agents import ToolNeed
from .forge import ReviewContext, ToolForge, Verdict
from .ledger import Ledger
from .policy import SafetyPolicy


class Collective:
    """Agents + shared ledger + forge.

    ``request(need)`` asks every agent for a proposal and puts each through the
    forge, with all the *other* agents as reviewers. The first accepted
    proposal wins. Later needs may build on earlier tools (``depends_on``),
    which is the self-improvement loop: the group's capabilities compound, and
    every increment passed the same gates.
    """

    def __init__(self, agents: Sequence, policy: SafetyPolicy | None = None, seed: int = 0):
        ids = [agent.id for agent in agents]
        if len(set(ids)) != len(ids):
            raise ValueError("agent ids must be unique")
        self.agents = list(agents)
        self.ledger = Ledger(ids, seed=seed)
        self.forge = ToolForge(self.ledger, policy)

    def request(self, need: ToolNeed) -> list[Verdict]:
        context = ReviewContext(need=need, prelude=self._toolbox_source(), policy=self.forge.policy)
        proposals = []
        for agent in self.agents:            # each drafts without seeing the others' drafts
            proposal = agent.propose(need, context)
            if proposal is not None:
                proposals.append(proposal)
        verdicts = []
        for proposal in sorted(proposals, key=lambda p: p.id):
            verdict = self.forge.submit(proposal, self.agents, need)
            verdicts.append(verdict)
            if verdict.accepted:
                break
        return verdicts

    def improve(self, needs: Iterable[ToolNeed]) -> dict[str, list[Verdict]]:
        return {need.name: self.request(need) for need in needs}

    def call(self, name: str, *args: Any, agent: str | None = None) -> SandboxResult:
        """Use a tool, reading the registry from ``agent``'s own replica."""
        return self.forge.call(name, *args, at=agent)

    def tools(self, agent: str | None = None) -> dict[str, dict]:
        return self.forge.tools(at=agent)

    def _toolbox_source(self) -> str:
        return "\n\n".join(f"# {r['name']}: {r['description']}\n{r['code']}"
                           for r in self.tools().values())
