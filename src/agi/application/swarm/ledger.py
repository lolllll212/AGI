"""Shared state for a group of agents, built on `conflux-state`.

Each agent owns a Conflux :class:`~conflux.Replica`. Every write is an
:class:`~conflux.Action`, signed by its author, and replicas only absorb
actions that pass :func:`conflux.validate_envelope`: valid signature, known
agent, and a key inside that agent's *namespace*. Because the underlying
structures are CRDTs, replicas converge to the same state whatever order the
messages arrive in - no locks, no coordinator.

Namespaces are what turn "agents share state" into "agents share state
*safely*": an agent may publish proposals and reviews under its own name, but
only the kernel (the tool forge) can write the tool registry.
"""
from __future__ import annotations

import random
import secrets
from typing import Any, Iterable

from conflux import Action, AgentRegistry, Replica, ValidationError, sign_action, state_hash, validate_envelope

KERNEL_NAMESPACES = ("tools/", "verdicts/", "stats/", "audit/")


def agent_namespaces(agent_id: str) -> tuple:
    return (f"proposals/{agent_id}/", f"reviews/{agent_id}/", f"notes/{agent_id}/")


class Ledger:
    def __init__(self, agents: Iterable[str], kernel: str = "forge", seed: int = 0):
        self.kernel = kernel
        self.registry = AgentRegistry()
        self.replicas: dict[str, Replica] = {}
        self.rejected: list[dict] = []          # envelopes refused by honest replicas
        self._secrets: dict[str, str] = {}
        self._outbox: list[dict] = []
        self._rng = random.Random(seed)
        for agent in agents:
            self.join(agent, agent_namespaces(agent))
        self.join(kernel, KERNEL_NAMESPACES)

    def join(self, agent_id: str, namespaces: Iterable[str]) -> None:
        if agent_id in self.replicas:
            raise ValueError(f"agent id '{agent_id}' is already registered")
        self._secrets[agent_id] = secrets.token_hex(32)
        self.registry.register(agent_id, secret=self._secrets[agent_id], namespaces=tuple(namespaces))
        self.replicas[agent_id] = Replica(agent_id)

    # -- honest writes ------------------------------------------------------ #
    def _guard(self, agent_id: str, key: str) -> Replica:
        if not self.registry.namespace_allows(agent_id, key):
            raise PermissionError(f"'{agent_id}' may not write '{key}'")
        return self.replicas[agent_id]

    def _publish(self, agent_id: str, action: Action) -> Action:
        self._outbox.append(sign_action(action, self._secrets[agent_id]))
        return action

    def put(self, agent_id: str, key: str, value: Any) -> Action:
        """Last-writer-wins register."""
        return self._publish(agent_id, self._guard(agent_id, key).register_set(key, value))

    def add(self, agent_id: str, key: str, value: Any) -> Action:
        """Observed-remove set."""
        return self._publish(agent_id, self._guard(agent_id, key).set_add(key, value))

    def count(self, agent_id: str, key: str, by: int = 1) -> Action:
        """Increment-only counter contribution."""
        return self._publish(agent_id, self._guard(agent_id, key).counter_inc(key, by))

    # -- the network -------------------------------------------------------- #
    def inject(self, envelope: dict) -> None:
        """Put an arbitrary envelope on the wire, as any network participant could."""
        self._outbox.append(envelope)

    def craft(self, claimed_agent: str, key: str, value: Any, signed_with: str | None = None) -> dict:
        """Build an envelope without any checks. For tests and red-teaming:
        simulates a compromised agent (default: signs with its own real key)
        or an impersonator (``signed_with`` = some other agent's id)."""
        tick = max(r.timestamp() for r in self.replicas.values()) + 1
        action = Action(f"act-{claimed_agent}-x{secrets.token_hex(4)}", claimed_agent, tick,
                        "lww_set", key, {"value": value})
        return sign_action(action, self._secrets[signed_with or claimed_agent])

    def sync(self) -> int:
        """Deliver every pending envelope to every replica, in random order.

        Each envelope is validated first; invalid ones are dropped and logged.
        Returns the number of envelopes rejected in this round.
        """
        pending, self._outbox = self._outbox, []
        self._rng.shuffle(pending)
        accepted, refused = [], 0
        for envelope in pending:
            try:
                accepted.append(validate_envelope(envelope, self.registry))
            except (ValidationError, TypeError, KeyError) as exc:
                action = envelope.get("action", {}) if isinstance(envelope, dict) else {}
                self.rejected.append({"agent": action.get("agent"), "key": action.get("key"),
                                      "reason": str(exc)})
                refused += 1
        for replica in self.replicas.values():
            order = accepted[:]
            self._rng.shuffle(order)             # delivery order must not matter
            replica.absorb(order)
        return refused

    # -- reads -------------------------------------------------------------- #
    def read(self, key: str, at: str | None = None) -> Any:
        return self.replicas[at or self.kernel].read(key)

    def keys(self, prefix: str = "", at: str | None = None) -> list[str]:
        return sorted(k for k in self.replicas[at or self.kernel].state.keys() if k.startswith(prefix))

    def digest(self, at: str | None = None) -> str:
        return state_hash(self.replicas[at or self.kernel].state)

    def converged(self) -> bool:
        """True when every replica holds byte-identical state."""
        return len({state_hash(r.state) for r in self.replicas.values()}) == 1
