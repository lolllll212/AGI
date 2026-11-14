"""A minimal 2-D rigid-point physics engine and a projectile environment.

Small on purpose, but real: gravity, quadratic air drag, wind and bouncing are
integrated numerically, so there is no closed form the agent could have
memorised. The only way to know where the ball lands is to throw it.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from .env import Environment, Transition


@dataclass
class Body:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    mass: float = 1.0
    restitution: float = 0.0        # 0 = sticks on landing, 1 = perfectly elastic
    resting: bool = False


class PhysicsWorld:
    """Point masses under gravity, drag ``F = -k |v_rel| v_rel`` and horizontal wind."""

    def __init__(self, gravity: float = 9.81, drag: float = 0.0, wind: float = 0.0,
                 dt: float = 0.002):
        self.gravity, self.drag, self.wind, self.dt = gravity, drag, wind, dt
        self.bodies: list[Body] = []
        self.time = 0.0

    def add(self, body: Body) -> Body:
        self.bodies.append(body)
        return body

    def step(self) -> None:
        dt = self.dt
        for body in self.bodies:
            if body.resting:
                continue
            rel_x, rel_y = body.vx - self.wind, body.vy
            speed = math.hypot(rel_x, rel_y)
            ax = -self.drag * speed * rel_x / body.mass
            ay = -self.gravity - self.drag * speed * rel_y / body.mass
            body.vx += ax * dt                      # semi-implicit Euler
            body.vy += ay * dt
            body.x += body.vx * dt
            body.y += body.vy * dt
            if body.y <= 0.0 and body.vy < 0.0:     # ground contact
                body.y = 0.0
                body.vy = -body.vy * body.restitution
                if body.vy < 0.05:
                    body.vx = body.vy = 0.0
                    body.resting = True
        self.time += dt

    def run(self, max_time: float = 60.0) -> None:
        """Advance until every body is at rest (or ``max_time`` elapses)."""
        while self.time < max_time and not all(b.resting for b in self.bodies):
            self.step()


def ideal_range(angle_deg: float, speed: float, gravity: float = 9.81) -> float:
    """Textbook range of a projectile in a vacuum: ``v^2 sin(2a) / g``."""
    return speed * speed * math.sin(math.radians(2 * angle_deg)) / gravity


class ProjectileWorld(Environment):
    """Throw a ball: ``{"angle": degrees, "speed": m/s}`` -> where it lands.

    ``drag`` and ``wind`` are hidden from the agent. ``gust`` adds per-throw
    randomness so that observations are noisy, as real ones are.
    """

    bounds = {"angle": (10.0, 80.0), "speed": (5.0, 40.0)}

    def __init__(self, drag: float = 0.01, wind: float = -2.0, gravity: float = 9.81,
                 gust: float = 0.0, seed: int = 0):
        super().__init__()
        self.drag, self.wind, self.gravity, self.gust = drag, wind, gravity, gust
        self.rng = random.Random(seed)

    def act(self, action: Mapping[str, float]) -> Transition:
        angle = math.radians(min(max(action["angle"], 1.0), 89.0))
        speed = max(0.1, action["speed"])
        wind = self.wind + (self.rng.gauss(0.0, self.gust) if self.gust else 0.0)
        world = PhysicsWorld(self.gravity, self.drag, wind)
        ball = world.add(Body(vx=speed * math.cos(angle), vy=speed * math.sin(angle)))
        apex = 0.0
        while not ball.resting and world.time < 120.0:
            world.step()
            apex = max(apex, ball.y)
        observation = {"distance": round(ball.x, 4), "flight_time": round(world.time, 4),
                       "apex": round(apex, 4)}
        return Transition(dict(action), observation)
