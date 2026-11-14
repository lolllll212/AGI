import math
import random

import pytest

from agi.world import (Body, CausalLearner, CodeWorld, FunctionWorld, GroundedAgent,
                       LocalRegressor, PhysicsWorld, ProjectileWorld, SCMWorld, WorldModel,
                       ideal_range, pearson)


# ---------------------------------------------------------------- physics --- #
@pytest.mark.parametrize("angle, speed", [(45, 20), (30, 15), (60, 25)])
def test_vacuum_trajectory_matches_the_closed_form(angle, speed):
    observed = ProjectileWorld(drag=0.0, wind=0.0).step({"angle": angle, "speed": speed}).observation
    assert observed["distance"] == pytest.approx(ideal_range(angle, speed), rel=0.01)
    theta = math.radians(angle)
    assert observed["flight_time"] == pytest.approx(2 * speed * math.sin(theta) / 9.81, rel=0.01)
    assert observed["apex"] == pytest.approx((speed * math.sin(theta)) ** 2 / (2 * 9.81), rel=0.01)


def test_drag_and_headwind_shorten_the_throw():
    throw = {"angle": 45, "speed": 20}
    vacuum = ProjectileWorld(drag=0.0, wind=0.0).step(throw).observation["distance"]
    drag = ProjectileWorld(drag=0.01, wind=0.0).step(throw).observation["distance"]
    headwind = ProjectileWorld(drag=0.01, wind=-5.0).step(throw).observation["distance"]
    tailwind = ProjectileWorld(drag=0.01, wind=5.0).step(throw).observation["distance"]
    assert headwind < drag < vacuum and drag < tailwind


def test_bouncing_loses_energy_and_comes_to_rest():
    world = PhysicsWorld()
    ball = world.add(Body(y=5.0, restitution=0.6))
    peaks, rising, top = [], False, 0.0
    while not ball.resting and world.time < 30:
        world.step()
        if ball.vy > 0:
            rising, top = True, max(top, ball.y)
        elif rising:
            peaks.append(top)
            rising, top = False, 0.0
    assert ball.resting and ball.y == 0.0
    assert len(peaks) >= 3 and all(a > b for a, b in zip(peaks, peaks[1:]))
    assert peaks[0] == pytest.approx(5.0 * 0.6 ** 2, rel=0.05)


def test_environment_logs_transitions_and_resets():
    env = ProjectileWorld()
    env.step({"angle": 40, "speed": 10})
    env.step({"angle": 50, "speed": 10})
    assert len(env.history) == 2 and env.history[0].action == {"angle": 40, "speed": 10}
    env.reset()
    assert env.history == []


def test_gusts_make_observations_noisy_but_reproducible():
    throws = lambda seed: [ProjectileWorld(gust=1.0, seed=seed).step({"angle": 45, "speed": 20})
                           .observation["distance"] for _ in range(1)]
    assert throws(1) == throws(1) and throws(1) != throws(2)


# ------------------------------------------------------------ world model --- #
def test_local_regressor_recovers_a_linear_function():
    rng = random.Random(0)
    model = LocalRegressor()
    assert model.predict([0.5, 0.5]) == (0.0, math.inf)
    for _ in range(60):
        x, y = rng.random(), rng.random()
        model.add([x, y], 3 * x - 2 * y + 1)
    estimate, distance = model.predict([0.4, 0.6])
    assert estimate == pytest.approx(3 * 0.4 - 2 * 0.6 + 1, abs=0.02) and distance < 0.2


def test_surprise_is_recorded_and_shrinks_with_experience():
    env = ProjectileWorld(drag=0.01, wind=-2.0)
    model = WorldModel(env.bounds, prior=lambda a: ideal_range(a["angle"], a["speed"]))
    throw = {"angle": 45.0, "speed": 20.0}
    first = model.update(throw, env.step(throw).observation["distance"])
    assert first.surprise > 5                       # the textbook is badly wrong here
    assert model.predict(throw) == pytest.approx(first.observed, abs=1e-6)
    nearby = {"angle": 46.0, "speed": 20.5}
    second = model.update(nearby, env.step(nearby).observation["distance"])
    assert second.surprise < first.surprise / 3


@pytest.mark.parametrize("target, seed", [(20.0, 0), (30.0, 1), (45.0, 2)])
def test_grounded_agent_beats_its_own_prior(target, seed):
    env = ProjectileWorld(drag=0.01, wind=-2.0)
    prior = lambda a: ideal_range(a["angle"], a["speed"])
    agent = GroundedAgent(env, WorldModel(env.bounds, prior=prior), "distance", seed=seed)
    log = agent.practice(target, 10)
    assert log[0].miss > 2.0                         # acting on belief alone misses
    assert log[-1].miss < 1.0 and log[-1].miss < log[0].miss / 4
    assert all(lo <= log[-1].action[k] <= hi for k, (lo, hi) in env.bounds.items())


def test_agent_without_any_prior_learns_from_babbling():
    env = ProjectileWorld(drag=0.01, wind=-2.0)
    agent = GroundedAgent(env, WorldModel(env.bounds), "distance", seed=0, warmup=6)
    log = agent.practice(30.0, 18)
    assert min(e.miss for e in log[-4:]) < 1.0 < max(e.miss for e in log[:6])


# ----------------------------------------------------------------- causal --- #
def coin(p, rng):
    return int(rng.random() < p)


def garden(seed=3):
    return SCMWorld({
        "sun": lambda v, r: coin(0.5, r),
        "ice_cream": lambda v, r: coin(0.85 if v["sun"] else 0.10, r),
        "sunburn": lambda v, r: coin(0.70 if v["sun"] else 0.05, r),
        "rain": lambda v, r: coin(0.3, r),
        "sprinkler": lambda v, r: coin(0.4, r),
        "wet_grass": lambda v, r: coin(0.95 if (v["rain"] or v["sprinkler"]) else 0.02, r),
        "slippery": lambda v, r: coin(0.90 if v["wet_grass"] else 0.03, r),
    }, hidden=["sun"], seed=seed)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_discovers_the_true_graph_and_flags_confounding(seed):
    graph = CausalLearner(garden(seed), seed=seed).discover()
    assert set(graph.direct) == {("rain", "wet_grass"), ("sprinkler", "wet_grass"),
                                 ("wet_grass", "slippery")}
    assert graph.causes("rain", "slippery") and ("rain", "slippery") not in graph.direct
    assert not graph.causes("slippery", "rain")                       # direction matters
    assert [(a, b) for a, b, _ in graph.spurious] == [("ice_cream", "sunburn")]
    assert graph.parents("wet_grass") == ["rain", "sprinkler"]
    assert graph.children("wet_grass") == ["slippery"]
    assert "NOT causal" in graph.explain()


def test_observation_alone_would_be_fooled():
    learner = CausalLearner(garden(), seed=0)
    assert learner.correlations(2000)[("ice_cream", "sunburn")] > 0.3
    delta, z = learner.experiment("ice_cream", "sunburn")
    assert abs(delta) < 0.1


def test_interaction_effects_need_the_right_context():
    world = SCMWorld({"a": lambda v, r: coin(0.5, r), "b": lambda v, r: coin(0.5, r),
                      "both": lambda v, r: int(v["a"] and v["b"])}, seed=0)
    graph = CausalLearner(world, seed=0).discover()
    assert set(graph.direct) == {("a", "both"), ("b", "both")}


def test_continuous_variables_and_custom_levels():
    world = SCMWorld({"dose": lambda v, r: r.uniform(0, 10),
                      "response": lambda v, r: 2.0 * v["dose"] + r.gauss(0, 1),
                      "noise": lambda v, r: r.gauss(0, 1)}, seed=0)
    graph = CausalLearner(world, levels={"dose": (0, 10), "response": (0, 20), "noise": (-1, 1)},
                          seed=0).discover()
    assert set(graph.direct) == {("dose", "response")}
    assert graph.direct[("dose", "response")] == pytest.approx(20.0, abs=1.0)


def test_hidden_variables_are_invisible_and_untouchable():
    world = garden()
    assert "sun" not in world.variables and "sun" not in world.sample()
    with pytest.raises(ValueError):
        world.sample(do={"sun": 1})
    assert world.sample(do={"rain": 1})["rain"] == 1


def test_pearson():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0


# ---------------------------------------------------------------- bridges --- #
def test_code_world_turns_execution_into_observation():
    world = CodeWorld()
    good = world.step("result = sum(range(5))")
    bad = world.step("result = [][0]")
    assert good.observation["value"] == 10 and good.reward == 1.0
    assert bad.observation["error"].startswith("IndexError") and bad.reward == 0.0
    assert len(world.history) == 2


def test_function_world_enforces_allowlist_budget_and_captures_failures():
    def lookup(city):
        if city == "Atlantis":
            raise LookupError("no such city")
        return {"city": city, "temp_c": 31}

    world = FunctionWorld({"weather": lookup}, budget=3)
    assert world.step({"tool": "weather", "args": {"city": "Patna"}}).observation["value"]["temp_c"] == 31
    assert "LookupError" in world.step({"tool": "weather", "args": {"city": "Atlantis"}}).observation["error"]
    assert "UnknownTool" in world.step({"tool": "rm_rf", "args": {}}).observation["error"]
    assert "BudgetExceeded" in world.step({"tool": "weather", "args": {"city": "Patna"}}).observation["error"]
