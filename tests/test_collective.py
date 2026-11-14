import json

import pytest

from agi.collective import (AuditLog, Builder, Collective, LLMAgent, Ledger, PolicyTampered, Review,
                            SafetyPolicy, SafetyReviewer, SpecReviewer, ToolForge, ToolNeed,
                            ToolProposal)
from agi.llm import ScriptedLLM
from agi.sandbox import SandboxPolicy

POLICY = SafetyPolicy(sandbox=SandboxPolicy(wall_time_s=2.0, cpu_time_s=1, memory_mb=256))
MEAN = "def mean(xs):\n    return sum(xs) / len(xs)\n"
MEAN_BUGGY = "def mean(xs):\n    return sum(xs) / (len(xs) - 1)\n"
MEAN_EVIL = "import os\ndef mean(xs):\n    os.system('id')\n    return sum(xs) / len(xs)\n"
MEAN_HANG = "def mean(xs):\n    while True:\n        pass\n"
VARIANCE = "def variance(xs):\n    m = mean(xs)\n    return sum((x - m) ** 2 for x in xs) / len(xs)\n"
SPREAD = "def spread(xs):\n    return variance(xs) ** 0.5\n"
NEED_MEAN = ToolNeed("mean", "arithmetic mean", ((([1, 2, 3],), 2.0), (([10],), 10.0), (([2, 3],), 2.5)))
NEED_VAR = ToolNeed("variance", "population variance", ((([1, 2, 3, 4],), 1.25), (([5, 5],), 0.0)))
NEED_SPREAD = ToolNeed("spread", "population standard deviation", ((([2, 4, 4, 4, 5, 5, 7, 9],), 2.0),))


class Vote:
    """A reviewer with a fixed opinion."""

    def __init__(self, agent_id, approve=True, **extra):
        self.id, self.approve, self.extra = agent_id, approve, extra

    def propose(self, need, context):
        return None

    def review(self, proposal, context):
        return Review(self.id, self.approve, ["because"], **self.extra)


@pytest.fixture(autouse=True)
def shared_policy_is_untouched():
    before = POLICY.fingerprint()
    yield
    assert POLICY.fingerprint() == before, "a test mutated the shared POLICY"


def team(*extra, policy=POLICY):
    return Collective([Builder("ada"), Builder("bob"), SpecReviewer("carol"), SafetyReviewer("sam"),
                       *extra], policy)


def propose(group, author, code, need=NEED_MEAN, tests=None, depends_on=(), reviewers=None):
    proposal = ToolProposal(need.name, need.description, code, tests or need.spec_tests(), author,
                            tuple(depends_on))
    return group.forge.submit(proposal, group.agents if reviewers is None else reviewers, need)


# ----------------------------------------------------------------- ledger --- #
def test_replicas_converge_and_namespaces_are_enforced():
    ledger = Ledger(["ada", "bob"])
    ledger.put("ada", "notes/ada/plan", {"step": 1})
    ledger.put("bob", "notes/bob/plan", {"step": 2})
    ledger.add("ada", "notes/ada/tags", "x")
    ledger.count("forge", "stats/accepted", 2)
    assert not ledger.converged()                       # nothing delivered yet
    assert ledger.sync() == 0 and ledger.converged()
    assert ledger.read("notes/ada/plan", at="bob") == {"step": 1}
    assert ledger.read("stats/accepted", at="ada") == 2
    assert ledger.read("notes/ada/tags", at="forge") == frozenset({"x"})
    assert ledger.keys("notes/", at="bob") == ["notes/ada/plan", "notes/ada/tags", "notes/bob/plan"]
    for agent, key in [("ada", "tools/x"), ("ada", "notes/bob/plan"), ("ada", "reviews/bob/p1"),
                       ("forge", "notes/ada/plan")]:
        with pytest.raises(PermissionError):
            ledger.put(agent, key, 1)
    with pytest.raises(ValueError):
        ledger.join("ada", ())


@pytest.mark.parametrize("seed", range(5))
def test_convergence_does_not_depend_on_delivery_order(seed):
    digests = set()
    for order_seed in (seed, seed + 100):
        ledger = Ledger(["a", "b", "c"], seed=order_seed)
        for i in range(6):
            agent = "abc"[i % 3]
            ledger.put(agent, f"notes/{agent}/k{i % 2}", i)
            ledger.count("forge", "stats/accepted")
            if i % 2:
                ledger.sync()
        ledger.sync()
        assert ledger.converged()
        digests.add(ledger.digest())
    assert len(digests) == 1


def test_rogue_envelopes_are_rejected_by_every_replica():
    ledger = Ledger(["ada", "mallory"])
    ledger.inject(ledger.craft("mallory", "tools/backdoor", {"code": "..."}))          # outside namespace
    ledger.inject(ledger.craft("ada", "reviews/ada/p1", {"approve": True}, signed_with="mallory"))
    ledger.inject(ledger.craft("forge", "tools/backdoor", {"code": "..."}, signed_with="mallory"))
    tampered = ledger.craft("mallory", "notes/mallory/x", 1)
    tampered["action"]["params"]["value"] = 2                                         # edit after signing
    ledger.inject(tampered)
    ledger.inject({"garbage": True})
    assert ledger.sync() == 5 and len(ledger.rejected) == 5
    assert all(ledger.keys("", at=agent) == [] for agent in ledger.replicas)
    reasons = " | ".join(r["reason"] for r in ledger.rejected)
    assert "signature verification failed" in reasons and "outside" in reasons
    ledger.inject(ledger.craft("mallory", "notes/mallory/x", 1))                      # a legitimate one
    assert ledger.sync() == 0 and ledger.read("notes/mallory/x", at="ada") == 1


# ------------------------------------------------------------------ gates --- #
def test_honest_tool_is_accepted_registered_and_callable_from_any_replica():
    group = team()
    verdict = propose(group, "bob", MEAN)
    assert verdict.accepted and verdict.version == 1 and verdict.stage == "accepted"
    assert verdict.approvals == ["ada", "carol", "sam"]                # the author is not a reviewer
    assert group.ledger.converged()
    for agent in ("ada", "carol", "sam", None):
        assert group.call("mean", [1, 2, 3, 4], agent=agent).value == 2.5
    record = group.tools()["mean"]
    assert record["author"] == "bob" and record["policy"] == POLICY.fingerprint()
    assert len(record["tests"]) >= 1 and group.ledger.read("stats/accepted") == 1
    assert group.ledger.read(f"verdicts/{verdict.proposal_id}", at="carol")["accepted"] is True


def test_gate_policy_blocks_unsafe_code_before_it_runs():
    group = team()
    verdict = propose(group, "ada", MEAN_EVIL)
    assert not verdict.accepted and verdict.stage == "policy"
    assert any("import of 'os'" in r for r in verdict.reasons)
    assert group.forge.sandbox.runs == 0 and group.tools() == {}


@pytest.mark.parametrize("name, code, tests, reason", [
    ("print", "def print(x):\n    return x\n", None, "not an acceptable tool name"),
    ("_hidden", "def _hidden(x):\n    return x\n", None, "not an acceptable tool name"),
    ("class", "x = 1\n", None, "not an acceptable tool name"),
    ("mean", "def average(xs):\n    return 0\n", None, "must define a top-level function 'mean'"),
    ("mean", MEAN, "x = 1\n", "at least one test_* function"),
    ("mean", MEAN, "import socket\ndef test_a():\n    pass\n", "tests: line 1"),
])
def test_gate_policy_structural_rules(name, code, tests, reason):
    group = team()
    need = ToolNeed(name, "x", ((([1],), 1),))
    verdict = propose(group, "ada", code, need, tests=tests)
    assert verdict.stage == "policy" and reason in " ".join(verdict.reasons)


def test_gate_author_tests():
    verdict = propose(team(), "ada", MEAN, tests="def test_a():\n    assert mean([1]) == 99\n")
    assert verdict.stage == "author-tests" and "TestFailed" in verdict.reasons[0]


def test_wrong_tool_with_self_serving_tests_is_caught_by_independent_review():
    group = team()
    verdict = propose(group, "ada", MEAN_BUGGY, tests="def test_a():\n    assert mean([1, 3]) == 4.0\n")
    assert not verdict.accepted and verdict.stage == "consensus"
    assert "bob" in verdict.rejections and "carol" in verdict.rejections
    assert "should be 2.0 but is 3.0" in " ".join(verdict.reasons)


def test_non_terminating_tool_is_vetoed():
    verdict = propose(team(), "ada", MEAN_HANG, tests="def test_a():\n    assert True\n")
    assert verdict.stage == "safety-veto" and verdict.reasons[0].startswith("VETO by sam")


def test_one_safety_veto_overrides_any_majority():
    group = team(Vote("v1"), Vote("v2"), Vote("v3"), Vote("doubter", False, safety_concern=True))
    verdict = propose(group, "bob", MEAN)
    assert verdict.stage == "safety-veto" and len(verdict.approvals) == 6
    assert verdict.rejections == ["doubter"]


def test_an_approval_backed_by_failing_tests_counts_as_a_rejection():
    liar = Vote("liar", True, tests="def test_impossible():\n    assert mean([1]) == 2\n")
    group = Collective([Builder("bob"), liar, Vote("v1", False)], POLICY)
    verdict = propose(group, "bob", MEAN)
    assert not verdict.accepted and verdict.approvals == [] and set(verdict.rejections) == {"liar", "v1"}


def test_authors_cannot_approve_themselves_or_pad_the_reviewer_list():
    group = Collective([Vote("bob"), Vote("v1")], POLICY)
    verdict = propose(group, "bob", MEAN, reviewers=[Vote("bob"), Vote("bob"), Vote("v1"), Vote("v1"),
                                                     Vote("stranger"), Vote("forge")])
    assert not verdict.accepted and verdict.stage == "consensus"
    assert "needs 2 independent reviewers, has 1" in verdict.reasons[0]


def test_quorum_arithmetic():
    def outcome(yes, no):
        voters = [Vote(f"y{i}") for i in range(yes)] + [Vote(f"n{i}", False) for i in range(no)]
        return propose(Collective([Builder("bob"), *voters], POLICY), "bob", MEAN).accepted
    assert outcome(2, 1) and outcome(4, 2)            # exactly two thirds
    assert not outcome(1, 1) and not outcome(3, 2)


def test_crashing_or_silent_reviewers_do_not_count_as_approval():
    class Crasher(Vote):
        def review(self, proposal, context):
            raise RuntimeError("boom")
    verdict = propose(Collective([Builder("bob"), Crasher("c1"), Vote("v1")], POLICY), "bob", MEAN)
    assert not verdict.accepted and "reviewer crashed" in " ".join(verdict.reasons)


# ------------------------------------------------- compounding and limits --- #
def test_tools_build_on_tools_within_a_depth_limit():
    group = team()
    assert propose(group, "bob", MEAN).accepted
    assert propose(group, "bob", VARIANCE, NEED_VAR, depends_on=("mean",)).accepted
    assert propose(group, "ada", SPREAD, NEED_SPREAD, depends_on=("variance",)).accepted
    assert group.call("spread", [2, 4, 4, 4, 5, 5, 7, 9]).value == 2.0
    assert group.forge.source("spread").index("def mean") < group.forge.source("spread").index("def spread")
    too_deep = ToolNeed("deeper", "x", ((([4, 4],), 0.0),))
    verdict = propose(group, "ada", "def deeper(xs):\n    return spread(xs)\n", too_deep,
                      depends_on=("spread",))
    assert verdict.stage == "policy" and "depth" in verdict.reasons[0]


def test_dependency_rules():
    group = team()
    assert propose(group, "bob", MEAN).accepted
    missing = propose(group, "bob", VARIANCE, NEED_VAR, depends_on=("median",))
    assert "unknown or revoked dependency 'median'" in missing.reasons[0]
    undeclared = propose(group, "bob", VARIANCE, NEED_VAR)             # uses mean without declaring it
    assert undeclared.stage == "author-tests" and "NameError" in undeclared.reasons[0]
    shadow = propose(group, "bob", MEAN + VARIANCE, NEED_VAR, depends_on=("mean",))
    assert "redefines its dependency" in shadow.reasons[0]
    selfish = propose(group, "bob", MEAN, depends_on=("mean",))
    assert "cannot depend on itself" in selfish.reasons[0]


def test_upgrades_must_not_regress():
    group = team()
    assert propose(group, "bob", MEAN).accepted
    rounded = "def mean(xs):\n    return round(sum(xs) / len(xs))\n"
    worse = propose(group, "ada", rounded, tests="def test_a():\n    assert mean([1, 3]) == 2\n")
    assert worse.stage == "regression" and "v1" in worse.reasons[0]
    assert group.tools()["mean"]["version"] == 1
    safer = "def mean(xs):\n    xs = list(xs)\n    return sum(xs) / len(xs)\n"
    better = propose(group, "ada", safer)
    assert better.accepted and better.version == 2 and group.tools()["mean"]["author"] == "ada"


def test_registry_capacity_is_a_hard_limit():
    policy = SafetyPolicy(sandbox=POLICY.sandbox, max_tools=1)
    group = team(policy=policy)
    assert propose(group, "bob", MEAN).accepted
    full = propose(group, "bob", "def other(xs):\n    return 1\n", ToolNeed("other", "x", ((([1],), 1),)))
    assert full.stage == "policy" and "registry is full" in full.reasons[0]


def test_revoking_a_tool_disables_it_and_its_dependents():
    group = team()
    propose(group, "bob", MEAN)
    propose(group, "bob", VARIANCE, NEED_VAR, depends_on=("mean",))
    group.forge.revoke("mean", "found to mishandle empty input")
    assert group.call("mean", [1]).error_type == "UnknownTool"
    assert group.call("variance", [1, 2], agent="ada").error_type == "IntegrityError"
    assert "mean" in group.forge.tools(include_revoked=True) and group.ledger.converged()
    with pytest.raises(KeyError):
        group.forge.revoke("mean", "again")


def test_approved_tools_still_run_sandboxed():
    group = team()
    sneaky = ("def mean(xs):\n    n = len(xs)\n    while n > 100:\n        n += 1\n"
              "    return sum(xs) / len(xs)\n")
    assert propose(group, "bob", sneaky).accepted           # passes review: the trigger is rare
    assert group.call("mean", [1, 2, 3]).value == 2.0
    outcome = group.call("mean", list(range(200)))          # ...but it cannot hang the host
    assert outcome.timed_out and not outcome.ok


def test_a_registry_record_that_fails_its_integrity_check_is_not_run():
    group = team()
    propose(group, "bob", MEAN)
    record = dict(group.tools()["mean"], code=MEAN_EVIL)     # simulate a corrupted replica
    group.ledger.replicas["ada"].register_set("tools/mean", record)
    assert group.call("mean", [1, 2], agent="ada").error_type == "IntegrityError"
    assert group.call("mean", [1, 2], agent="carol").value == 1.5


def test_policy_cannot_be_changed_after_construction():
    group = team(policy=SafetyPolicy(sandbox=POLICY.sandbox))     # private copy: this test mutates it
    with pytest.raises(Exception):
        group.forge.policy.quorum = 0.0                      # frozen dataclass
    object.__setattr__(group.forge.policy, "quorum", 0.0)    # force it anyway
    with pytest.raises(PolicyTampered):
        propose(group, "bob", MEAN)
    with pytest.raises(PolicyTampered):
        group.call("mean", [1])
    assert group.forge.audit.entries[-1]["event"] == "policy-tampered"


def test_tools_approved_under_another_policy_are_refused():
    group = team()
    propose(group, "bob", MEAN)
    stricter = SafetyPolicy(sandbox=POLICY.sandbox, quorum=1.0)
    other_forge = ToolForge(group.ledger, stricter)
    assert other_forge.call("mean", [1, 2]).error_type == "PolicyMismatch"


def test_audit_log_detects_tampering():
    group = team()
    propose(group, "ada", MEAN_EVIL)
    propose(group, "bob", MEAN)
    log = group.forge.audit
    assert log.verify() and [e["event"] for e in log.entries] == [
        "forge-started", "proposed", "rejected", "proposed", "accepted"]
    log.entries[2]["data"]["stage"] = "accepted"
    assert not log.verify()
    fresh = AuditLog()
    fresh.append("a")
    fresh.append("b")
    del fresh.entries[0]
    assert not fresh.verify()


# ----------------------------------------------------------- orchestration --- #
def test_collective_takes_the_first_proposal_that_survives():
    group = Collective([Builder("ada", {"mean": {"code": MEAN_BUGGY}}),
                        Builder("bob", {"mean": {"code": MEAN}}),
                        Builder("eve", {"mean": {"code": MEAN_EVIL}}),
                        SpecReviewer("carol"), SafetyReviewer("sam")], POLICY)
    verdicts = group.request(NEED_MEAN)
    assert [v.accepted for v in verdicts].count(True) == 1 and verdicts[-1].author == "bob"
    assert all(v.stage in ("policy", "author-tests", "consensus") for v in verdicts[:-1])
    assert group.request(ToolNeed("median", "x", ((([1],), 1),))) == []      # nobody has a draft
    with pytest.raises(ValueError):
        Collective([Builder("a"), Builder("a")])


def test_improve_builds_a_dependent_toolchain():
    drafts = {"mean": {"code": MEAN}, "variance": {"code": VARIANCE, "depends_on": ("mean",)},
              "spread": {"code": SPREAD, "depends_on": ("variance",)}}
    group = Collective([Builder("bob", drafts), SpecReviewer("carol"), SafetyReviewer("sam")], POLICY)
    results = group.improve([NEED_MEAN, NEED_VAR, NEED_SPREAD])
    assert all(v[-1].accepted for v in results.values())
    assert sorted(group.tools("sam")) == ["mean", "spread", "variance"]
    assert group.ledger.converged() and group.forge.audit.verify()


def test_spec_reviewer_needs_a_spec_and_safety_reviewer_flags_nondeterminism():
    group = team()
    proposal = ToolProposal("mean", "x", MEAN, NEED_MEAN.spec_tests(), "bob")
    assert not group.forge.submit(proposal, group.agents, need=None).accepted
    flaky = "import random\ndef mean(xs):\n    return sum(xs) / len(xs) + random.random()\n"
    verdict = propose(team(), "bob", flaky, tests="def test_a():\n    assert mean([1]) >= 1\n")
    assert "non-deterministic" in " ".join(verdict.reasons)


# -------------------------------------------------------------- llm agent --- #
def test_llm_agents_can_author_and_review():
    author = LLMAgent("gpt", ScriptedLLM(["```json\n" + json.dumps({
        "code": MEAN, "tests": "def test_a():\n    assert mean([2, 4]) == 3\n", "depends_on": []}) + "\n```"]))
    # the collective asks every agent for a draft first; the critic declines, then reviews
    critic = LLMAgent("critic", ScriptedLLM(["I have no draft for this.", json.dumps({
        "approve": True, "reasons": ["looks right"], "safety_concern": False,
        "tests": "def test_edge():\n    assert mean([5]) == 5\n"})]))
    group = Collective([author, critic, SpecReviewer("carol")], POLICY)
    (verdict,) = group.request(NEED_MEAN)
    assert verdict.accepted and verdict.approvals == ["critic", "carol"]
    assert "test_edge" in " ".join(group.tools()["mean"]["tests"])       # the critic's test is kept
    assert "define a top-level function named `mean`" in author.llm.prompts[0]
    assert "def mean(xs)" in critic.llm.prompts[1]


@pytest.mark.parametrize("reply", ["I approve!", json.dumps({"approve": "yes"}), json.dumps({"reasons": []})])
def test_llm_reviewer_fails_closed(reply):
    review = LLMAgent("critic", ScriptedLLM([reply])).review(
        ToolProposal("mean", "x", MEAN, "", "bob"), _context())
    assert review.approve is False


def test_llm_author_with_unusable_output_proposes_nothing():
    assert LLMAgent("gpt", ScriptedLLM(["no idea"])).propose(NEED_MEAN, _context()) is None


def _context():
    from agi.collective import ReviewContext
    return ReviewContext(need=NEED_MEAN, prelude="", policy=POLICY)
