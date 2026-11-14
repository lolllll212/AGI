import pytest

from agi.core.sandbox import Sandbox
from agi.memory import (CycleError, EpisodicStore, HierarchicalMemory, ProceduralMemory, SemanticGraph,
                        WorkingMemory, align, estimate_tokens, extract_triples, make_index, render_answer)
from agi.memory import vector as vector_module


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


# ---------------------------------------------------------------- working --- #
def test_token_estimate_is_monotonic_and_plausible():
    sentence = "The quick brown fox jumps over the lazy dog."
    assert 9 <= estimate_tokens(sentence) <= 14
    assert estimate_tokens(sentence * 3) == 3 * estimate_tokens(sentence)
    assert estimate_tokens("") == 0 and estimate_tokens("internationalization") == 5


def fill(memory, turns):
    memory.add("system", "You are an agent. Always cite figures.", pinned=True)
    memory.add("user", "Task: reconcile the ledger for March.", pinned=True)
    for turn in range(turns):
        memory.add("assistant", f"Thought: check batch {turn}")
        memory.add("tool", f"batch {turn} total 4{turn}.50 " + "row data " * 60, tool="query_data", ok=True)


def test_context_stays_within_budget_and_keeps_pinned_and_recent():
    memory = WorkingMemory(budget=500, keep_recent=4)
    fill(memory, 20)
    assert memory.tokens <= 500
    contents = [m.content for m in memory.messages]
    assert contents[0].startswith("You are an agent") and contents[1].startswith("Task: reconcile")
    assert any("batch 19" in c for c in contents)                    # the latest work survives
    assert not any(c.startswith("batch 0 total") for c in contents)   # the oldest does not, verbatim
    stats = memory.stats()
    assert stats.compactions >= 1 and stats.summarised_messages > 0 and stats.tokens == memory.tokens


def test_compaction_summarises_before_it_drops():
    memory = WorkingMemory(budget=600, keep_recent=2, low_water=0.5)
    fill(memory, 6)
    summaries = [m for m in memory.messages if m.role == "summary"]
    assert len(summaries) == 1 and summaries[0].content.startswith("[Summary of")
    assert "query_data returned: batch" in summaries[0].content       # extractive: real content
    assert memory.as_chat()[2]["role"] == "system"                    # a summary reads as a system note


def test_model_written_summaries_and_fallback_when_the_model_fails():
    prompts = []

    def summarizer(prompt):
        prompts.append(prompt)
        return "Batches 0-3 reconciled; totals 40.50, 41.50, 42.50, 43.50."

    memory = WorkingMemory(budget=600, keep_recent=2, low_water=0.5, summarizer=summarizer)
    fill(memory, 6)
    assert prompts and "facts and figures found (exactly)" in prompts[0]
    assert any("Batches 0-3 reconciled" in m.content for m in memory.messages)

    def broken(prompt):
        raise TimeoutError("model down")

    fragile = WorkingMemory(budget=600, keep_recent=2, low_water=0.5, summarizer=broken)
    fill(fragile, 6)                                                  # must not raise
    assert fragile.tokens <= 600 and any(m.role == "summary" for m in fragile.messages)


def test_one_oversized_message_is_shrunk_not_lost():
    memory = WorkingMemory(budget=300, keep_recent=2, shrink_over=50)
    memory.add("system", "sys", pinned=True)
    memory.add("tool", "HEAD " + "x " * 3000 + " TAIL")
    (message,) = [m for m in memory.messages if not m.pinned]
    assert message.content.startswith("HEAD") and message.content.endswith("TAIL")
    assert "characters omitted" in message.content and memory.stats().shrunk == 1


def test_compaction_in_big_steps_keeps_the_prompt_prefix_stable():
    memory = WorkingMemory(budget=4000, keep_recent=4)
    memory.add("system", "You are an agent. " * 20, pinned=True)
    for turn in range(30):
        memory.add("user", f"step {turn}: " + "detail " * 30)
        memory.render()
    # Appending leaves the prefix alone, so most of every prompt repeats the previous one.
    assert memory.cache_hit_ratio > 0.8
    assert memory.stats().compactions <= 3                            # rare, large compactions
    memory.clear()
    assert [m.role for m in memory.messages] == ["system"]
    with pytest.raises(ValueError):
        WorkingMemory(low_water=0)


# --------------------------------------------------------------- episodic --- #
BACKENDS = ["python"] + (["faiss"] if vector_module._faiss is not None else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_similar_traces_are_found_and_unrelated_ones_are_not(backend):
    store = EpisodicStore(backend=backend)
    assert store.backend == backend
    store.add_trace("total sales for the North region", "success", [{"action": "query_data"}], "3090.75", tags=["sales"])
    store.add_trace("count ERROR lines in the server log", "failure", [{"action": "read_file"}], None, tags=["logs"])
    store.add("note", "the cat sat on the mat")
    hits = store.search("sales total in the north region")
    assert hits[0].record.data["task"] == "total sales for the North region" and hits[0].score > 0.5
    assert all("cat" not in hit.record.text for hit in hits)
    assert [h.record.data["outcome"] for h in store.search("server log errors", kind="trace")] == ["failure"]
    assert store.search("north region sales", tags=["logs"]) == []
    assert store.search("quantum chromodynamics lattice") == []


def test_backends_agree():
    if vector_module._faiss is None:
        pytest.skip("faiss is not installed")
    texts = [f"report {i} on {topic}" for i, topic in enumerate(
        ["sales", "weather", "sales forecast", "server errors", "regional sales totals", "holiday plans"])]
    ranked = []
    for backend in ("python", "faiss"):
        store = EpisodicStore(backend=backend)
        for text in texts:
            store.add("note", text)
        ranked.append([(h.record.text, round(h.score, 5)) for h in store.search("sales totals", k=4)])
    assert ranked[0] == ranked[1] and ranked[0][0][0] == "report 4 on regional sales totals"
    with pytest.raises(ValueError):
        make_index(8, "milvus")


@pytest.mark.parametrize("backend", BACKENDS)
def test_preferences_keep_one_current_value_and_their_history(backend):
    clock = Clock()
    store = EpisodicStore(backend=backend, clock=clock)
    store.set_preference("currency", "INR")
    clock.now += 60
    store.set_preference("currency", "USD")
    store.set_preference("tone", "brief")
    assert store.preferences() == {"currency": "USD", "tone": "brief"}
    assert [r.data["value"] for r in store.history("currency", "preference")] == ["INR", "USD"]
    found = store.search("currency", kind="preference")
    assert [h.record.data["value"] for h in found] == ["USD"]          # the old value is not searchable
    assert store.stats()["indexed"] == 2


def test_snapshots_answer_what_was_true_when():
    clock = Clock()
    store = EpisodicStore(clock=clock)
    store.snapshot("deploy", {"version": "1.0", "replicas": 2})
    clock.now = 2000.0
    store.snapshot("deploy", {"version": "1.1", "replicas": 3})
    assert store.state_at("deploy") == {"version": "1.1", "replicas": 3}
    assert store.state_at("deploy", when=1500.0) == {"version": "1.0", "replicas": 2}
    assert store.state_at("deploy", when=500.0) is None and store.state_at("other") is None
    assert len(store.history("deploy")) == 2


def test_recency_weighting_and_time_filters():
    clock = Clock(0.0)
    store = EpisodicStore(clock=clock)
    old = store.add("note", "deploy the billing service to production")
    clock.now = 10_000.0
    new = store.add("note", "deploy the billing service to production today")
    plain = store.search("deploy billing service production")
    assert plain[0].record.id == old.id                                # the closer match wins on similarity
    recent = store.search("deploy billing service production", half_life=1000.0)
    assert recent[0].record.id == new.id                               # unless recency counts
    assert [h.record.id for h in store.search("deploy billing", since=5000.0)] == [new.id]
    assert [r.id for r in store.recent(limit=1)] == [new.id]
    store.delete(new.id)
    assert store.get(new.id) is None and len(store.search("deploy billing")) == 1


def test_episodic_store_persists_and_reindexes(tmp_path):
    path = str(tmp_path / "episodic.db")
    first = EpisodicStore(path)
    first.add_trace("convert the March report to CSV", "success", [{"action": "write_file"}], "done")
    first.set_preference("format", "csv")
    first.close()
    second = EpisodicStore(path)
    assert second.search("march report csv")[0].record.kind == "trace"
    assert second.preferences() == {"format": "csv"}
    with pytest.raises(ValueError, match="dimensions"):
        second.embed = lambda text: [1.0, 0.0]
        second.add("note", "x")


def test_a_neural_embedder_can_be_plugged_in():
    synonyms = {"car": 0, "automobile": 0, "vehicle": 0, "banana": 1, "fruit": 1}

    def embed(text):
        vector = [0.0, 0.0, 0.01]
        for word in text.lower().split():
            if word in synonyms:
                vector[synonyms[word]] += 1.0
        return vector

    store = EpisodicStore(embed=embed)
    store.add("note", "the automobile needs a service")
    store.add("note", "a banana is a fruit")
    assert store.dim == 3
    assert "automobile" in store.search("car")[0].record.text          # no shared word: meaning only


# --------------------------------------------------------------- semantic --- #
def test_triples_are_extracted_conservatively():
    text = ("The API depends on the database and the cache. Redis is a cache. Delhi is the capital of India. "
            "Rain causes wet roads. The engine is part of the car. Who knows what this means")
    assert extract_triples(text) == [
        ("API", "depends_on", "database"), ("API", "depends_on", "cache"), ("Redis", "is_a", "cache"),
        ("Delhi", "capital_of", "India"), ("Rain", "causes", "wet roads"), ("engine", "part_of", "car")]
    assert extract_triples("The car contains an engine.") == [("engine", "part_of", "car")]


@pytest.fixture
def stack():
    graph = SemanticGraph()
    graph.add_text("api depends on database and cache. database depends on disk. cache depends on network. "
                   "worker depends on database.")
    return graph


def test_dependency_queries(stack):
    assert stack.dependencies("API") == ["database", "cache", "disk", "network"]
    assert stack.dependents("disk") == ["database", "api", "worker"]
    order = stack.build_order(["api"])
    assert order[-1] == "api" and order.index("disk") < order.index("database") < order.index("api")
    assert set(stack.build_order()) == {"api", "database", "cache", "disk", "network", "worker"}
    assert [str(f) for f in stack.path("disk", "cache")] == [
        "database --depends_on--> disk", "api --depends_on--> database", "api --depends_on--> cache"]
    assert stack.path("disk", "nowhere") is None and stack.path("disk", "disk") == []


def test_cycles_are_found_and_block_ordering(stack):
    assert stack.find_cycle() is None
    stack.add_fact("disk", "depends_on", "api")
    cycle = stack.find_cycle()
    assert cycle[0] == cycle[-1] and {"api", "database", "disk"} <= set(cycle)
    with pytest.raises(CycleError, match="dependency cycle"):
        stack.build_order()


def test_functional_predicates_replace_and_keep_history():
    clock = Clock()
    graph = SemanticGraph(functional=["version"], clock=clock)
    graph.add_fact("billing", "version", "1.0")
    clock.now = 2000.0
    graph.add_fact("billing", "version", "1.1")
    graph.add_fact("billing", "tag", "payments")
    graph.add_fact("billing", "tag", "critical")                      # not functional: both hold
    assert graph.ask("billing", "version") == ["1.1"]
    assert graph.ask("billing", "version", at=1500.0) == ["1.0"]
    assert sorted(graph.ask("billing", "tag")) == ["critical", "payments"]
    assert [(f.object, f.valid_to) for f in graph.history("billing", "version")] == [("1.0", 2000.0), ("1.1", None)]
    assert graph.retract("billing", "tag", "critical") == 1 and graph.ask("billing", "tag") == ["payments"]
    assert graph.stats()["historical_facts"] == 2


def test_confirmation_from_independent_sources_raises_confidence():
    graph = SemanticGraph()
    graph.add_fact("Pluto", "is_a", "planet", confidence=0.6, source="old textbook")
    again = graph.add_fact("pluto", "is_a", "Planet", confidence=0.5, source="blog")
    assert again.confidence == pytest.approx(0.8) and again.source == "old textbook; blog"
    same = graph.add_fact("Pluto", "is_a", "planet", confidence=0.5, source="blog")
    assert same.confidence == pytest.approx(0.8)                      # the same source again adds nothing
    assert len(graph.facts("pluto")) == 1 and graph.facts(min_confidence=0.9) == []
    assert graph.entity("PLUTO").name == "Pluto"                      # first spelling is the display name


def test_rules_infer_facts_without_storing_them():
    graph = SemanticGraph()
    graph.add_text("A poodle is a dog. A dog is a mammal. A mammal is an animal.")
    graph.add_fact("mammal", "has", "fur", confidence=0.9)
    graph.add_fact("alice", "married_to", "bob")
    graph.add_fact("api", "depends_on", "db")
    assert graph.ask("poodle", "is_a") == ["dog"]
    graph.add_rule("transitive", "is_a")
    graph.add_rule("inherits", "is_a", "has")
    graph.add_rule("symmetric", "married_to")
    graph.add_rule("inverse", "depends_on", "required_by")
    assert graph.ask("poodle", "is_a") == ["dog", "mammal", "animal"]
    assert graph.holds("poodle", "has", "fur") and graph.holds("bob", "married_to", "alice")
    assert graph.ask("db", "required_by") == ["api"]
    inferred = graph.facts("poodle", "has")[0]
    assert inferred.inferred and inferred.confidence == pytest.approx(0.8 * 0.8 * 0.9)   # weakest-chain product
    assert graph.facts("poodle", "has", include_inferred=False) == []
    assert graph.stats()["facts"] == 6                                # nothing inferred was written
    with pytest.raises(ValueError):
        graph.add_rule("telepathic", "x")
    with pytest.raises(ValueError):
        graph.add_rule("inverse", "x")


def test_entities_descriptions_and_relevance(stack):
    stack.add_entity("api", type="service", owner="platform")
    stack.add_entity("API", port=8080)
    entity = stack.entity("api")
    assert entity.type == "service" and entity.attributes == {"owner": "platform", "port": 8080}
    assert [e.id for e in stack.entities("service")] == ["api"]
    text = stack.describe("api")
    assert text.startswith("api (service)") and "- api depends on database (confidence 0.80)" in text
    assert stack.describe("ghost") == "Nothing is known about 'ghost'."
    relevant = stack.relevant("Why is the worker slow when the disk is full?")
    assert {str(f) for f in relevant} == {"worker --depends_on--> database", "database --depends_on--> disk"}
    assert stack.relevant("unrelated question about apis") == []      # whole words only
    with pytest.raises(ValueError):
        stack.add_entity("  ")


def test_graph_persists_and_exports_to_networkx(tmp_path):
    path = str(tmp_path / "semantic.db")
    graph = SemanticGraph(path)
    graph.add_text("api depends on db. db depends on disk.")
    graph.close()
    reopened = SemanticGraph(path)
    assert reopened.dependencies("api") == ["db", "disk"]
    assert reopened.central(1)[0][0] == "db"                          # the most connected node
    networkx = pytest.importorskip("networkx")
    exported = reopened.to_networkx()
    assert isinstance(exported, networkx.MultiDiGraph)
    assert set(exported.nodes) == {"api", "db", "disk"} and exported.has_edge("api", "db", key="depends_on")
    assert networkx.has_path(exported, "api", "disk")


# ------------------------------------------------------------- procedural --- #
NORTH = "What is the total sales for the North region in sales.csv?"
CALL = {"tool": "query_data", "args": {"path": "sales.csv", "sql": "select sum(amount) from data where region='North'"}}


def test_alignment_finds_swapped_values_only():
    assert align(NORTH, "what is the total sales for the South region in sales.csv?") == {"North": "South"}
    assert align(NORTH, NORTH) == {}
    assert align("sum 12 and 30", "sum 7 and 1,200") == {"12": "7", "30": "1,200"}
    assert align("total for North", "total for North and South") is None        # wording added
    assert align("copy a to b then a to c", "copy x to b then y to c") is None  # inconsistent swap


def test_trajectory_is_adapted_only_when_the_change_is_carried_through():
    memory = ProceduralMemory()
    memory.add_trajectory(NORTH, [CALL], answer="The North total is 3,090.75.", last_output="3090.75")
    adapted = memory.adapt("What is the total sales for the South region in sales.csv?")
    assert adapted.calls[0].args["sql"] == "select sum(amount) from data where region='South'"
    assert adapted.answer_template == "The South total is {#0:,.2f}." and adapted.substitutions == {"North": "South"}
    assert adapted.output_figures == 1
    assert 0.6 < adapted.confidence < 0.8 and not adapted.exact
    assert memory.adapt(NORTH).exact
    # "average" replaces "total", but nothing in the stored call says "total": not safely adaptable.
    assert memory.adapt("What is the average sales for the North region in sales.csv?") is None
    assert memory.adapt("Summarise the weather in Patna") is None
    assert memory.adapt("What is the total sales for the South region in sales.csv, and who sold most?") is None


def test_numbers_are_carried_into_arguments_with_their_types():
    memory = ProceduralMemory()
    memory.add_trajectory("What is 15% of 2,400?", [{"tool": "calculator", "args": {"expression": "2400 * 15 / 100"}},
                                                    {"tool": "repeat", "args": {"times": 15}}])
    adapted = memory.adapt("What is 15% of 3,600?")                     # "2,400" in the task is 2400 in the call
    assert adapted.calls[0].args == {"expression": "3600 * 15 / 100"} and adapted.calls[1].args == {"times": 15}
    adapted = memory.adapt("What is 20% of 2,400?")
    assert adapted.calls[0].args == {"expression": "2400 * 20 / 100"} and adapted.calls[1].args == {"times": 20}
    assert memory.adapt("What is 15% of 2,400 plus 3?") is None


def test_unreliable_trajectories_stop_being_offered():
    memory = ProceduralMemory(min_reliability=0.5)
    stored = memory.add_trajectory(NORTH, [CALL])
    assert stored.reliability == 0.75
    memory.record_use(stored.id, True)
    assert memory.get(stored.id).reliability == pytest.approx(2 / 3)
    for _ in range(3):
        memory.record_use(stored.id, False)
    assert memory.get(stored.id).reliability == pytest.approx(2 / 6)
    assert memory.adapt(NORTH) is None and memory.stats()["retired"] == 1
    assert memory.record_use("ghost", True) is None
    again = memory.add_trajectory(NORTH, [CALL])                       # re-learning refreshes, no duplicate
    assert again.id == stored.id and memory.stats()["trajectories"] == 1
    assert memory.add_trajectory("nothing done", []) is None


def test_answer_templates_are_only_kept_when_unambiguous():
    template = ProceduralMemory._template
    assert template("The North total is 3,090.75.", "round(sum(amount), 2)\n3090.75") == "The North total is {#1:,.2f}."
    assert template("It is East.", "East") == "It is {result}."
    assert template("Total 10.00, or 10.00 per head.", "10.00") == "Total {#0:.2f}, or {#0:.2f} per head."
    assert template("Totals 5 and 7", "5 | 7") == "Totals {#0:.0f} and {#1:.0f}"
    assert template("15% of 2400 is 360", "360", task="What is 15% of 2400?") == "15% of 2400 is {#0:.0f}"
    assert template("About three thousand", "3090.75") is None         # the figure is not in the answer
    assert template("It is 3090.75, up 41.5 on last year", "3090.75") is None      # 41.5 comes from nowhere
    assert template("Roughly 3.1", "3.05 | 3.14") is None              # two different figures round to it
    assert template("x", "y" * 3000) is None


def test_templates_are_refilled_from_fresh_output():
    assert render_answer("The South total is {#1:,.2f}.", "round(sum(amount), 2)\n2491.15", 2) == "The South total is 2,491.15."
    assert render_answer("It is {result}.", " West \n") == "It is West."
    assert render_answer("Totals {#0:.0f} and {#1:.0f}", "5 | 7 | 9", expected_figures=2) is None    # shape changed
    assert render_answer("Value {#3:.1f}", "1 2") is None
    assert render_answer("Set {{a}} has {#0:.0f}", "4") == "Set {a} has 4"


def test_snippets_must_pass_their_tests_in_the_sandbox():
    memory = ProceduralMemory(sandbox=Sandbox())
    stored = memory.add_snippet("mean", "def mean(xs):\n    return sum(xs) / len(xs)",
                                "arithmetic mean (average) of a list of numbers", ["assert mean([1, 2, 3]) == 2"],
                                tags=["stats"])
    assert stored.kind == "snippet" and memory.get("mean").code.startswith("def mean")
    with pytest.raises(ValueError, match="failed verification: AssertionError"):
        memory.add_snippet("bad", "def f(x):\n    return x + 1", "increment", ["assert f(1) == 3"])
    with pytest.raises(ValueError, match="failed verification"):
        memory.add_snippet("evil", "import os\nos.system('id')", "x", ["assert True"])
    with pytest.raises(ValueError, match="at least one test"):
        memory.add_snippet("untested", "x = 1", "x", [])
    with pytest.raises(RuntimeError, match="sandbox is required"):
        ProceduralMemory().add_snippet("m", "x = 1", "x", ["assert x == 1"])
    assert [m.procedure.name for m in memory.search("average of a list of numbers")] == ["mean"]
    assert memory.search("deploy kubernetes cluster") == []
    memory.forget("mean")
    assert memory.stats()["snippets"] == 0


def test_workflows_are_parameterised(tmp_path):
    path = str(tmp_path / "procedures.db")
    memory = ProceduralMemory(path)
    memory.add_workflow("monthly_report", "summarise a month's sales into a report file", [
        {"tool": "query_data", "args": {"path": "{file}", "sql": "select sum(amount) from data where month={month}"}},
        {"tool": "write_file", "args": {"path": "report_{month}.txt", "content": "{note}", "limit": "{limit}"}}])
    memory.close()
    memory = ProceduralMemory(path)
    assert memory.get("monthly_report").params == ["file", "limit", "month", "note"]
    calls = memory.instantiate("monthly_report", file="sales.csv", month=3, note="March", limit=10)
    assert calls[0].args == {"path": "sales.csv", "sql": "select sum(amount) from data where month=3"}
    assert calls[1].args == {"path": "report_3.txt", "content": "March", "limit": 10}    # lone placeholder keeps its type
    with pytest.raises(ValueError, match="needs values for: limit, note"):
        memory.instantiate("monthly_report", file="x", month=1)
    with pytest.raises(KeyError):
        memory.instantiate("nope")


# -------------------------------------------------------------- hierarchy --- #
def test_hierarchy_files_runs_and_builds_context(tmp_path):
    steps = [{"action": "query_data", "args": CALL["args"], "ok": True, "observation": "3090.75"}]
    with HierarchicalMemory(str(tmp_path / "mem"), sandbox=Sandbox()) as memory:
        memory.episodic.set_preference("currency", "INR")
        memory.semantic.add_text("sales.csv is a data file. North is a region.")
        memory.procedural.add_snippet("pct", "def pct(a, b):\n    return 100 * a / b", "percentage of a total",
                                      ["assert pct(1, 4) == 25"])
        memory.remember_run(NORTH, True, steps, "The North total is 3,090.75.")
        memory.remember_run("What is the total for Mars?", False,
                            [{"action": "query_data", "args": {}, "ok": False, "observation": "no rows"}], None)
        context = memory.context_for("What is the total sales for the South region in sales.csv, as a percentage?")
        assert "USER PREFERENCES\n- currency: INR" in context
        assert "[success] What is the total sales for the North region" in context and "(used: query_data)" in context
        assert "KNOWN FACTS" in context and "sales.csv is a data file" in context
        assert "VERIFIED CODE YOU CAN REUSE" in context and "def pct" in context
        assert memory.context_for("xyzzy") == "USER PREFERENCES\n- currency: INR"
        stats = memory.stats()
        assert stats["procedural"]["trajectories"] == 1               # the failed run is not a procedure
        assert stats["episodic"]["records"]["trace"] == 2
    with HierarchicalMemory(str(tmp_path / "mem")) as reopened:
        assert reopened.procedural.adapt("What is the total sales for the East region in sales.csv?") is not None
        assert reopened.episodic.preferences() == {"currency": "INR"}


def test_adapting_a_template_leaves_its_placeholders_alone():
    memory = ProceduralMemory()
    memory.add_trajectory("Count from 0 to 40", [{"tool": "count", "args": {"start": 0, "stop": 40}}],
                          answer="From 0 to 40 there are 41 numbers", last_output="41")
    adapted = memory.adapt("Count from 2 to 50")
    assert adapted.calls[0].args == {"start": 2, "stop": 50}
    assert adapted.answer_template == "From 2 to 50 there are {#0:.0f} numbers"     # "{#0" is not a task value
    assert render_answer(adapted.answer_template, "49", adapted.output_figures) == "From 2 to 50 there are 49 numbers"


def test_a_figure_that_could_be_task_or_output_is_never_templated():
    # "40" is both an operand and the result: replaying this would misreport the operand.
    assert ProceduralMemory._template("0 plus 40 is 40", "40", task="Add 0 to 40") is None
    memory = ProceduralMemory()
    stored = memory.add_trajectory("Add 0 to 40", [{"tool": "calculator", "args": {"expression": "0 + 40"}}],
                                   answer="0 plus 40 is 40", last_output="40")
    assert stored.answer_template is None and memory.adapt("Add 2 to 50").answer_template is None
