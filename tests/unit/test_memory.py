import json

import pytest

from agi.application.memory.engine import MemoryEngine
from agi.application.memory.extract import HeuristicExtractor, LLMExtractor, error_signature
from agi.domain.entities.memory import FAILURE, SUCCESS, Episode, Lesson, Step
from agi.domain.services.embedding import keywords
from agi.infrastructure.adapters.llm.scripted import ScriptedLLM
from agi.infrastructure.adapters.persistence.knowledge_graph import KnowledgeGraph

CSV_FIX = Episode(
    "Parse the customer CSV export and compute the average order value", SUCCESS, tags=("csv",),
    solution="strip thousands separators before float()",
    steps=[Step("float(row['amount'])", ok=False,
                error="ValueError: could not convert string to float: '1,200.50'"),
           Step("float(row['amount'].replace(',', ''))", ok=True)])
CSV_FAIL = Episode(
    "Parse the vendor CSV file and total the invoice amounts", FAILURE, tags=("csv",),
    steps=[Step("float(row['total'])", ok=False,
                error="ValueError: could not convert string to float: '3,400.00'")])
API_FAIL = Episode(
    "Call the weather API for 200 cities", FAILURE, tags=("api",),
    steps=[Step("requests.get(url) in a tight loop", ok=False, error="HTTPError: 429 Too Many Requests")])


def fresh(*episodes, **kwargs):
    memory = MemoryEngine(**kwargs)
    for episode in episodes:
        memory.record(episode)
    return memory


def test_error_signature_strips_literals():
    assert error_signature("KeyError: 'user_42'") == ("KeyError", "'…'")
    assert error_signature("IndexError: index 7 out of range") == ("IndexError", "index N out of range")
    assert error_signature("something odd")[0] == "Error"
    assert error_signature("Timeout: exceeded 5s")[0] == "Timeout"
    assert error_signature(None) == ("Error", "")


def test_extractor_pairs_failure_with_what_worked():
    lessons = {l.kind: l for l in HeuristicExtractor().extract(CSV_FIX)}
    assert set(lessons) == {"fix", "procedure"}
    assert "worked instead" in lessons["fix"].advice and ".replace(" in lessons["fix"].advice
    assert lessons["fix"].payload["error_type"] == "ValueError"
    assert lessons["procedure"].payload["solution"] == CSV_FIX.solution


def test_extractor_unrecovered_failure_is_an_avoid_lesson():
    (lesson,) = HeuristicExtractor().extract(API_FAIL)
    assert lesson.kind == "avoid" and "HTTPError" in lesson.tags and "429" not in lesson.advice


def test_extractor_groups_a_noisy_trace():
    steps = [Step(f"attempt {i}", ok=False, error=f"KeyError: 'k{i}'") for i in range(50)]
    lessons = HeuristicExtractor().extract(Episode("look things up", FAILURE, steps=steps))
    assert len(lessons) == 1 and len(lessons[0].payload["failed"]) == 4


def test_episode_level_error_without_steps_still_teaches():
    (lesson,) = HeuristicExtractor().extract(Episode("deploy", FAILURE, error="Timeout: 30s"))
    assert lesson.kind == "avoid" and lesson.payload["error_type"] == "Timeout"
    assert HeuristicExtractor().extract(Episode("deploy", SUCCESS)) == []


def test_same_error_in_related_tasks_is_compressed_into_one_lesson():
    memory = fresh(CSV_FAIL, CSV_FIX)          # failure first, the fix arrives later
    errors = [l for l in memory.lessons() if l.kind in ("avoid", "fix")]
    assert len(errors) == 1
    assert errors[0].kind == "fix" and errors[0].support == 2      # upgraded: now knows what works
    assert len(errors[0].episodes) == 2


def test_recording_the_same_episode_twice_adds_support_not_duplicates():
    memory = fresh(API_FAIL, API_FAIL)
    assert memory.stats()["lessons"] == 1 and memory.lessons()[0].support == 2


def test_recall_finds_relevant_and_ignores_irrelevant():
    memory = fresh(CSV_FIX, CSV_FAIL, API_FAIL)
    hits = memory.recall("Load a CSV of payments and sum the amount column", tags=("csv",))
    assert hits and all("csv" in h.lesson.tags for h in hits)
    assert hits[0].lesson.kind == "fix" and hits[0].graph_score > 0
    assert memory.recall("write a haiku about autumn") == []
    api = memory.recall("poll the pricing API for every product", tags=("api",))
    assert [h.lesson.payload["error_type"] for h in api] == ["HTTPError"]
    assert MemoryEngine().recall("anything") == []


def test_recall_respects_k_and_kinds():
    memory = fresh(CSV_FIX, CSV_FAIL, API_FAIL)
    query = "Parse the customer CSV export and compute the average order value"
    assert len(memory.recall(query, k=1)) == 1
    assert {h.lesson.kind for h in memory.recall(query, kinds=["procedure"])} == {"procedure"}


def test_memory_persists_across_processes(tmp_path):
    path = str(tmp_path / "memory.db")
    with MemoryEngine(path) as memory:
        memory.record(CSV_FIX)
        before = sorted(l.id for l in memory.lessons())
    with MemoryEngine(path) as memory:           # a brand-new engine on the same file
        assert sorted(l.id for l in memory.lessons()) == before
        assert memory.recall("average order value of the CSV export", tags=("csv",))
        assert memory.stats()["episodes"] == 1


def test_feedback_changes_utility_and_ranking():
    memory = MemoryEngine()
    a = memory.learn(Lesson("fact", "resizing images for the gallery", "use the thumbnail cache"))
    b = memory.learn(Lesson("fact", "resizing images for the gallery page", "regenerate every time"))
    assert a.utility == b.utility == 0.5
    for _ in range(4):
        memory.feedback(b.id, helped=True)
        memory.feedback(a.id, helped=False)
    assert memory.get(b.id).utility > 0.8 > 0.2 > memory.get(a.id).utility
    assert memory.recall("resizing images for the gallery")[0].lesson.id == b.id
    memory.feedback("ls-does-not-exist", helped=True)      # unknown ids are ignored


def test_consolidate_retires_useless_lessons_without_deleting_them():
    memory = MemoryEngine()
    lesson = memory.learn(Lesson("fact", "tuning the cache size", "always double it"))
    for _ in range(5):
        memory.feedback(lesson.id, helped=False)
    assert memory.consolidate()["retired"] == 1
    assert memory.recall("tuning the cache size") == []
    assert memory.stats()["retired"] == 1 and memory.graph.get(lesson.id) is not None


def test_consolidate_compresses_old_traces_but_keeps_lessons():
    memory = fresh(CSV_FIX, CSV_FAIL, API_FAIL)
    lessons_before = memory.stats()["lessons"]
    assert memory.consolidate(keep_traces=1)["traces_compressed"] == 2
    kept = [e for e in memory.episodes() if e["steps"]]
    assert len(kept) == 1 and memory.stats()["lessons"] == lessons_before
    assert memory.consolidate(keep_traces=1)["traces_compressed"] == 0      # idempotent


def test_consolidate_merges_duplicates_and_rewires_the_graph():
    memory = MemoryEngine(merge_threshold=2.0)        # disable merge-on-write to create duplicates
    a = memory.learn(Lesson("fact", "rotating the api keys", "restart the gateway afterwards", ("ops",)))
    b = memory.learn(Lesson("fact", "rotating the api key", "restart the gateway afterwards", ("ops", "keys")))
    memory.feedback(b.id, helped=True)
    memory.merge_threshold = 0.8
    assert memory.consolidate()["merged"] == 1
    (merged,) = memory.lessons()
    assert merged.id == a.id and merged.support == 2 and merged.helped == 1 and "keys" in merged.tags
    assert memory.graph.get(b.id) is None
    assert any(other == "concept:keys" for other, _, _ in memory.graph.neighbors(a.id, "about"))


def test_forget_removes_a_lesson():
    memory = fresh(API_FAIL)
    memory.forget(memory.lessons()[0].id)
    assert memory.stats()["lessons"] == 0


def test_context_is_a_prompt_block_or_empty():
    memory = fresh(CSV_FIX)
    block = memory.context("average order value from the CSV export", tags=("csv",))
    assert block.startswith("Lessons from past experience") and "1. [" in block
    assert memory.context("compose a sonnet") == ""


def test_graph_links_spread_and_dot():
    graph = KnowledgeGraph()
    for name in "abc":
        graph.put(name, "concept", name)
    graph.link("a", "rel", "b", 1.0)
    graph.link("b", "rel", "c", 0.5)
    graph.link("a", "rel", "b", 1.0)                       # accumulates
    assert graph.edges("a") == [("a", "rel", "b", 2.0)]
    activation = graph.spread({"a": 1.0}, hops=2, decay=0.5)
    assert activation["b"] == pytest.approx(0.5) and activation["c"] == pytest.approx(0.125)
    assert '"a" -- "b"' in graph.to_dot(kinds=("concept",))
    graph.delete("b")
    assert graph.edges() == [] and graph.stats()["nodes"] == {"concept": 2}


def test_keywords_drop_stopwords_and_stem():
    assert keywords("Sorting the lists of numbers") == ["sort", "list", "number"]


def test_llm_extractor_uses_the_model_and_keeps_verified_solutions():
    reply = "Sure!\n```json\n" + json.dumps([{
        "kind": "fix", "trigger": "parsing money strings from spreadsheets",
        "advice": "normalise separators before converting", "tags": ["parsing"]}]) + "\n```"
    llm = ScriptedLLM([reply])
    lessons = LLMExtractor(llm).extract(CSV_FIX)
    assert [l.kind for l in lessons] == ["fix", "procedure"]
    assert lessons[0].tags == ("csv", "parsing") and "EPISODE" in llm.prompts[0]


@pytest.mark.parametrize("reply", ["I could not find any lessons.", "[{\"kind\": \"fix\"}]", "{not json"])
def test_llm_extractor_falls_back_on_unusable_output(reply):
    lessons = LLMExtractor(ScriptedLLM([reply])).extract(CSV_FIX)
    assert {l.kind for l in lessons} == {"fix", "procedure"}        # heuristic result


def test_llm_extractor_survives_a_crashing_model():
    def broken(prompt):
        raise ConnectionError("model unavailable")
    assert LLMExtractor(broken).extract(API_FAIL)[0].kind == "avoid"
