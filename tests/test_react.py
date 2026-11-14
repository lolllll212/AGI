import json

import pytest

from agi.llm import ScriptedLLM
from agi.memory import MemoryEngine
from agi.react import (Conversation, GroundingVerifier, LLMVerifier, ReActAgent, ReActStep,
                       numbers_in, parse_reply)
from agi.tools import Param, Tool, ToolRegistry, calculator_tool, standard_tools


def act(tool, args, thought="next"):
    return f"Thought: {thought}\nAction: {tool}\nAction Input: {json.dumps(args)}"


def final(answer, thought="done"):
    return f"Thought: {thought}\nFinal Answer: {answer}"


def calc(*replies, **options):
    llm = ScriptedLLM(list(replies))
    return ReActAgent(llm, ToolRegistry([calculator_tool()]), **options), llm


# ---------------------------------------------------------------- parsing --- #
def test_parse_action_and_final_answer():
    parsed = parse_reply('Thought: add them\nAction: calculator\nAction Input: {"expression": "1+1"}')
    assert (parsed.thought, parsed.action, parsed.args) == ("add them", "calculator", {"expression": "1+1"})
    done = parse_reply("Thought: easy\nFinal Answer: 2\nbecause 1+1")
    assert done.answer == "2\nbecause 1+1" and done.action is None and done.thought == "easy"


@pytest.mark.parametrize("reply", [
    'Action: calculator\nAction Input: {"expression": "1+1"}',                       # no thought
    'thought: x\naction: calculator\naction input: {"expression": "1+1"}',           # lower case
    '**Thought:** x\n**Action:** `calculator`\n**Action Input:** {"expression": "1+1"}',  # markdown
    'Thought: x\nAction: calculator\nAction Input:\n```json\n{"expression": "1+1"}\n```',  # fenced
    'Thought: x\nAction: calculator(expression)\nAction Input: {"expression": "1+1"}',
    'Thought: x\nAction: calculator\nAction Input: 1+1',                             # bare value
    'Thought: x\nAction: calculator\nAction Input: "1+1"',
    'Thought: x\nAction: calculator\nAction_Input: {"expression": "1+1"}',
    '{"thought": "x", "action": "calculator", "action_input": {"expression": "1+1"}}',  # pure JSON
])
def test_parser_is_lenient_about_form(reply):
    registry = ToolRegistry([calculator_tool()])
    parsed = parse_reply(reply, registry)
    assert parsed.problem is None and parsed.action == "calculator"
    assert parsed.args == {"expression": "1+1"}


def test_anything_after_the_action_is_discarded():
    reply = ('Thought: x\nAction: calculator\nAction Input: {"expression": "2*21"}\n'
             'Observation: 99\nThought: so it is 99\nFinal Answer: 99')
    parsed = parse_reply(reply)
    assert parsed.action == "calculator" and parsed.answer is None and parsed.args == {"expression": "2*21"}
    no_marker = parse_reply('Thought: x\nAction: calculator\nAction Input: {"expression": "2*21"}\nFinal Answer: 99')
    assert no_marker.action == "calculator" and no_marker.answer is None


def test_parser_reports_unusable_replies():
    assert "neither an Action nor a Final Answer" in parse_reply("I think the answer is 4.").problem
    assert "empty" in parse_reply("Thought: x\nFinal Answer:").problem
    two_params = ToolRegistry([Tool("f", "x", lambda a, b: 0, {"a": Param(), "b": Param()})])
    assert "JSON object" in parse_reply("Action: f\nAction Input: hello", two_params).problem
    assert parse_reply("Action: now\nAction Input: none").args == {}
    assert parse_reply('{"final_answer": "42"}').answer == "42"


def test_numbers_in_text():
    assert numbers_in("Total 1,234.50 for 3 items, -7 and 12% in 2024; v2 and a1 ignored") == [
        (1234.5, 2), (3.0, 0), (-7.0, 0), (12.0, 0), (2024.0, 0)]


# --------------------------------------------------------------- the loop --- #
def test_a_simple_run():
    agent, llm = calc(act("calculator", {"expression": "17 * 23"}), final("17 times 23 is 391."))
    run = agent.run("What is 17 times 23?")
    assert run.finished and run.answer == "17 times 23 is 391." and run.grounded is True
    assert run.success and run.tool_calls == 1 and run.llm_calls == 2 and run.tool_errors == 0
    assert "Observation: 391" in llm.prompts[1] and llm.prompts[1].endswith("Thought:")
    assert "calculator(expression: string)" in llm.prompts[0] and "Task: What is 17 times 23?" in llm.prompts[0]
    assert "answered" in str(run) and "Final Answer: 17 times 23 is 391." in run.transcript()


def test_the_model_cannot_invent_an_observation():
    reply = act("calculator", {"expression": "17 * 23"}) + "\nObservation: 400\nFinal Answer: 400"
    agent, llm = calc(reply, final("391"))
    run = agent.run("What is 17 times 23?")
    assert run.answer == "391" and run.steps[0].observation == "391"
    assert "Observation: 400" not in llm.prompts[1]


def test_tool_errors_are_fed_back_and_can_be_recovered_from():
    agent, llm = calc(act("calculator", {"expression": "2 ^ 10"}),
                      act("calculator", {"expression": "2 ** 10"}), final("1024"))
    run = agent.run("What is 2 to the power 10?")
    assert run.answer == "1024" and run.tool_errors == 1 and [s.ok for s in run.steps] == [False, True]
    assert "write ** instead" in llm.prompts[1]


def test_unknown_tools_and_bad_arguments_are_explained():
    agent, llm = calc(act("wolfram", {"q": "1+1"}), act("calculator", {"expr": "1+1"}),
                      act("calculator", {"expression": "1+1"}), final("It is 2."))
    run = agent.run("1+1?")
    assert run.finished and run.tool_errors == 2
    assert "Available tools: calculator" in llm.prompts[1] and "unexpected argument(s) 'expr'" in llm.prompts[2]


def test_format_errors_get_a_reminder_then_the_run_stops():
    agent, llm = calc("The answer is probably 4.", act("calculator", {"expression": "2+2"}), final("4"))
    run = agent.run("2+2?")
    assert run.finished and "FormatError" in llm.prompts[1] and run.steps[0].action is None
    stuck, _ = calc("hmm", "well", "let me think")
    run = stuck.run("2+2?")
    assert not run.finished and run.stop_reason == "format" and run.answer is None and not run.success


def test_repeating_a_call_is_called_out_and_bounded():
    again = act("calculator", {"expression": "2+2"})
    agent, llm = calc(again, again, final("4"))
    run = agent.run("2+2?")
    assert run.finished and "RepeatedCall" in llm.prompts[2] and len(agent.tools.log) == 1
    looping = ReActAgent(ScriptedLLM(lambda p: final("It is 4.") if "no steps left" in p else again),
                         ToolRegistry([calculator_tool()]))
    run = looping.run("2+2?")
    assert run.stop_reason == "loop" and run.answer == "It is 4." and not run.finished
    assert len(looping.tools.log) == 1


def test_step_limit_forces_a_closing_answer():
    counter = iter(range(100))
    llm = ScriptedLLM(lambda p: final("I could not finish; partial result 5.") if "no steps left" in p
                      else act("calculator", {"expression": f"{next(counter)} + 1"}))
    run = ReActAgent(llm, ToolRegistry([calculator_tool()]), max_steps=5).run("count forever")
    assert run.stop_reason == "max_steps" and run.tool_calls == 5 and run.llm_calls == 6
    assert run.answer.startswith("I could not finish") and not run.finished and not run.success


def test_model_endpoint_failure_is_reported_not_raised():
    def broken(prompt):
        raise ConnectionError("endpoint down")
    run = ReActAgent(broken, ToolRegistry([calculator_tool()])).run("2+2?")
    assert run.stop_reason == "llm_error" and "endpoint down" in run.error and run.answer is None


def test_on_step_callback_sees_every_step():
    seen = []
    agent, _ = calc(act("calculator", {"expression": "1+1"}), final("2"), on_step=seen.append)
    agent.run("1+1?")
    assert [s.action for s in seen] == ["calculator"] and isinstance(seen[0], ReActStep)


def test_old_observations_are_shortened_to_fit_the_context():
    long_tool = Tool("dump", "returns a lot", lambda n: "x" * 3000 + f" END{n}",
                     {"n": Param("integer")})
    replies = [act("dump", {"n": i}) for i in range(6)] + [final("done")]
    llm = ScriptedLLM(replies)
    agent = ReActAgent(llm, ToolRegistry([long_tool]), max_context_chars=9000, verifier=None)
    assert agent.run("dump things").finished
    last = llm.prompts[-1]
    assert len(last) < 14_000 and "older output shortened" in last
    assert "END5" in last and "END4" in last and "END3" in last and "END0" not in last


# ----------------------------------------------------------- verification --- #
def test_unsupported_figures_are_challenged_once():
    agent, llm = calc(act("calculator", {"expression": "1200 * 1.07"}),
                      final("The total is 1,290."), final("The total is 1,284."))
    run = agent.run("Add 7% tax to 1200.")
    assert run.answer == "The total is 1,284." and run.challenges == 1 and run.grounded is True
    assert "[verifier] The figure(s) 1290" in llm.prompts[2]


def test_a_model_that_insists_is_reported_as_not_grounded():
    agent, _ = calc(final("Revenue was 48,213.77."), final("Revenue was 48,213.77."))
    run = agent.run("What was revenue last quarter?")
    assert run.finished and run.grounded is False and run.success is False and run.challenges == 1
    assert "NOT grounded" in str(run)


def test_grounding_rules():
    verifier = GroundingVerifier()
    step = lambda text, ok=True: ReActStep("", "tool", {}, text, ok)
    supported = lambda task, answer, *obs: verifier.unsupported(task, answer, [step(o) for o in obs])
    assert supported("", "It is 1,234.57", "1234.5678") == []                 # rounding
    assert supported("", "about 1235", "1234.5678") == []
    assert supported("", "a share of 12.3%", "0.1234") == []                  # ratio as percent
    assert supported("Add 7% to 1200", "1200 plus 7% is 1284", "1284") == []  # numbers from the task
    assert supported("", "there are 3 files and 2 errors") == []              # small counts
    assert supported("", "It is 1,300", "1234.5678") == ["1300"]
    assert supported("", "It is 99.5 and 42", "42") == ["99.5"]
    assert verifier.unsupported("", "It is 500", [step("500", ok=False)]) == ["500"]   # errors are not evidence
    assert verifier("", "nothing numeric here", []) is None


def test_verifier_can_be_disabled_or_replaced():
    agent, _ = calc(final("It is 987,654."), verifier=None)
    run = agent.run("guess")
    assert run.finished and run.grounded is None and run.success and "unverified" in str(run)
    strict = lambda task, answer, steps: None if "please" in answer else "say please"
    agent, llm = calc(final("4"), final("4, please"), verifier=strict)
    assert agent.run("2+2?").answer == "4, please" and "[verifier] say please" in llm.prompts[1]


def test_llm_verifier():
    steps = [ReActStep("", "calculator", {}, "391", True)]
    critic = ScriptedLLM(['{"supported": false, "problem": "the tool said 391"}', '{"supported": true}', "garbage"])
    verifier = LLMVerifier(critic)
    assert "the tool said 391" in verifier("17*23?", "400", steps)
    assert verifier("17*23?", "391", steps) is None
    assert verifier("17*23?", "391", steps) is None            # advisory: unusable review is no objection
    assert "[calculator] 391" in critic.prompts[0]


# ----------------------------------------------------------------- memory --- #
def test_runs_are_remembered_and_lessons_reach_the_next_prompt(tmp_path):
    (tmp_path / "sales.csv").write_text("region,amount\nNorth,100\nSouth,250\n")
    path = str(tmp_path / "memory.db")
    sql = "select sum(amount) from data where region = '{}'"
    with MemoryEngine(path) as memory:
        llm = ScriptedLLM([act("query_data", {"file": "sale.csv", "sql": sql.format("North")}),
                           act("query_data", {"file": "sales.csv", "sql": sql.format("North")}),
                           final("North sold 100.")])
        run = ReActAgent(llm, standard_tools(tmp_path), memory=memory, tags=("sales",)).run(
            "What is the total sales amount for the North region in sales.csv?")
        assert run.success and run.recalled == [] and {l.kind for l in run.learned} == {"fix", "procedure"}
        fix = next(l for l in run.learned if l.kind == "fix")
        assert fix.payload["error_type"] == "FileNotFound" and "sales.csv" in fix.payload["worked"]
        assert "LESSONS" not in llm.prompts[0]

    with MemoryEngine(path) as memory:                       # a new session, a similar task
        llm = ScriptedLLM([act("query_data", {"file": "sales.csv", "sql": sql.format("South")}),
                           final("South sold 250.")])
        run = ReActAgent(llm, standard_tools(tmp_path), memory=memory, tags=("sales",)).run(
            "What is the total sales amount for the South region in sales.csv?")
        assert run.success and len(run.recalled) == 2
        assert "LESSONS FROM EARLIER RUNS" in llm.prompts[0] and "FileNotFound" in llm.prompts[0]
        assert all(memory.get(l.id).helped == 1 for l in run.recalled)
        assert memory.stats()["episodes"] == 2


def test_failed_runs_are_remembered_as_failures():
    memory = MemoryEngine()
    agent = ReActAgent(ScriptedLLM([final("It was 9,999.")] * 2), ToolRegistry([calculator_tool()]), memory=memory)
    run = agent.run("What did the server cost?", check=lambda answer: "1,250" in answer)
    assert run.success is False
    (episode,) = memory.episodes()
    assert episode["outcome"] == "failure" and [l.kind for l in run.learned] == ["avoid"]
    assert ReActAgent(ScriptedLLM([final("x")]), ToolRegistry(), memory=memory, verifier=None).run(
        "something else entirely", check=lambda a: 1 / 0).success is False       # a crashing check


def test_memory_tools_work_inside_the_loop(tmp_path):
    path = str(tmp_path / "memory.db")
    with MemoryEngine(path) as memory:
        llm = ScriptedLLM([act("remember", {"fact": "The office wifi password rotates on the 1st of each month",
                                            "topic": "office wifi password"}), final("Saved.")])
        assert ReActAgent(llm, standard_tools(memory=memory), memory=memory).run(
            "Remember that the office wifi password rotates on the 1st of each month.").finished
    with MemoryEngine(path) as memory:
        llm = ScriptedLLM([act("recall", {"query": "when does the office wifi password rotate"}),
                           final("It rotates on the 1st of each month.")])
        run = ReActAgent(llm, standard_tools(memory=memory), memory=memory).run(
            "When does the office wifi password change?")
        assert run.finished and "rotates on the 1st" in run.steps[0].observation


def test_conversation_carries_earlier_turns():
    llm = ScriptedLLM([act("calculator", {"expression": "40 * 52"}), final("That is 2080 hours a year."),
                       act("calculator", {"expression": "2080 * 35"}), final("At 35 an hour that is 72800.")])
    chat = Conversation(ReActAgent(llm, ToolRegistry([calculator_tool()])))
    assert chat.ask("How many hours is 40 hours a week for 52 weeks?").finished
    second = chat.ask("And what does that earn at 35 an hour?")
    assert second.answer.endswith("72800.") and second.grounded
    assert "EARLIER IN THIS CONVERSATION" in llm.prompts[2] and "You answered: That is 2080 hours" in llm.prompts[2]
    assert "EARLIER IN THIS CONVERSATION" not in llm.prompts[0]
