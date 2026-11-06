"""Exercise the author-supplied runtime and loop with real SQLite storage."""

import json
import tempfile
import unittest
from pathlib import Path

from meta_feedback import LLMConfig, LLMRuntime, Orchestrator, SQLiteStore
from meta_feedback.schemas import RunStatus


class ScriptedCompletion:
    """Replace only native model inference, keeping prompts and parsing real."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.responses)
        if isinstance(content, Exception):
            raise content
        if not isinstance(content, str):
            content = json.dumps(content)
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def iteration(output, summary, decision="stop", ready=True):
    return [
        {"situation_analysis": "Inspect facts", "reasoning_direction": "Calculate", "sub_goal": "Solve"},
        {"reasoning_output": output},
        {"state_summary": summary},
        {"decision": decision, "reason": "Checked", "answer_ready": ready},
    ]


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SQLiteStore(str(Path(self.directory.name) / "trace.db"))

    def engine(self, responses, max_steps=3):
        self.runtime = LLMRuntime(LLMConfig())
        self.model = ScriptedCompletion(responses)
        self.runtime.llm = self.model
        self.runtime._loaded = True
        return Orchestrator(runtime=self.runtime, store=self.store,
                            max_steps=max_steps, use_kv_cache=False)

    def test_original_role_prompts_and_stop_order(self):
        engine = self.engine(iteration("2 + 3 = 5", "Answer is 5", ready=False)
                             + [{"final_answer": "5"}])
        events = []
        state = engine.run("What is 2 + 3?", on_event=events.append)
        self.assertEqual(state.final_answer, "5")
        self.assertEqual(state.step_index, 1)
        self.assertEqual([e.event_type for e in events],
                         ["m_planning", "c_reasoning", "m_summary_output", "m_decision", "final_answer"])
        self.assertEqual([c["messages"][0]["content"].splitlines()[0] for c in self.model.calls],
                         ["You are a meta-reasoning analyst.", "You are a reasoning executor.",
                          "You are a meta-reasoning analyst.", "You are a meta-reasoning analyst.",
                          "You are a meta-reasoning analyst."])
        self.assertEqual(self.store.get_run(state.run_id)["final_answer"], "5")

    def test_progress_uses_latest_output_and_summary_not_full_history(self):
        engine = self.engine(iteration("OLD_RAW", "summary1", "progress")
                             + iteration("latest output", "summary2", "progress")
                             + iteration("solved", "summary3") + [{"final_answer": "9"}])
        state = engine.run("Find the result")
        prompt2 = self.model.calls[4]["messages"][1]["content"]
        prompt3 = self.model.calls[8]["messages"][1]["content"]
        self.assertIn("OLD_RAW", prompt2)
        self.assertIn("summary1", prompt2)
        self.assertIn("latest output", prompt3)
        self.assertIn("summary2", prompt3)
        self.assertNotIn("OLD_RAW", prompt3)
        self.assertEqual(state.step_index, 3)

    def test_chain_has_no_missing_nodes(self):
        engine = self.engine(iteration("work", "summary") + [{"final_answer": "done"}])
        state = engine.run("Question")
        nodes = self.store.get_nodes_by_run(state.run_id)
        ids = {n["node_id"] for n in nodes}
        for link in self.store.get_chain_links_by_run(state.run_id):
            self.assertIn(link["from_node_id"], ids)
            self.assertIn(link["to_node_id"], ids)

    def test_forced_answer_preserves_requested_format(self):
        engine = self.engine(iteration("work", "summary", "progress")
                             + [{"final_answer": "b"}], max_steps=1)
        engine.run("Question", final_answer_format="one of a, b, c, d, e")
        self.assertIn("one of a, b, c, d, e", self.model.calls[-1]["messages"][1]["content"])

    def test_forced_answer_reports_completed_steps_and_connects_final_node(self):
        engine = self.engine(iteration("work", "summary", "progress")
                             + [{"final_answer": "done"}], max_steps=1)
        events = []
        state = engine.run("Question", on_event=events.append)
        self.assertEqual(state.step_index, 1)
        self.assertTrue(events[-1].data["forced"])
        self.assertEqual(events[-1].step_index, 1)
        links = self.store.get_chain_links_by_run(state.run_id)
        self.assertEqual(links[-1]["link_type"], "finalizes")

    def test_summary_log_preserves_previous_and_updated_states(self):
        engine = self.engine(iteration("work1", "summary1", "progress")
                             + iteration("work2", "summary2") + [{"final_answer": "done"}])
        state = engine.run("Question")
        logs = self.store.get_state_summary_logs_by_run(state.run_id)
        self.assertEqual([(r["prev_state_summary"], r["new_state_summary"]) for r in logs],
                         [("", "summary1"), ("summary1", "summary2")])

    def test_original_raw_text_fallbacks_are_preserved(self):
        engine = self.engine([iteration("work", "summary")[0], "Unstructured reasoning",
                              "Summary as plain text", "Invalid decision",
                              "Final answer: yes"], max_steps=1)
        state = engine.run("Question")
        self.assertEqual(state.state_summary, "Summary as plain text")
        self.assertEqual(state.final_answer, "yes")
        self.assertIn("Unstructured reasoning", self.model.calls[2]["messages"][1]["content"])

    def test_failed_generation_marks_database_run_failed(self):
        engine = self.engine([RuntimeError("inference failed")])
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            engine.run("Question")
        self.assertEqual(engine.current_state.current_status, RunStatus.FAILED)
        self.assertEqual(self.store.get_run(engine.current_state.run_id)["current_status"], "failed")

    def test_explicit_zero_temperature_reaches_model(self):
        self.engine([{"final_answer": "done"}])
        self.runtime._generate([{"role": "user", "content": "Question"}], temperature=0.0)
        self.assertEqual(self.model.calls[0]["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
