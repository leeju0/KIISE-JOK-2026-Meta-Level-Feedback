import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run
from meta_feedback import LLMConfig, LLMRuntime
from test_method import ScriptedCompletion, iteration


class CLITests(unittest.TestCase):
    def test_cli_runs_original_pipeline_and_writes_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.touch()
            output = Path(directory) / "trace.json"
            database = Path(directory) / "nested" / "trace.db"
            completion = ScriptedCompletion(iteration("work", "summary") + [{"final_answer": "5"}])
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("meta_feedback.llm_runtime._import_llama_cpp", return_value=lambda **kwargs: completion), \
                    patch.dict(os.environ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                run.main(["--model", str(model), "--question", "Question", "--db", str(database),
                          "--output", str(output), "--n-gpu-layers", "0"])
            payload = json.loads(output.read_text())
            self.assertEqual(stdout.getvalue().strip(), "5")
            self.assertEqual(payload["final_answer"], "5")
            self.assertEqual(payload["steps"], 1)
            self.assertEqual(len(payload["trace"]["c_logs"]), 1)
            self.assertEqual(payload["trace"]["run"]["global_goal"], "Question")
            self.assertEqual(payload["generation"]["top_p"], 0.95)

    def test_invalid_cli_input_exits_before_loading_model(self):
        for extras in [["--max-steps", "0"], ["--top-p", "2"], ["--temperature", "nan"]]:
            with self.subTest(extras=extras), patch("meta_feedback.llm_runtime._import_llama_cpp") as load, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                run.main(["--model", "missing.gguf", "--question", "Question"] + extras)
            self.assertEqual(result.exception.code, 2)
            load.assert_not_called()

    def test_runtime_sets_gpu_before_native_import_and_keeps_original_chatml(self):
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.touch()
            def import_backend():
                seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
                def constructor(**kwargs):
                    seen.append(kwargs)
                    return object()
                return constructor
            with patch.dict(os.environ), patch("meta_feedback.llm_runtime._import_llama_cpp", side_effect=import_backend), \
                    contextlib.redirect_stdout(io.StringIO()):
                runtime = LLMRuntime(LLMConfig(model_path=str(model), gpu_id="2"))
                runtime.load()
                runtime.load()
            self.assertEqual(seen[0], "2")
            self.assertEqual(seen[1]["chat_format"], "chatml")
            self.assertEqual(len(seen), 2)


if __name__ == "__main__":
    unittest.main()
