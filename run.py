"""Iterative Reasoning Framework for Large Language Models via Meta-Level Feedback."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

from meta_feedback import LLMConfig, LLMRuntime, Orchestrator, SQLiteStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Instruction-tuned GGUF model path")
    parser.add_argument("--question", required=True, help="Question and any supporting context")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repeat-penalty", type=float, default=1.1)
    parser.add_argument("--n-ctx", type=int, default=15000)
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="-1: all layers; 0: CPU")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--answer-format", help="Optional constraint for the original final-answer prompt")
    parser.add_argument("--db", type=Path, default=Path("outputs/reasoning.db"), help="SQLite trace database")
    parser.add_argument("--output", type=Path, help="Export this run and its trace to JSON")
    args = parser.parse_args(argv)
    if not args.question.strip() or args.max_steps < 1:
        parser.error("--question must not be empty and --max-steps must be positive")
    if args.max_tokens < 1 or args.n_ctx <= args.max_tokens:
        parser.error("--n-ctx must exceed positive --max-tokens")
    if not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("--temperature must be finite and >= 0; --top-p must be in (0, 1]")
    if args.top_k < 0 or not math.isfinite(args.repeat_penalty) or args.repeat_penalty <= 0:
        parser.error("--top-k must be >= 0 and --repeat-penalty must be finite and positive")
    model_path = args.model.expanduser()
    if not model_path.is_file():
        parser.error(f"GGUF model file not found: {model_path}")

    config = LLMConfig(
        model_path=str(model_path), gpu_id=args.gpu, n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx, temperature=args.temperature, max_tokens=args.max_tokens,
        top_p=args.top_p, top_k=args.top_k, repeat_penalty=args.repeat_penalty,
    )
    try:
        # Keep model diagnostics off stdout, which contains only the final answer.
        with redirect_stdout(sys.stderr):
            runtime = LLMRuntime(config)
            runtime.load()
            store = SQLiteStore(str(args.db))
            orchestrator = Orchestrator(runtime=runtime, store=store, max_steps=args.max_steps)
            state = orchestrator.run(args.question, final_answer_format=args.answer_format)
        if args.output:
            payload = {
                "run_id": state.run_id, "query": state.global_goal,
                "final_answer": state.final_answer, "steps": state.step_index,
                "status": state.current_status.value, "state_summary": state.state_summary,
                "generation": {**asdict(config), "max_steps": args.max_steps,
                               "chat_format": "chatml", "answer_format": args.answer_format},
                "trace": store.get_chain_for_visualization(state.run_id),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(state.final_answer)
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
