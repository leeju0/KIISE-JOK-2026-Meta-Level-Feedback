<h1 align="center">
  Iterative Reasoning Framework for Large Language Models via Meta-Level Feedback
</h1>

<p align="center">
  <strong>메타 수준 피드백을 활용한 대형 언어 모델의 단계적 추론 프레임워크</strong>
</p>

<p align="center">
  <strong>Juyoung Lee · Jaegyoon Ahn</strong><br>
  Incheon National University
</p>

<p align="center">
  <em>Journal of KIISE (정보과학회논문지), 53(10), 1032–1040, 2026</em>
</p>

<p align="center">
  <a href="https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003394239">
    <img alt="KCI 우수등재학술지" src="https://img.shields.io/badge/KCI-%EC%9A%B0%EC%88%98%EB%93%B1%EC%9E%AC%ED%95%99%EC%88%A0%EC%A7%80-D4A017?style=flat-square">
  </a>
  <img alt="Journal of KIISE · 2026" src="https://img.shields.io/badge/Journal%20of%20KIISE-2026-2563EB?style=flat-square">
</p>

<p align="center">
  <a href="https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003394239"><strong>📄 Read Paper (KCI)</strong></a>
  &nbsp;·&nbsp;
  <a href="#framework"><strong>🧠 Framework</strong></a>
  &nbsp;·&nbsp;
  <a href="#quick-start"><strong>🚀 Quick Start</strong></a>
  &nbsp;·&nbsp;
  <a href="#citation"><strong>📚 Citation</strong></a>
</p>

---

## Overview

This repository provides a standalone Python implementation of the **iterative reasoning framework with meta-level feedback** described in our paper. A **single LLM** operates in two logical sessions: the **meta session (M)** controls the reasoning process, and the **execution session (C)** carries out each reasoning step.

**Key characteristics**

- **Meta-level control:** Plans the next sub-goal based on the current reasoning state.
- **Iterative execution:** Alternates planning, reasoning, summarization, and continuation decisions.
- **Inspectable traces:** Records intermediate reasoning, summaries, decisions, and links in SQLite.
- **Standalone inference:** Supports an instruction-tuned GGUF model through the original runtime.

## Abstract

Large Language Models (LLMs) have demonstrated strong performance on complex reasoning tasks, and various methods have been proposed to further improve their reasoning ability. Chain-of-Thought (CoT) has the limitation of error propagation along a single reasoning path, while self-consistency alleviates this issue but requires high computational cost. In addition, although recent commercial LLMs such as GPT-4o and GPT-5 exhibit enhanced reasoning performance, their reasoning processes are still carried out largely internally, making them difficult to observe and interpret. To address these limitations, this study proposes **an iterative reasoning framework** that separates a single LLM into a meta session and an execution session. The meta session monitors the current reasoning state and determines the goal and direction of the next step, while the execution session performs the actual reasoning accordingly. Through this design, the proposed framework improves the stability, controllability, and explainability of reasoning, and demonstrates superior performance over existing methods in multi-hop and causal reasoning tasks.

## Framework

<p align="center">
  <img src="assets/figures/framework.png" alt="Overall architecture of the proposed meta-level feedback framework" width="850">
</p>

<p align="center"><em>Figure 1. Overall framework of the proposed method.</em></p>

The same LLM takes on two complementary roles:

| Session | Responsibility |
| :--- | :--- |
| **Meta session (M)** | Examines the current state, plans the next sub-goal, updates the summary, and decides whether to continue. |
| **Execution session (C)** | Performs the reasoning task specified by the meta session. |

The framework follows a **plan → reason → summarize → decide** loop. Its reasoning trace exposes intermediate outputs, state summaries, and termination decisions for inspection.

## Reasoning Example

<p align="center">
  <img src="assets/figures/reasoning-example.png" alt="Iterative reasoning trace on a MathQA example" width="850">
</p>

<p align="center"><em>Figure 2. Iterative reasoning example of the proposed framework on a MathQA problem.</em></p>

Both figures are extracted from the authors' manuscript. Publication details are available on the [KCI article page](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003394239).

## Quick Start

**Requirements:** Python 3.9 or later and an instruction-tuned GGUF model compatible with the original `chatml` runtime format.

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` (Command Prompt) or `.venv\Scripts\Activate.ps1` (PowerShell).

### 2. Run a question

```bash
python run.py \
  --model /path/to/model.gguf \
  --question "Alice has 3 apples and buys 2 more. How many apples does she have?" \
  --output outputs/example.json
```

The CLI prints the final answer. Reasoning traces are recorded in `outputs/reasoning.db`; `--output` additionally exports the run as JSON.

### 3. Optional CUDA acceleration

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall -r requirements.txt
```

Use `--n-gpu-layers 0` for CPU inference or `--gpu 0` to select a CUDA device. Consult the [llama-cpp-python installation guide](https://llama-cpp-python.readthedocs.io/en/latest/) for platform-specific setup.

### Default settings

| Option | Default |
| :--- | :--- |
| `--max-steps` | `3` |
| `--max-tokens` | `2048` |
| `--temperature` | `0.1` |
| `--top-p` | `0.95` |
| `--top-k` | `40` |
| `--repeat-penalty` | `1.1` |
| `--n-ctx` | `15000` |

Use `--max-steps 4` for a four-step limit. These values follow the supplied source-code defaults and **do not imply that every published experiment used identical settings**. Set `--db` to change the SQLite log path and `--answer-format` to pass an optional answer constraint. Run `python run.py --help` for all options.

## Repository Structure

| File | Purpose |
| :--- | :--- |
| `meta_feedback/orchestrator.py` | M/C iterative reasoning loop and step events |
| `meta_feedback/prompts.py` | System/task prompts and final-answer extraction |
| `meta_feedback/llm_runtime.py` | Shared GGUF runtime and response parsing |
| `meta_feedback/sqlite_store.py` | Reasoning-chain nodes, links, and session logs |
| `meta_feedback/schemas.py` | Structured responses and trace schemas |
| `meta_feedback/config.py` | Portable model settings |
| `run.py` | Single-question command-line interface |

<details>
<summary><strong>🔧 Implementation details and release adjustments</strong></summary>

<br>

This release is extracted from the authors' original `dual_thinking_machine` source. It preserves the M/C system prompts, planning and reasoning prompts, state summaries, continuation decision logic, final-answer extractor, and JSON parsing fallbacks.

The original experimental path is used with role-specific KV persistence disabled. The `chatml` message format and raw-text fallbacks are retained. Full traces are logged separately; subsequent planning uses the previous reasoning output and the updated summary.

For standalone use, package imports, model paths, and CLI entry points were adapted. Integration fixes address the initial chain link, previous-summary logging, completed-step count, final link at the iteration limit, answer-format propagation on forced termination, and explicit zero-temperature overrides. CUDA device selection now precedes the native backend import.

The underlying **plan → reason → summarize → decide** sequence and prompt text are preserved.

</details>

## Tests and Reproducibility

The included tests exercise the original runtime, parsing, orchestration, CLI, and SQLite storage with scripted model outputs. Control-flow tests can run **without a model or GPU**:

```bash
python -m unittest discover -s tests -v
```

> [!NOTE]
> Native GGUF inference and the published benchmark scores have **not** been revalidated in this standalone release.

## Citation

If you find this work useful, please cite the [published paper](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003394239):

```bibtex
@article{ART003394239,
  author  = {이주영 and 안재균},
  title   = {메타 수준 피드백을 활용한 대형 언어 모델의 단계적 추론 프레임워크},
  journal = {정보과학회논문지},
  year    = {2026},
  volume  = {53},
  number  = {10},
  pages   = {1032--1040},
  issn    = {2383-630X},
  url     = {https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003394239}
}
```

<p align="center">
  <sub>Published in <strong>Journal of KIISE</strong> · 2026 · <a href="https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003394239">KCI Article</a></sub>
</p>
