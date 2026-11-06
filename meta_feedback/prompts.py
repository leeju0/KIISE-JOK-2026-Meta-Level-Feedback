"""
Iterative Reasoning with Meta-Level Feedback - Prompts
All prompt templates for M and C sessions
"""

from typing import Optional


# =============================================================================
# System Prompts
# =============================================================================

M_SYSTEM_PROMPT = """You are a meta-reasoning analyst.
Your role is to analyze reasoning state, create sub-goals, and decide when to stop.
Always output valid JSON only. No extra text."""

C_SYSTEM_PROMPT = """You are a reasoning executor.
Your role is to perform actual reasoning based on given sub-goals.
Always output valid JSON only. No extra text."""


# =============================================================================
# A) State Summary Prompt
# =============================================================================

def build_state_summary_prompt(
    global_goal: str,
    current_input: Optional[str] = None,
    previous_state_summary: Optional[str] = None,
    new_output: Optional[str] = None
) -> str:
    """
    현재 추론 상황 요약 프롬프트
    - Iter 1: global_goal만
    - C output 후: global_goal + previous_state_summary + new_output
    - 그 외: global_goal + current_input + previous_state_summary
    """
    prompt = """Summarize the current reasoning state.

Global goal:
{global_goal}

Current input:
{current_input}

Previous state summary:
{previous_state_summary}

New output:
{new_output}

Summarize what has been done and what remains to solve.
Output JSON only.

Output format:
{{
  "state_summary": "Current reasoning state summary"
}}

Output : 
"""
    
    return prompt.format(
        global_goal=global_goal,
        current_input=current_input or global_goal,
        previous_state_summary=previous_state_summary or "(none - first step)",
        new_output=new_output or "(none)"
    )


# =============================================================================
# B) Progress/Stop Decision Prompt
# =============================================================================

def build_m_decision_prompt(
    global_goal: str,
    state_summary: str
) -> str:
    """
    M의 progress/stop 판정 프롬프트
    """
    prompt = """You decide whether to continue reasoning.

Global goal:
{global_goal}

State summary:
{state_summary}

Rules:
- If goal is solved, choose stop.
- If key uncertainty remains, choose progress.
- Output JSON only.

Output format:
{{
  "decision": "progress or stop",
  "reason": "one short sentence",
  "answer_ready": true or false
}}
Output : 
"""
    
    return prompt.format(
        global_goal=global_goal,
        state_summary=state_summary
    )


# =============================================================================
# C) M Planning Prompt (BG-HMG style)
# =============================================================================

def build_m_planning_prompt(
    global_goal: str,
    current_input: str,
    state_summary: str
) -> str:
    """
    M의 planning 프롬프트 (situation analysis, reasoning direction, sub-goal)
    BG-HMG의 review feedback 스타일 재사용
    """
    prompt = """You are a reasoning planner.

Global goal:
{global_goal}

Current input:
{current_input}

State summary:
{state_summary}

Generate:
1) situation_analysis
2) reasoning_direction
3) sub_goal

Rules:
- Be concrete.
- Keep each field short.
- Output JSON only.

Output format:
{{
  "situation_analysis": "...",
  "reasoning_direction": "...",
  "sub_goal": "..."
}}
Output : 
"""
    
    return prompt.format(
        global_goal=global_goal,
        current_input=current_input,
        state_summary=state_summary
    )


# =============================================================================
# D) C Reasoning Prompt (BG-HMG style, Top-N removed)
# =============================================================================

def build_c_reasoning_prompt(
    global_goal: str,
    current_input: str,
    situation_analysis: str,
    reasoning_direction: str,
    sub_goal: str,
    state_summary: str
) -> str:
    """
    C의 추론 프롬프트
    """
    prompt = """You are the reasoner.

Global goal:
{global_goal}

Current input:
{current_input}

Situation analysis:
{situation_analysis}

Reasoning direction:
{reasoning_direction}

Sub goal:
{sub_goal}

State summary:
{state_summary}

Do reasoning to achieve the sub goal.
Output JSON only.

Output format:
{{
  "reasoning_output": "Your reasoning result"
}}
Output : 
"""
    
    return prompt.format(
        global_goal=global_goal,
        current_input=current_input,
        situation_analysis=situation_analysis,
        reasoning_direction=reasoning_direction,
        sub_goal=sub_goal,
        state_summary=state_summary
    )


# =============================================================================
# E) Final Answer Extractor Prompt
# =============================================================================

def build_final_answer_extractor_prompt(
    global_goal: str,
    state_summary: str,
    final_answer_format: Optional[str] = None
) -> str:
    """
    최종 답 추출 프롬프트
    - final_answer_format이 주어지면 해당 형식으로 답을 제한
    """
    answer_format = final_answer_format or "short answer phrase"

    prompt = """You extract the final answer from reasoning state.

Global goal:
{global_goal}

State summary:
{state_summary}

Based on the reasoning state, answer the global goal question.
Output JSON only.

Output format:
{{
    "final_answer": "{answer_format}"
}}
Output : 
"""
    
    return prompt.format(
        global_goal=global_goal,
                state_summary=state_summary,
                answer_format=answer_format
    )

# =============================================================================
# Message Formatting for llama-cpp-python
# =============================================================================

def format_messages_for_m(user_prompt: str) -> list:
    """M 세션용 메시지 포맷"""
    return [
        {"role": "system", "content": M_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]


def format_messages_for_c(user_prompt: str) -> list:
    """C 세션용 메시지 포맷"""
    return [
        {"role": "system", "content": C_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
