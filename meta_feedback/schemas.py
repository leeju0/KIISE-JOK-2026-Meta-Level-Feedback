"""
Iterative Reasoning with Meta-Level Feedback - Data Schemas
Pydantic models for JSON input/output validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class NodeType(str, Enum):
    USER_INPUT = "user_input"
    M_SUBGOAL = "m_subgoal"
    C_OUTPUT = "c_output"
    FINAL_ANSWER = "final_answer"


class SessionRole(str, Enum):
    USER = "user"
    M = "M"
    C = "C"


class LinkType(str, Enum):
    NEXT = "next"
    EMITS = "emits"
    FINALIZES = "finalizes"


class RunStatus(str, Enum):
    RUNNING = "running"
    STOP = "stop"
    FAILED = "failed"


class Decision(str, Enum):
    PROGRESS = "progress"
    STOP = "stop"


# =============================================================================
# M Session Output Schemas
# =============================================================================

class StateSummaryOutput(BaseModel):
    """현재 추론 상황 요약"""
    state_summary: str = Field(..., description="Current reasoning state summary")


class MPlanningOutput(BaseModel):
    """M의 planning 결과 (situation analysis, sub-goal, reasoning direction)"""
    situation_analysis: str = Field(..., description="Analysis of current state")
    reasoning_direction: str = Field(..., description="Direction for C's reasoning")
    sub_goal: str = Field(..., description="Concrete sub-goal for this step")


class MDecisionOutput(BaseModel):
    """M의 progress/stop 판정 결과"""
    decision: Decision = Field(..., description="progress or stop")
    reason: str = Field(..., description="Short reason for decision")
    answer_ready: bool = Field(..., description="Whether final answer is ready")


# =============================================================================
# C Session Output Schemas
# =============================================================================

class CReasoningOutput(BaseModel):
    """C의 추론 결과"""
    reasoning_output: str = Field(..., description="Reasoning result for this step")


# =============================================================================
# Final Answer Schema
# =============================================================================

class FinalAnswerOutput(BaseModel):
    """최종 답 추출 결과"""
    final_answer: str = Field(..., description="Final answer")


# =============================================================================
# SQLite Table Schemas (for reference / serialization)
# =============================================================================

class NodeRecord(BaseModel):
    """node table record"""
    node_id: str
    text_label: str
    node_type: NodeType
    step_index: int
    prev_node_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    session_role: SessionRole


class ChainLinkRecord(BaseModel):
    """chain_link table record"""
    from_node_id: str
    to_node_id: str
    link_type: LinkType
    step_index: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class RunRecord(BaseModel):
    """run table record"""
    run_id: str
    global_goal: str
    current_status: RunStatus = RunStatus.RUNNING
    final_answer: Optional[str] = None
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None


class MLogRecord(BaseModel):
    """m_log table record"""
    run_id: str
    step_index: int
    input_summary: str
    chain_state_summary: str
    prior_chain_json: Optional[str] = None
    situation_analysis: Optional[str] = None
    sub_goal: Optional[str] = None
    reasoning_direction: Optional[str] = None
    decision: Optional[str] = None
    decision_reason: Optional[str] = None
    raw_response: str


class CLogRecord(BaseModel):
    """c_log table record"""
    run_id: str
    step_index: int
    prompt_text: str
    raw_response: str
    extracted_answer: Optional[str] = None


class StateSummaryLogRecord(BaseModel):
    """state_summary_log table record"""
    run_id: str
    step_index: int
    prev_input_text: str
    prev_output_text: str
    prev_state_summary: Optional[str] = None
    new_state_summary: str
    raw_response: str
    compression_ratio: Optional[float] = None


class KVStateRecord(BaseModel):
    """kv_state table record"""
    run_id: str
    role: Literal["M", "C"]
    step_index: int
    state_ref: str  # file path or identifier
    token_count: Optional[int] = None
    byte_size: Optional[int] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Orchestrator State
# =============================================================================

class OrchestrationState(BaseModel):
    """Current state during orchestration loop"""
    run_id: str
    global_goal: str
    current_input: str
    step_index: int = 1
    state_summary: str = "(empty)"
    current_status: RunStatus = RunStatus.RUNNING
    
    # Latest outputs
    m_planning: Optional[MPlanningOutput] = None
    c_output: Optional[CReasoningOutput] = None
    m_decision: Optional[MDecisionOutput] = None
    final_answer: Optional[str] = None
