"""
Iterative Reasoning with Meta-Level Feedback - Orchestrator
Main step loop execution for M/C dual session reasoning
"""

import uuid
from datetime import datetime
from typing import Optional, Callable, Generator, Dict, Any

from .schemas import (
    OrchestrationState, NodeRecord, ChainLinkRecord,
    MLogRecord, CLogRecord, StateSummaryLogRecord,
    NodeType, SessionRole, LinkType, RunStatus, Decision
)
from .llm_runtime import LLMRuntime, get_runtime
from .sqlite_store import SQLiteStore, get_store


class StepEvent:
    """Event emitted during orchestration for UI updates"""
    
    def __init__(
        self,
        event_type: str,
        step_index: int,
        data: Dict[str, Any]
    ):
        self.event_type = event_type  # m_summary, m_planning, c_reasoning, m_decision, final_answer
        self.step_index = step_index
        self.data = data
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "step_index": self.step_index,
            "data": self.data,
            "timestamp": self.timestamp
        }


class Orchestrator:
    """
    Main orchestration engine for iterative reasoning with meta-level feedback.
    
    Runs the author-supplied plan → reason → summarize → decide loop.
    Role-specific KV persistence is disabled, as in the original experiments.
    """
    
    def __init__(
        self,
        runtime: Optional[LLMRuntime] = None,
        store: Optional[SQLiteStore] = None,
        max_steps: int = 3,
        use_kv_cache: bool = False
    ):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if use_kv_cache:
            raise ValueError("Role-specific KV persistence is not included; use_kv_cache must be False")
        self.runtime = runtime or get_runtime()
        self.store = store or get_store()
        self.max_steps = max_steps  # default 3
        
        self._current_state: Optional[OrchestrationState] = None
    
    def run(
        self,
        query: str,
        final_answer_format: Optional[str] = None,
        on_event: Optional[Callable[[StepEvent], None]] = None
    ) -> OrchestrationState:
        """
        Run the full orchestration loop.
        
        Args:
            query: User's input question
            final_answer_format: Optional final answer constraint for extractor prompt
            on_event: Optional callback for step events (for UI updates)
        
        Returns:
            Final orchestration state
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        # Ensure model is loaded
        if not self.runtime.is_loaded():
            self.runtime.load()
        
        # Initialize run
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        self.store.create_run(run_id, query)
        
        state = OrchestrationState(
            run_id=run_id,
            global_goal=query,
            current_input=query,
            step_index=1,
            state_summary="",
            current_status=RunStatus.RUNNING
        )
        self._current_state = state
        
        # Insert initial node
        prev_node_id = self._insert_node(
            state, query, NodeType.USER_INPUT, SessionRole.USER, prev_node_id=None
        )
        
        
        try:
            while state.step_index <= self.max_steps:
                # Determine effective_input
                effective_input = state.global_goal if state.step_index == 1 else state.current_input
                
                # ============================================================
                # Step A) M creates planning (situation analysis, sub-goal)
                # ============================================================
                
                planning_result, planning_raw = self.runtime.m_plan(
                    global_goal=state.global_goal,
                    current_input=effective_input,
                    state_summary=state.state_summary
                )
                state.m_planning = planning_result
                
                
                # Log M planning
                self._log_m_planning(state, effective_input, planning_raw)
                
                # Insert M sub-goal node
                m_node_id = self._insert_node(
                    state,
                    planning_result.sub_goal if planning_result else "",
                    NodeType.M_SUBGOAL,
                    SessionRole.M,
                    prev_node_id=prev_node_id
                )
                
                # Insert link
                self._insert_link(prev_node_id, m_node_id, LinkType.EMITS, state.step_index)
                
                if on_event:
                    on_event(StepEvent("m_planning", state.step_index, {
                        "situation_analysis": planning_result.situation_analysis if planning_result else "",
                        "reasoning_direction": planning_result.reasoning_direction if planning_result else "",
                        "sub_goal": planning_result.sub_goal if planning_result else "",
                        "raw": planning_raw
                    }))
                
                # ============================================================
                # Step B) C performs reasoning
                # ============================================================
                
                c_result, c_raw = self.runtime.c_reason(
                    global_goal=state.global_goal,
                    current_input=effective_input,
                    situation_analysis=planning_result.situation_analysis if planning_result else "",
                    reasoning_direction=planning_result.reasoning_direction if planning_result else "",
                    sub_goal=planning_result.sub_goal if planning_result else "",
                    state_summary=state.state_summary
                )
                state.c_output = c_result
                
                
                # Log C reasoning
                self._log_c_reasoning(state, c_raw)
                
                # Insert C output node
                c_output_text = c_result.reasoning_output if c_result else c_raw
                c_node_id = self._insert_node(
                    state,
                    c_output_text,
                    NodeType.C_OUTPUT,
                    SessionRole.C,
                    prev_node_id=m_node_id
                )
                
                # Insert link
                self._insert_link(m_node_id, c_node_id, LinkType.NEXT, state.step_index)
                
                if on_event:
                    on_event(StepEvent("c_reasoning", state.step_index, {
                        "reasoning_output": c_result.reasoning_output if c_result else "",
                        "raw": c_raw
                    }))
                
                # ============================================================
                # Step C) M updates state_summary with c_output
                # ============================================================
                
                previous_summary = state.state_summary
                summary_result2, summary_raw2 = self.runtime.m_summarize(
                    global_goal=state.global_goal,
                    previous_state_summary=state.state_summary,
                    new_output=c_output_text
                )
                
                if summary_result2 and summary_result2.state_summary:
                    state.state_summary = summary_result2.state_summary
                
                self._log_state_summary(state, c_output_text, state.state_summary, summary_raw2, previous_summary)
                
                
                if on_event:
                    on_event(StepEvent("m_summary_output", state.step_index, {
                        "state_summary": state.state_summary,
                        "raw": summary_raw2
                    }))
                
                # ============================================================
                # Step D) M decides progress/stop
                # ============================================================
                decision_result, decision_raw = self.runtime.m_decide(
                    global_goal=state.global_goal,
                    state_summary=state.state_summary
                )
                state.m_decision = decision_result
                
                
                # Log decision
                self._log_m_decision(state, decision_raw)
                
                if on_event:
                    on_event(StepEvent("m_decision", state.step_index, {
                        "decision": decision_result.decision.value if decision_result else "progress",
                        "reason": decision_result.reason if decision_result else "",
                        "answer_ready": decision_result.answer_ready if decision_result else False,
                        "raw": decision_raw
                    }))
                
                # ============================================================
                # Handle decision
                # ============================================================
                decision = decision_result.decision if decision_result else Decision.PROGRESS
                
                if decision == Decision.STOP:
                    # Extract final answer
                    final_result, final_raw = self.runtime.m_extract_final_answer(
                        global_goal=state.global_goal,
                        state_summary=state.state_summary,
                        final_answer_format=final_answer_format
                    )
                    
                    final_answer = final_result.final_answer if final_result else "no"
                    state.final_answer = final_answer
                    state.current_status = RunStatus.STOP
                    
                    # Insert final answer node
                    final_node_id = self._insert_node(
                        state,
                        final_answer,
                        NodeType.FINAL_ANSWER,
                        SessionRole.M,
                        prev_node_id=c_node_id
                    )
                    
                    # Insert finalizes link
                    self._insert_link(c_node_id, final_node_id, LinkType.FINALIZES, state.step_index)
                    
                    # Update run status
                    self.store.update_run_status(run_id, RunStatus.STOP, final_answer)
                    
                    if on_event:
                        on_event(StepEvent("final_answer", state.step_index, {
                            "final_answer": final_answer,
                            "raw": final_raw
                        }))
                    
                    break
                
                else:  # PROGRESS
                    # Prepare for next iteration
                    prev_node_id = c_node_id
                    state.current_input = c_output_text
                    state.step_index += 1
            
            else:
                # Max steps reached - force answer extraction from current state
                state.step_index = self.max_steps
                final_result, final_raw = self.runtime.m_extract_final_answer(
                    global_goal=state.global_goal,
                    state_summary=state.state_summary,
                    final_answer_format=final_answer_format
                )
                
                final_answer = final_result.final_answer if final_result else "Unable to determine answer"
                state.final_answer = final_answer
                state.current_status = RunStatus.STOP
                
                # Insert final answer node
                final_node_id = self._insert_node(
                    state,
                    f"[Max {self.max_steps} steps reached] {final_answer}",
                    NodeType.FINAL_ANSWER,
                    SessionRole.M,
                    prev_node_id=prev_node_id
                )
                
                self._insert_link(prev_node_id, final_node_id, LinkType.FINALIZES, state.step_index)
                self.store.update_run_status(run_id, RunStatus.STOP, final_answer)
                
                if on_event:
                    on_event(StepEvent("final_answer", state.step_index, {
                        "final_answer": final_answer,
                        "forced": True,
                        "raw": final_raw
                    }))
        
        except Exception as e:
            state.current_status = RunStatus.FAILED
            self.store.update_run_status(run_id, RunStatus.FAILED)
            raise e
        
        finally:
            self._current_state = state
        
        return state
    
    def run_generator(
        self,
        query: str
    ) -> Generator[StepEvent, None, OrchestrationState]:
        """
        Run orchestration as a generator, yielding events.
        Useful for Streamlit streaming updates.
        """
        events = []
        
        def collect_event(event: StepEvent):
            events.append(event)
        
        # This is a simplified version - for true streaming,
        # we'd need to restructure the run() method
        state = self.run(query, on_event=collect_event)
        
        for event in events:
            yield event
        
        return state
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _insert_node(
        self,
        state: OrchestrationState,
        text: str,
        node_type: NodeType,
        role: SessionRole,
        prev_node_id: Optional[str]
    ) -> str:
        """Insert a node and return its ID"""
        node_id = f"{state.run_id}_{node_type.value}_{state.step_index}"
        
        record = NodeRecord(
            node_id=node_id,
            text_label=text[:500] if text else "",  # Truncate for storage
            node_type=node_type,
            step_index=state.step_index,
            prev_node_id=prev_node_id,
            session_role=role
        )
        
        self.store.insert_node(record)
        return node_id
    
    def _insert_link(
        self,
        from_id: str,
        to_id: str,
        link_type: LinkType,
        step_index: int
    ):
        """Insert a chain link"""
        record = ChainLinkRecord(
            from_node_id=from_id,
            to_node_id=to_id,
            link_type=link_type,
            step_index=step_index
        )
        
        self.store.insert_chain_link(record)
    
    def _log_state_summary(
        self,
        state: OrchestrationState,
        input_text: str,
        output_text: str,
        raw_response: str,
        previous_summary: str
    ):
        """Log state summary update"""
        record = StateSummaryLogRecord(
            run_id=state.run_id,
            step_index=state.step_index,
            prev_input_text=input_text[:1000] if input_text else "",
            prev_output_text=output_text[:1000] if output_text else "",
            prev_state_summary=previous_summary,
            new_state_summary=state.state_summary,
            raw_response=raw_response
        )
        
        self.store.insert_state_summary_log(record)
    
    def _log_m_planning(
        self,
        state: OrchestrationState,
        input_text: str,
        raw_response: str
    ):
        """Log M planning output"""
        planning = state.m_planning
        
        record = MLogRecord(
            run_id=state.run_id,
            step_index=state.step_index,
            input_summary=input_text[:500] if input_text else "",
            chain_state_summary=state.state_summary,
            situation_analysis=planning.situation_analysis if planning else "",
            sub_goal=planning.sub_goal if planning else "",
            reasoning_direction=planning.reasoning_direction if planning else "",
            raw_response=raw_response
        )
        
        self.store.insert_m_log(record)
    
    def _log_m_decision(
        self,
        state: OrchestrationState,
        raw_response: str
    ):
        """Log M decision"""
        decision = state.m_decision
        
        record = MLogRecord(
            run_id=state.run_id,
            step_index=state.step_index,
            input_summary="(decision)",
            chain_state_summary=state.state_summary,
            decision=decision.decision.value if decision else "",
            decision_reason=decision.reason if decision else "",
            raw_response=raw_response
        )
        
        self.store.insert_m_log(record)
    
    def _log_c_reasoning(
        self,
        state: OrchestrationState,
        raw_response: str
    ):
        """Log C reasoning output"""
        c_output = state.c_output
        
        # Build prompt text for logging
        prompt_parts = []
        if state.m_planning:
            prompt_parts.append(f"sub_goal: {state.m_planning.sub_goal}")
            prompt_parts.append(f"direction: {state.m_planning.reasoning_direction}")
        
        record = CLogRecord(
            run_id=state.run_id,
            step_index=state.step_index,
            prompt_text="\n".join(prompt_parts),
            raw_response=raw_response,
            extracted_answer=c_output.reasoning_output if c_output else ""
        )
        
        self.store.insert_c_log(record)
    
    @property
    def current_state(self) -> Optional[OrchestrationState]:
        """Get current orchestration state"""
        return self._current_state


# Convenience function
def run_iterative_reasoning(
    query: str,
    on_event: Optional[Callable[[StepEvent], None]] = None,
    max_steps: int = 3
) -> OrchestrationState:
    """
    Run iterative reasoning with meta-level feedback on a query.
    
    Args:
        query: User's input question
        on_event: Optional callback for step events
        max_steps: Maximum number of reasoning steps
    
    Returns:
        Final orchestration state
    """
    orchestrator = Orchestrator(max_steps=max_steps)
    return orchestrator.run(query, on_event=on_event)
