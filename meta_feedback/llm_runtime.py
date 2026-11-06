"""
Iterative Reasoning with Meta-Level Feedback - LLM Runtime
Direct model loading - loads LLM based on user config
"""

import os
import json
import re
from typing import Optional, Dict, Any, Tuple, List, TYPE_CHECKING

# Lazy import for llama_cpp
if TYPE_CHECKING:
    from llama_cpp import Llama

from .prompts import (
    format_messages_for_m, format_messages_for_c,
    build_state_summary_prompt, build_m_planning_prompt,
    build_m_decision_prompt, build_c_reasoning_prompt,
    build_final_answer_extractor_prompt
)
from .schemas import (
    StateSummaryOutput, MPlanningOutput, MDecisionOutput,
    CReasoningOutput, FinalAnswerOutput, Decision
)
from .config import get_config, LLMConfig


def _import_llama_cpp():
    """Lazy import llama_cpp with helpful error message"""
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError as e:
        raise ImportError(
            "llama_cpp not found. Install dependencies with:\n"
            "  pip install -r requirements.txt"
        ) from e


class LLMRuntime:
    """
    LLM Runtime - Direct Loading Mode
    Loads LLM directly based on config settings (model path, GPU, etc.)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        # Get config
        cfg = config or get_config().llm
        
        # Model settings from config
        self.model_path = cfg.model_path
        self.gpu_id = cfg.gpu_id
        self.n_ctx = cfg.n_ctx
        self.n_gpu_layers = cfg.n_gpu_layers
        self.n_batch = cfg.n_batch
        self.n_threads = cfg.n_threads
        self.verbose = cfg.verbose
        
        # Generation settings
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k
        self.repeat_penalty = cfg.repeat_penalty
        
        self.llm: Optional[Any] = None
        self._loaded = False
    
    def load(self) -> "LLMRuntime":
        """Load LLM based on config"""
        if self._loaded:
            return self
        
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        # CUDA visibility must be selected before importing the native backend.
        Llama = _import_llama_cpp()
        print(f"[LLMRuntime] Using GPU: {self.gpu_id}")
        print(f"[LLMRuntime] Loading: {self.model_path}")
        
        # Build kwargs
        llama_kwargs = {
            "model_path": self.model_path,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "verbose": self.verbose,
            "n_batch": self.n_batch,
            "chat_format": "chatml",  # Use chatml to avoid Jinja template issues (strftime_now)
        }
        if self.n_threads is not None:
            llama_kwargs["n_threads"] = self.n_threads
        
        self.llm = Llama(**llama_kwargs)
        self._loaded = True
        print(f"[LLMRuntime] Model loaded!")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model configuration"""
        return {
            "model_path": self.model_path,
            "gpu_id": self.gpu_id,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "loaded": self._loaded
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded
    
    def _generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate with loaded model"""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            stop=stop
        )
        
        return response["choices"][0]["message"]["content"]
    
    def _parse_json(self, text: str, schema_class) -> Tuple[Any, str]:
        """
        Parse JSON from LLM response.
        Returns (parsed_object, raw_text)
        """
        # Try to extract JSON from response
        raw_text = text.strip()
        
        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if json_match:
            json_str = json_match.group()
            try:
                data = json.loads(json_str)
                parsed = schema_class(**data)
                return parsed, raw_text
            except (json.JSONDecodeError, Exception) as e:
                if self.verbose:
                    print(f"JSON parse error: {e}")
        
        # Return None if parsing fails - let caller handle fallback
        return None, raw_text
    
    # =========================================================================
    # M Session Functions
    # =========================================================================
    
    def m_summarize(
        self,
        global_goal: str,
        current_input: Optional[str] = None,
        previous_state_summary: Optional[str] = None,
        new_output: Optional[str] = None
    ) -> Tuple[StateSummaryOutput, str]:
        """
        M: Generate state summary
        - 입력 반영: global_goal, current_input, previous_state_summary
        - 출력 반영: global_goal, state_summary, c_output (new_output)
        """
        prompt = build_state_summary_prompt(
            global_goal=global_goal,
            current_input=current_input,
            previous_state_summary=previous_state_summary,
            new_output=new_output
        )
        
        messages = format_messages_for_m(prompt)
        raw_response = self._generate(messages)
        
        parsed, raw = self._parse_json(raw_response, StateSummaryOutput)
        
        # Fallback: if parsing failed, use raw response as state_summary
        if parsed is None:
            # Try to extract meaningful text
            summary_text = raw_response.strip()
            # Remove JSON artifacts if present
            if summary_text.startswith('{'):
                summary_text = raw_response
            parsed = StateSummaryOutput(state_summary=summary_text)
        
        return parsed, raw
    
    def m_plan(
        self,
        global_goal: str,
        current_input: str,
        state_summary: str
    ) -> Tuple[MPlanningOutput, str]:
        """
        M: Create planning (situation analysis, reasoning direction, sub-goal)
        """
        prompt = build_m_planning_prompt(
            global_goal=global_goal,
            current_input=current_input,
            state_summary=state_summary
        )
        
        messages = format_messages_for_m(prompt)
        raw_response = self._generate(messages)
        
        parsed, raw = self._parse_json(raw_response, MPlanningOutput)
        return parsed, raw
    
    def m_decide(
        self,
        global_goal: str,
        state_summary: str
    ) -> Tuple[MDecisionOutput, str]:
        """
        M: Decide progress/stop
        """
        prompt = build_m_decision_prompt(
            global_goal=global_goal,
            state_summary=state_summary
        )
        
        messages = format_messages_for_m(prompt)
        raw_response = self._generate(messages)
        
        parsed, raw = self._parse_json(raw_response, MDecisionOutput)
        
        # Ensure decision is valid
        if parsed and hasattr(parsed, 'decision'):
            if isinstance(parsed.decision, str):
                decision_str = parsed.decision.lower().strip()
                if decision_str in ('progress', 'stop'):
                    parsed.decision = Decision(decision_str)
                else:
                    parsed.decision = Decision.PROGRESS  # default
        
        return parsed, raw
    
    def m_extract_final_answer(
        self,
        global_goal: str,
        state_summary: str,
        final_answer_format: Optional[str] = None
    ) -> Tuple[FinalAnswerOutput, str]:
        """
        M: Extract final answer
        """
        prompt = build_final_answer_extractor_prompt(
            global_goal=global_goal,
            state_summary=state_summary,
            final_answer_format=final_answer_format
        )
        
        messages = format_messages_for_m(prompt)
        raw_response = self._generate(messages)
        
        parsed, raw = self._parse_json(raw_response, FinalAnswerOutput)
        
        # If parsing failed, use raw response as answer
        if parsed is None:
            # Try to extract meaningful answer from raw text
            answer = raw_response.strip()
            # Remove common prefixes
            for prefix in ['Answer:', 'Final answer:', 'The answer is']:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
            parsed = FinalAnswerOutput(final_answer=answer)
        
        return parsed, raw
    
    # =========================================================================
    # C Session Functions
    # =========================================================================
    
    def c_reason(
        self,
        global_goal: str,
        current_input: str,
        situation_analysis: str,
        reasoning_direction: str,
        sub_goal: str,
        state_summary: str
    ) -> Tuple[CReasoningOutput, str]:
        """
        C: Perform reasoning based on M's planning
        """
        prompt = build_c_reasoning_prompt(
            global_goal=global_goal,
            current_input=current_input,
            situation_analysis=situation_analysis,
            reasoning_direction=reasoning_direction,
            sub_goal=sub_goal,
            state_summary=state_summary
        )
        
        messages = format_messages_for_c(prompt)
        raw_response = self._generate(messages)
        
        parsed, raw = self._parse_json(raw_response, CReasoningOutput)
        return parsed, raw


# Global instance
_runtime_instance: Optional[LLMRuntime] = None


def get_runtime(config: Optional[LLMConfig] = None) -> LLMRuntime:
    """
    Get or create LLM runtime instance.
    
    Args:
        config: Optional LLMConfig to use. If None, uses global config.
    
    Returns:
        LLMRuntime instance
    """
    global _runtime_instance
    if _runtime_instance is None:
        _runtime_instance = LLMRuntime(config=config)
    return _runtime_instance


def reset_runtime():
    """Reset the global runtime instance (for reconnecting to server)"""
    global _runtime_instance
    _runtime_instance = None


def create_runtime(config: Optional[LLMConfig] = None) -> LLMRuntime:
    """
    Create a new LLM runtime instance (doesn't use global singleton).
    
    Args:
        config: Optional LLMConfig to use
    
    Returns:
        New LLMRuntime instance
    """
    return LLMRuntime(config=config)
