"""Portable model configuration, using the supplied experiment settings."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM model configuration"""
    # Model selection
    model_path: str = ""
    
    # GPU settings
    gpu_id: str = "0"  # CUDA_VISIBLE_DEVICES
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    
    # Context settings
    n_ctx: int = 15000  # Context window size
    
    # Generation settings
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    
    # Runtime settings
    verbose: bool = False
    n_batch: int = 512
    n_threads: Optional[int] = None  # None = auto


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)


_config_instance = Config()


def get_config() -> Config:
    return _config_instance
