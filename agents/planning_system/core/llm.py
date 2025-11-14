"""
Language Model Configuration

Defines LLM configuration for agents (Li component).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os


@dataclass
class LLMConfig:
    """
    Language model configuration for an agent (Li).

    This defines which model to use and how to interact with it.
    """

    # Model selection
    model_name: str  # e.g., "gpt-4o", "claude-3.5-sonnet", "o1-preview"
    provider: str = "openai"  # "openai", "anthropic", "google", "openrouter"

    # Generation parameters
    temperature: float = 0.7  # Sampling temperature (0.0 = deterministic)
    max_tokens: int = 16384  # Max output tokens
    top_p: float = 1.0  # Nucleus sampling parameter

    # Request management
    timeout: int = 60  # API timeout (seconds)
    max_retries: int = 3  # Retry on failures

    # Cost optimization
    use_cache: bool = True  # Enable prompt caching (if supported)

    # API credentials (loaded from environment)
    api_key: Optional[str] = None

    # Custom configurations
    system_prompt_template: str = ""  # Custom system prompt
    few_shot_examples: List[Dict] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Load API key from environment if not provided"""
        if self.api_key is None:
            self.api_key = self._get_api_key()

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment based on provider"""
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_var = key_map.get(self.provider, "OPENAI_API_KEY")
        return os.getenv(env_var)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "use_cache": self.use_cache,
            "system_prompt_template": self.system_prompt_template,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Create from dictionary"""
        return cls(**data)


# Preset configurations for common use cases

def get_high_reasoning_config() -> LLMConfig:
    """Configuration for complex planning and reasoning tasks"""
    return LLMConfig(
        model_name="o1-preview",
        provider="openai",
        temperature=1.0,
        max_tokens=32768,
    )


def get_code_generation_config() -> LLMConfig:
    """Configuration for code generation tasks"""
    return LLMConfig(
        model_name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        temperature=0.5,
        max_tokens=8192,
    )


def get_fast_execution_config() -> LLMConfig:
    """Configuration for quick, simple tasks"""
    return LLMConfig(
        model_name="gpt-4o-mini",
        provider="openai",
        temperature=0.3,
        max_tokens=4096,
    )


def get_default_config() -> LLMConfig:
    """Default balanced configuration"""
    return LLMConfig(
        model_name="gpt-4o-2024-08-06",
        provider="openai",
        temperature=0.7,
        max_tokens=16384,
    )
