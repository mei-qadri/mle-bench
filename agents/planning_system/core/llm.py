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
    model_name: str  # e.g., "gpt-4o", "claude-3.5-sonnet", "openai/gpt-oss-20b"
    provider: str = "openai"  # "openai", "anthropic", "google", "openrouter", "huggingface"

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
            "huggingface": "HF_TOKEN",  # Hugging Face token
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


# Hugging Face / Open-Source Model Configurations

def get_gpt_oss_20b_config() -> LLMConfig:
    """
    Configuration for GPT OSS 20B (open-source reasoning model).

    Fits in 16GB RAM with mxfp4 quantization.
    """
    return LLMConfig(
        model_name="openai/gpt-oss-20b",
        provider="huggingface",
        temperature=1.0,  # Reasoning models benefit from higher temperature
        max_tokens=16384,
        extra_params={
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": None,  # Set to "kernels-community/vllm-flash-attn3" for Hopper GPUs
            "use_kernels": False,  # Set to True for MegaBlocks MoE optimization
        },
    )


def get_gpt_oss_120b_config() -> LLMConfig:
    """
    Configuration for GPT OSS 120B (large open-source reasoning model).

    Fits on single H100 GPU with mxfp4 quantization.
    """
    return LLMConfig(
        model_name="openai/gpt-oss-120b",
        provider="huggingface",
        temperature=1.0,
        max_tokens=32768,
        extra_params={
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "kernels-community/vllm-flash-attn3",  # Recommended for Hopper
        },
    )


def get_llama_3_8b_config(load_in_4bit: bool = False, force_cpu: bool = False) -> LLMConfig:
    """
    Configuration for Llama 3.1 8B Instruct (open-source).

    Args:
        load_in_4bit: Use 4-bit quantization to reduce memory (~5GB GPU)
        force_cpu: Force CPU execution (slow but uses RAM instead of GPU)
    """
    extra_params = {
        "device_map": "cpu" if force_cpu else "auto",
        "torch_dtype": "auto",
        "load_in_4bit": load_in_4bit and not force_cpu,  # Can't use quantization on CPU
    }

    return LLMConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        provider="huggingface",
        temperature=0.7,
        max_tokens=8192,
        extra_params=extra_params,
    )


def get_mistral_7b_config(load_in_4bit: bool = False) -> LLMConfig:
    """
    Configuration for Mistral 7B Instruct (open-source).

    Args:
        load_in_4bit: Use 4-bit quantization
    """
    return LLMConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        provider="huggingface",
        temperature=0.7,
        max_tokens=8192,
        extra_params={
            "device_map": "auto",
            "torch_dtype": "auto",
            "load_in_4bit": load_in_4bit,
        },
    )


def get_qwen_7b_config(load_in_4bit: bool = False) -> LLMConfig:
    """
    Configuration for Qwen 2.5 7B Instruct (open-source).

    Args:
        load_in_4bit: Use 4-bit quantization
    """
    return LLMConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        provider="huggingface",
        temperature=0.7,
        max_tokens=8192,
        extra_params={
            "device_map": "auto",
            "torch_dtype": "auto",
            "load_in_4bit": load_in_4bit,
        },
    )


# Open-Source Model Presets (matching commercial API presets)
# Note: All presets use the SAME model to avoid loading multiple models into GPU memory

def get_oss_high_reasoning_config() -> LLMConfig:
    """Open-source configuration for complex planning and reasoning tasks"""
    # Use same model as other tasks to avoid CUDA OOM
    # Check env var for CPU mode
    force_cpu = os.getenv("LLM_FORCE_CPU", "false").lower() == "true"
    return get_llama_3_8b_config(load_in_4bit=True, force_cpu=force_cpu)


def get_oss_code_generation_config() -> LLMConfig:
    """Open-source configuration for code generation tasks"""
    force_cpu = os.getenv("LLM_FORCE_CPU", "false").lower() == "true"
    return get_llama_3_8b_config(load_in_4bit=True, force_cpu=force_cpu)


def get_oss_fast_execution_config() -> LLMConfig:
    """Open-source configuration for quick, simple tasks"""
    # Use same model as other tasks to avoid CUDA OOM
    force_cpu = os.getenv("LLM_FORCE_CPU", "false").lower() == "true"
    return get_llama_3_8b_config(load_in_4bit=True, force_cpu=force_cpu)


def get_oss_default_config() -> LLMConfig:
    """Open-source default balanced configuration"""
    force_cpu = os.getenv("LLM_FORCE_CPU", "false").lower() == "true"
    return get_llama_3_8b_config(load_in_4bit=True, force_cpu=force_cpu)
