"""
Hugging Face Transformers LLM Provider

Supports local and remote open-source models via Hugging Face transformers.
"""

from typing import Dict, Any, Optional, List
import os


class HuggingFaceLLM:
    """
    LLM provider for Hugging Face transformers models.

    Supports:
    - Local models (downloaded and cached)
    - Remote models from Hugging Face Hub
    - Quantized models (4-bit, 8-bit, mxfp4, etc.)
    - Various optimizations (Flash Attention, MegaBlocks MoE, etc.)
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        attn_implementation: Optional[str] = None,
        use_kernels: bool = False,
        trust_remote_code: bool = False,
        max_memory: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize Hugging Face LLM.

        Args:
            model_name: Model ID (e.g., "openai/gpt-oss-20b", "meta-llama/Llama-3.1-8B")
            device_map: Device mapping strategy ("auto", "balanced", etc.)
            torch_dtype: Data type ("auto", "float16", "bfloat16", etc.)
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            attn_implementation: Attention implementation (e.g., "kernels-community/vllm-flash-attn3")
            use_kernels: Use optimized kernels (e.g., MegaBlocks MoE)
            trust_remote_code: Allow remote code execution
            max_memory: Maximum memory per device
        """
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.attn_implementation = attn_implementation
        self.use_kernels = use_kernels
        self.trust_remote_code = trust_remote_code
        self.max_memory = max_memory
        self.kwargs = kwargs

        self.model = None
        self.tokenizer = None

        # Check for HF token
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    def _load_model(self):
        """Load the model and tokenizer"""
        if self.model is not None:
            return  # Already loaded

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library not installed. "
                "Install with: pip install transformers accelerate"
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            token=self.hf_token,
        )

        # Prepare model loading kwargs
        model_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "token": self.hf_token,
        }

        # Add quantization settings
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        # Add attention optimization
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        # Add kernel optimization
        if self.use_kernels:
            model_kwargs["use_kernels"] = True

        # Add max memory constraints
        if self.max_memory:
            model_kwargs["max_memory"] = self.max_memory

        # Merge additional kwargs
        model_kwargs.update(self.kwargs)

        # Load model
        print(f"Loading model: {self.model_name}")
        print(f"Configuration: {model_kwargs}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        print(f"✓ Model loaded successfully")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **generate_kwargs
    ) -> str:
        """
        Generate a response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **generate_kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        self._load_model()

        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        # Generation parameters
        gen_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }
        gen_config.update(generate_kwargs)

        # Generate
        outputs = self.model.generate(**inputs, **gen_config)

        # Decode only the new tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response

    def extract_final_message(self, response: str) -> str:
        """
        Extract final message from models that use channels (like GPT OSS).

        Args:
            response: Full model response

        Returns:
            Final message intended for user
        """
        # For models with channel system (like GPT OSS)
        if "<|channel|>final<|message|>" in response:
            parts = response.split("<|channel|>final<|message|>")
            if len(parts) > 1:
                return parts[-1].split("<|end|>")[0].strip()

        # For regular models, return as-is
        return response

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        self._load_model()
        return len(self.tokenizer.encode(text))

    def __repr__(self) -> str:
        return f"HuggingFaceLLM(model={self.model_name})"


def create_gpt_oss_20b(
    attn_implementation: Optional[str] = None,
    use_kernels: bool = False,
) -> HuggingFaceLLM:
    """
    Create GPT OSS 20B model (fits in 16GB with mxfp4).

    Args:
        attn_implementation: Use Flash Attention 3 if on Hopper GPU
        use_kernels: Use MegaBlocks MoE kernels if not using mxfp4

    Returns:
        Configured HuggingFaceLLM instance
    """
    return HuggingFaceLLM(
        model_name="openai/gpt-oss-20b",
        device_map="auto",
        torch_dtype="auto",
        attn_implementation=attn_implementation,
        use_kernels=use_kernels,
    )


def create_gpt_oss_120b(
    tensor_parallel_size: int = 1,
    attn_implementation: Optional[str] = None,
) -> HuggingFaceLLM:
    """
    Create GPT OSS 120B model (fits on single H100 with mxfp4).

    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        attn_implementation: Use Flash Attention 3 if on Hopper GPU

    Returns:
        Configured HuggingFaceLLM instance
    """
    device_map = {"tp_plan": "auto"} if tensor_parallel_size > 1 else "auto"

    return HuggingFaceLLM(
        model_name="openai/gpt-oss-120b",
        device_map=device_map,
        torch_dtype="auto",
        attn_implementation=attn_implementation,
    )


def create_llama_3_8b(load_in_4bit: bool = False) -> HuggingFaceLLM:
    """
    Create Llama 3.1 8B Instruct model.

    Args:
        load_in_4bit: Use 4-bit quantization to reduce memory

    Returns:
        Configured HuggingFaceLLM instance
    """
    return HuggingFaceLLM(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=load_in_4bit,
    )


def create_mistral_7b(load_in_4bit: bool = False) -> HuggingFaceLLM:
    """
    Create Mistral 7B Instruct model.

    Args:
        load_in_4bit: Use 4-bit quantization

    Returns:
        Configured HuggingFaceLLM instance
    """
    return HuggingFaceLLM(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=load_in_4bit,
    )


def create_qwen_7b(load_in_4bit: bool = False) -> HuggingFaceLLM:
    """
    Create Qwen 2.5 7B Instruct model.

    Args:
        load_in_4bit: Use 4-bit quantization

    Returns:
        Configured HuggingFaceLLM instance
    """
    return HuggingFaceLLM(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=load_in_4bit,
    )
