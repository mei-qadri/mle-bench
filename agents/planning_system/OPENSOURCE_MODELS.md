# Using Open-Source Models

The agentic planning system supports open-source models from Hugging Face, allowing you to run the system without commercial API keys and costs.

## Benefits

✓ **No API costs** - Run locally without paying per token
✓ **Full privacy** - Your data never leaves your machine
✓ **No rate limits** - Generate as much as you need
✓ **Offline operation** - Works without internet after download
✓ **Customizable** - Fine-tune models for your specific needs

## Supported Models

### Reasoning Models

**GPT OSS 20B** (Hypothetical example from your description)
- Parameters: 21B (3.6B active with MoE)
- Memory: 16GB with mxfp4 quantization
- Use case: Complex reasoning, agentic tasks
- Config: `get_gpt_oss_20b_config()`

**GPT OSS 120B**
- Parameters: 117B (5.1B active with MoE)
- Memory: Single H100 (80GB) with mxfp4
- Use case: Advanced reasoning, high-quality planning
- Config: `get_gpt_oss_120b_config()`

### General Purpose Models

**Llama 3.1 8B Instruct**
- Parameters: 8B
- Memory: ~4GB with 4-bit quantization
- Use case: General purpose, good balance
- Config: `get_llama_3_8b_config(load_in_4bit=True)`

**Mistral 7B Instruct**
- Parameters: 7B
- Memory: ~4GB with 4-bit quantization
- Use case: Efficient, good quality
- Config: `get_mistral_7b_config(load_in_4bit=True)`

**Qwen 2.5 7B Instruct**
- Parameters: 7B
- Memory: ~4GB with 4-bit quantization
- Use case: Multilingual, competitive performance
- Config: `get_qwen_7b_config(load_in_4bit=True)`

## Installation

### Basic Requirements

```bash
# Core transformers dependencies
pip install transformers accelerate torch

# For quantization support
pip install bitsandbytes  # 4-bit/8-bit quantization
```

### For GPT OSS Models (mxfp4)

```bash
# Triton 3.4+ for mxfp4 kernel support
pip install "triton>=3.4"

# Kernels library for optimizations
pip install kernels
```

### Optional Optimizations

```bash
# Flash Attention (for supported GPUs)
pip install flash-attn

# AMD ROCm support
# (Install ROCm following AMD documentation)
```

## Quick Start

### 1. Using Pre-configured Models

```python
from agents.planning_system.core.llm import get_llama_3_8b_config
from agents.planning_system.planning.planner import PlanningModule, ProblemSpecification

# Create LLM configuration for Llama 3.1 8B
llm_config = get_llama_3_8b_config(load_in_4bit=True)

# Create planner with open-source model
planner = PlanningModule(llm_config=llm_config)

# Define problem
problem_spec = ProblemSpecification(
    task_description="Your ML task",
    competition_id="your-competition",
    evaluation_metric="accuracy",
)

# Generate plan (uses Llama 3.1 instead of GPT-4)
plan = planner.plan_workflow(problem_spec)
```

### 2. Custom Model Configuration

```python
from agents.planning_system.core.llm import LLMConfig

# Configure any Hugging Face model
llm_config = LLMConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    provider="huggingface",
    temperature=0.7,
    max_tokens=8192,
    extra_params={
        "device_map": "auto",
        "torch_dtype": "auto",
        "load_in_4bit": True,  # Reduce memory usage
    },
)
```

### 3. Advanced: GPT OSS with Optimizations

```python
from agents.planning_system.core.llm import LLMConfig

# GPT OSS 20B with Flash Attention (Hopper GPUs)
llm_config = LLMConfig(
    model_name="openai/gpt-oss-20b",
    provider="huggingface",
    temperature=1.0,
    max_tokens=16384,
    extra_params={
        "device_map": "auto",
        "torch_dtype": "auto",
        # Use Flash Attention 3 with sink attention (H100/H200)
        "attn_implementation": "kernels-community/vllm-flash-attn3",
    },
)
```

## Memory Requirements

| Model | Precision | Memory | Recommended GPU |
|-------|-----------|--------|-----------------|
| GPT OSS 20B | mxfp4 | 16GB | RTX 3090, 4090, A100 |
| GPT OSS 20B | bf16 | 48GB | A100, H100 |
| GPT OSS 120B | mxfp4 | 80GB | H100, H200 |
| Llama 3.1 8B | 4-bit | 4GB | Any modern GPU |
| Mistral 7B | 4-bit | 4GB | Any modern GPU |
| Qwen 2.5 7B | 4-bit | 4GB | Any modern GPU |

## Optimization Strategies

### 1. Quantization

**4-bit Quantization** (Recommended for most users)
```python
extra_params = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
}
```

**8-bit Quantization** (Better quality, more memory)
```python
extra_params = {
    "load_in_8bit": True,
}
```

**mxfp4** (For GPT OSS models)
- Automatically enabled on compatible GPUs
- Requires `triton>=3.4` and `kernels` library
- Best memory/quality tradeoff

### 2. Attention Optimizations

**Flash Attention 3** (Hopper GPUs: H100, H200)
```python
extra_params = {
    "attn_implementation": "kernels-community/vllm-flash-attn3",
}
```

**MegaBlocks MoE Kernels** (For MoE models without mxfp4)
```python
extra_params = {
    "use_kernels": True,
}
```

### 3. Multi-GPU Support

**Tensor Parallelism** (For large models)
```python
# Run with: torchrun --nproc_per_node=4 your_script.py
extra_params = {
    "device_map": {"tp_plan": "auto"},
}
```

## Running the Example

```bash
# Interactive example
python agents/planning_system/examples/opensource_models_example.py

# Follow the prompts to select a model and generate a plan
```

## Authentication

Some models require Hugging Face authentication:

```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"

# Or set in Python
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"
```

## Performance Comparison

Based on the GPT OSS paper benchmarks:

| Model | Size | Low Complexity | Medium | High | Cost |
|-------|------|----------------|--------|------|------|
| GPT-4o (API) | - | 19.0% | 3.2% | 5.6% | $$$ |
| o1-preview (API) | - | 34.3% | 8.8% | 10.0% | $$$$ |
| GPT OSS 20B (Local) | 21B | ~15% | ~2% | ~3% | Free |
| Llama 3.1 8B (Local) | 8B | ~10% | ~1% | ~2% | Free |

*Note: Performance varies by task. Open-source models excel at cost-effectiveness.*

## Troubleshooting

### Out of Memory

**Solution 1:** Use quantization
```python
extra_params = {"load_in_4bit": True}
```

**Solution 2:** Use a smaller model
```python
# Instead of 70B, use 8B
llm_config = get_llama_3_8b_config(load_in_4bit=True)
```

**Solution 3:** Reduce context/generation length
```python
llm_config = LLMConfig(
    model_name="...",
    max_tokens=4096,  # Reduce from 16384
)
```

### Slow Generation

**Solution 1:** Use Flash Attention (if supported)
```python
extra_params = {"attn_implementation": "flash_attention_2"}
```

**Solution 2:** Use optimized kernels
```python
extra_params = {"use_kernels": True}
```

**Solution 3:** Use smaller model or reduce batch size

### Model Download Fails

**Solution 1:** Check authentication
```bash
huggingface-cli login
```

**Solution 2:** Use mirror (in restricted regions)
```python
export HF_ENDPOINT="https://hf-mirror.com"
```

**Solution 3:** Download manually and load locally
```python
model_name = "/path/to/downloaded/model"
```

## Best Practices

1. **Start Small**: Begin with Llama 3.1 8B (4-bit) before trying larger models
2. **Use Quantization**: 4-bit quantization reduces memory 75% with minimal quality loss
3. **Monitor GPU Memory**: Use `nvidia-smi` to track usage
4. **Cache Models**: Models are cached in `~/.cache/huggingface/` after first download
5. **Batch Processing**: Process multiple problems together for efficiency

## Advanced: Fine-Tuning

You can fine-tune models for your specific ML engineering tasks:

```python
# Use trl library for fine-tuning
from trl import SFTTrainer

# Fine-tune on your ML problem-solving examples
# See: https://github.com/huggingface/trl
```

## Resources

- **Transformers Docs**: https://huggingface.co/docs/transformers
- **Model Hub**: https://huggingface.co/models
- **Quantization Guide**: https://huggingface.co/docs/transformers/quantization
- **Flash Attention**: https://github.com/Dao-AILab/flash-attention

## Summary

Open-source models provide a viable alternative to commercial APIs:

**Use Open-Source When:**
- Privacy is critical
- Budget is limited
- Offline operation needed
- High volume of generations
- Want to fine-tune

**Use Commercial APIs When:**
- Need cutting-edge performance
- Don't have GPU resources
- Want zero setup
- Occasional use only

The agentic planning system supports both seamlessly!
