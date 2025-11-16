# GPU Memory Management for Open-Source Models

## Problem: CUDA Out of Memory

When running with `--model-provider opensource`, you might encounter:
```
CUDA out of memory. Tried to allocate X MiB. GPU has X GiB total, Y MiB free.
```

## Solutions

### Solution 1: Use Single Model (IMPLEMENTED ✓)

**All agents now use the same model** to avoid loading multiple models into GPU memory.

With `--model-provider opensource`, every agent uses:
- **Llama 3.1 8B Instruct** with 4-bit quantization
- **Memory requirement**: ~5GB GPU RAM
- **Benefit**: Only one model loaded, regardless of number of agents

### Solution 2: Force CPU Execution (Slow but Works)

If GPU memory is full or unavailable, use CPU mode:

```bash
# Set environment variable to force CPU mode
export LLM_FORCE_CPU=true

# Run with opensource models on CPU
python agents/planning_system/run.py run <competition> \
  --model-provider opensource \
  --data-dir /path/to/data
```

**Trade-offs:**
- ✓ Uses system RAM instead of GPU memory
- ✓ Works even with no GPU or full GPU
- ✗ Much slower inference (~10-50x slower depending on CPU)

### Solution 3: Clear GPU Memory from Other Processes

Check what's using GPU memory:
```bash
nvidia-smi
```

If another process is using GPU:
```bash
# Kill specific process
kill <PID>

# Or restart Python kernel if in Jupyter
# Kernel → Restart Kernel
```

### Solution 4: Use Smaller Model (Future)

For very limited memory scenarios, you can modify the config to use smaller models:

```python
# In agents/planning_system/core/llm.py
# Change get_oss_default_config() to use:
# - Llama 3.2 3B (~2GB with 4-bit)
# - Qwen 2.5 1.5B (~1GB with 4-bit)
```

## Memory Requirements

| Model | Full Precision | 8-bit | 4-bit |
|-------|---------------|-------|-------|
| Llama 3.1 8B | ~16GB | ~8GB | ~5GB |
| Mistral 7B | ~14GB | ~7GB | ~4GB |
| Llama 3.2 3B | ~6GB | ~3GB | ~2GB |

## Recommended Setup

**If you have ~10GB+ free GPU memory:**
```bash
# Use GPU with 4-bit quantization (default)
python agents/planning_system/run.py run <competition> \
  --model-provider opensource \
  --data-dir /path/to/data
```

**If GPU is full but you have 16GB+ system RAM:**
```bash
# Use CPU mode
export LLM_FORCE_CPU=true
python agents/planning_system/run.py run <competition> \
  --model-provider opensource \
  --data-dir /path/to/data
```

## Debugging

Check GPU memory usage:
```bash
nvidia-smi

# Or with watch to monitor continuously
watch -n 1 nvidia-smi
```

Check current PyTorch memory allocation:
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

Clear PyTorch cache:
```python
import torch
torch.cuda.empty_cache()
```
