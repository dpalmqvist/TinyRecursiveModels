# MPS (Metal Performance Shaders) Training Guide

## Overview

TinyRecursiveModels now supports training on Apple Silicon (M1/M2/M3/M4) using MPS (Metal Performance Shaders). This enables GPU-accelerated training on Mac without requiring NVIDIA CUDA.

## Requirements

- **Hardware**: Mac with Apple Silicon (M1, M2, M3, or M4)
- **macOS**: 12.3 or later
- **PyTorch**: 1.12 or later with MPS support

## Device Auto-Detection

The training and generation scripts automatically detect the best available device:

1. **CUDA** (if NVIDIA GPU available)
2. **MPS** (if Apple Silicon Mac)
3. **CPU** (fallback)

```python
# Device detection happens automatically in both scripts:
# pretrain.py and generate.py

Using device: mps
```

## Training on MPS

### Basic Training

```bash
# The script automatically detects and uses MPS
python pretrain.py --config-name cfg_chat \
    data_paths=[data/my-chat-dataset]
```

No special flags needed! The code will automatically detect MPS and use it.

### Example Chat Model Training

```bash
# 1. Prepare example dataset
cd examples && ./prepare_example_chat_dataset.sh

# 2. Train on MPS (auto-detected)
python pretrain.py --config-name cfg_chat \
    data_paths=[data/example-chat-dataset] \
    global_batch_size=4 \
    epochs=10
```

## Generation/Inference on MPS

```bash
# Auto-detects MPS
python generate.py \
    --checkpoint checkpoints/.../step_1000 \
    --prompt "Hello!" \
    --max-new-tokens 50

# Or explicitly specify device
python generate.py \
    --checkpoint checkpoints/.../step_1000 \
    --prompt "Hello!" \
    --device mps
```

## MPS-Specific Considerations

### 1. Model Compilation

`torch.compile()` may not work on MPS yet. The code automatically disables it:

```python
# In pretrain.py:
if "DISABLE_COMPILE" not in os.environ and device != "mps":
    model = torch.compile(model)
```

To manually disable compilation on any device:
```bash
DISABLE_COMPILE=1 python pretrain.py --config-name cfg_chat ...
```

### 2. Distributed Training Not Supported

MPS does not support distributed training (multi-GPU). The code will raise an error if you try:

```bash
# This will fail on MPS:
torchrun --nproc-per-node 2 pretrain.py ...

# Error: Distributed training is not supported on MPS
```

**Solution**: Use single-device training on Mac, or use CUDA for multi-GPU.

### 3. Memory Management

Apple Silicon has unified memory (shared between CPU and GPU). Monitor total system memory usage:

```bash
# Check memory usage during training
activity monitor
# Or use:
top -o mem
```

**Tips for reducing memory usage:**

```bash
# Reduce batch size
python pretrain.py --config-name cfg_chat \
    global_batch_size=8 \  # Smaller batch

# Reduce sequence length
python pretrain.py --config-name cfg_chat \
    arch.seq_len=512 \  # Shorter sequences

# Reduce model size
python pretrain.py --config-name cfg_chat \
    arch.hidden_size=512 \
    arch.L_layers=2
```

### 4. Performance Characteristics

| Device | Speed (relative) | Memory | Multi-GPU |
|--------|------------------|--------|-----------|
| CUDA A100 | 10x | 40-80 GB | ✅ Yes |
| CUDA RTX 4090 | 8x | 24 GB | ✅ Yes |
| MPS M3 Max | 1x (baseline) | 32-128 GB | ❌ No |
| MPS M2 | 0.7x | 16-96 GB | ❌ No |
| CPU | 0.1x | System RAM | ❌ No |

**Note**: MPS is significantly faster than CPU but slower than high-end CUDA GPUs.

### 5. Precision Support

MPS supports both float32 and bfloat16:

```yaml
# In config/arch/trm_chat.yaml
forward_dtype: bfloat16  # Works on MPS
```

## Troubleshooting

### Issue: "MPS backend out of memory"

**Solution 1**: Reduce batch size
```bash
python pretrain.py --config-name cfg_chat global_batch_size=2
```

**Solution 2**: Reduce model size
```bash
python pretrain.py --config-name cfg_chat \
    arch.hidden_size=384 \
    arch.L_layers=2
```

**Solution 3**: Reduce sequence length
```bash
python -m dataset.build_chat_dataset \
    --max-length 512 \  # Instead of 2048
    ...
```

### Issue: "MPS is not available"

**Cause**: PyTorch not built with MPS support or macOS too old.

**Solution**:
```bash
# Check MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Update PyTorch
pip install --upgrade torch torchvision torchaudio

# Or install nightly build for latest MPS support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Issue: Slower than expected

**Possible causes**:
1. Model too small (GPU underutilized)
2. Disk I/O bottleneck (use SSD)
3. Memory swapping (reduce batch size)

**Solutions**:
```bash
# Increase model utilization
python pretrain.py --config-name cfg_chat \
    global_batch_size=16 \  # Larger batch if memory allows
    arch.hidden_size=768 \  # Larger model
    arch.L_layers=4

# Use SSD for data storage (not external HDD)
# Check Activity Monitor for memory pressure
```

### Issue: Training crashes without error

**Cause**: Memory exhaustion.

**Solution**: Reduce memory usage (see Memory Management section above).

## Performance Tips

### 1. Optimal Batch Size

Find the largest batch size that fits in memory:

```bash
# Start small and increase
for bs in 2 4 8 16 32; do
    echo "Testing batch size: $bs"
    python pretrain.py --config-name cfg_chat \
        global_batch_size=$bs \
        epochs=1 \
        || break
done
```

### 2. Model Size vs Speed

| Configuration | Parameters | Speed | Memory |
|---------------|------------|-------|--------|
| Tiny | ~7M | Fast | ~2 GB |
| Small | ~30M | Medium | ~6 GB |
| Medium | ~80M | Slow | ~12 GB |
| Large | ~150M | Very Slow | ~20 GB |

**Recommendation for MPS**: Start with Small (30M) config:

```yaml
hidden_size: 768
L_layers: 4
num_heads: 12
```

### 3. Data Loading

```bash
# Reduce num_workers for MPS (unified memory)
# This is handled automatically in the code

# Ensure data is on fast storage (SSD)
mv data/ /path/to/ssd/data/
```

### 4. Mixed Precision

```yaml
# Use bfloat16 for faster training
forward_dtype: bfloat16  # Default in trm_chat.yaml
```

## Comparison with CUDA

### Advantages of MPS

✅ No external GPU required
✅ Large unified memory (up to 128 GB on M3 Max)
✅ Lower power consumption
✅ Silent operation (no fans)
✅ Portable (laptop training)

### Disadvantages of MPS

❌ Slower than high-end CUDA GPUs
❌ No distributed training support
❌ Fewer optimized kernels
❌ Less mature ecosystem

## Best Practices

### For Development/Prototyping

```bash
# Use MPS for rapid iteration
python pretrain.py --config-name cfg_chat \
    data_paths=[data/small-dataset] \
    global_batch_size=8 \
    epochs=10 \
    arch.hidden_size=512 \
    arch.L_layers=2
```

### For Production Training

```bash
# Use CUDA for final training runs
# Train on cloud GPU (A100, H100) or local CUDA GPU
# Then load checkpoint on Mac for testing/deployment
```

### Hybrid Workflow

1. **Develop on Mac (MPS)**: Fast iteration, model debugging
2. **Train on CUDA**: Scale up for production
3. **Deploy anywhere**: Load checkpoints on any device

```bash
# On Mac (MPS)
python pretrain.py --config-name cfg_chat \
    data_paths=[data/dev-dataset] \
    epochs=10

# On Cloud (CUDA)
python pretrain.py --config-name cfg_chat \
    data_paths=[data/full-dataset] \
    epochs=100 \
    global_batch_size=128

# Load checkpoint anywhere
python generate.py \
    --checkpoint checkpoints/.../step_10000 \
    --device mps  # or cuda, or cpu
```

## Example: Complete Training on M3 Max

```bash
# System: MacBook Pro M3 Max, 96GB RAM

# 1. Prepare dataset
python -m dataset.build_chat_dataset \
    --input-file data/conversations.json \
    --output-dir data/chat-mps \
    --tokenizer gpt2 \
    --max-length 1024 \
    --subsample-size 10000

# 2. Train with optimal settings for MPS
python pretrain.py --config-name cfg_chat \
    data_paths=[data/chat-mps] \
    global_batch_size=16 \
    epochs=50 \
    arch.hidden_size=768 \
    arch.L_layers=4 \
    lr=3e-4 \
    ema=True

# Expected training speed:
# ~5-10 steps/second (depending on sequence length)
# ~30M parameter model
# ~12-16 GB memory usage

# 3. Generate responses
python generate.py \
    --checkpoint checkpoints/.../step_5000 \
    --interactive
```

## Monitoring

```bash
# Terminal 1: Training
python pretrain.py --config-name cfg_chat ...

# Terminal 2: Monitor GPU
while true; do
    python -c "
import torch
if torch.backends.mps.is_available():
    print(f'MPS is active')
"
    sleep 5
done

# Terminal 3: Monitor memory
top -o mem | head -20
```

## Summary

- ✅ **MPS support is automatic** - no special flags needed
- ✅ **Works out of the box** on Apple Silicon Macs
- ⚠️ **Single device only** - no distributed training
- ⚠️ **Slower than CUDA** - but much faster than CPU
- 💡 **Great for development** - perfect for prototyping and testing

For questions or issues with MPS, check:
- PyTorch MPS documentation
- Apple Metal documentation
- TinyRecursiveModels issues page
