# Training on MacBook with MPS (Apple Silicon)

This guide helps you train TinyRecursiveModels on a MacBook with limited GPU memory.

## Memory Issue Solutions

Your Mac has ~27GB of unified memory. The default config tries to use too much. Here are solutions:

### 🚀 Quick Start (Recommended)

Run the automated setup:

```bash
./setup_mps_training.sh
```

Then train:

```bash
source .env_mps
python pretrain.py --config-name mps_training \
  data_paths="[data/dolly-512]" \
  +run_name=mps-chat-model
```

### Manual Setup

If you prefer manual control:

#### 1. Set Memory Environment Variables

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Use all available memory
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7   # Release memory at 70%
```

Or add to your `~/.zshrc` or `~/.bashrc`:

```bash
echo 'export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0' >> ~/.zshrc
echo 'export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7' >> ~/.zshrc
source ~/.zshrc
```

#### 2. Prepare Memory-Efficient Dataset

```bash
# Use shorter sequences (512 instead of 2048)
python -m dataset.build_chat_dataset \
  --input-file data/raw/dolly-15k.json \
  --output-dir data/dolly-512 \
  --format alpaca \
  --tokenizer gpt2 \
  --max-length 512 \
  --train-split 0.95
```

#### 3. Train with Small Model Config

```bash
python pretrain.py \
  arch=trm_small \
  data_paths="[data/dolly-512]" \
  batch_size=2 \
  gradient_accumulation_steps=16 \
  ema=false \
  +run_name=mps-chat
```

## Configuration Trade-offs

| Setting | Default | MPS-Optimized | Impact |
|---------|---------|---------------|--------|
| **seq_len** | 2048 | 512 | 4x less memory, shorter context |
| **batch_size** | 16 | 2 | 8x less memory, use grad accumulation |
| **hidden_size** | 512 | 256 | 4x less memory, smaller model |
| **L_layers** | 4 | 2 | 2x less memory, less capacity |
| **H_cycles** | 3 | 2 | Less recursive reasoning |
| **EMA** | true | false | 2x less memory, no averaging |

## Memory Optimization Levels

### Level 1: Mild (fits 16-32GB)
```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/dolly-512]" \
  batch_size=4 \
  arch.hidden_size=384 \
  arch.L_layers=3 \
  ema=false \
  +run_name=mps-mild
```

### Level 2: Moderate (fits 8-16GB) - **Recommended**
```bash
python pretrain.py \
  arch=trm_small \
  data_paths="[data/dolly-512]" \
  batch_size=2 \
  gradient_accumulation_steps=16 \
  ema=false \
  +run_name=mps-moderate
```

### Level 3: Aggressive (fits 4-8GB)
```bash
# First, create ultra-short sequence dataset
python -m dataset.build_chat_dataset \
  --input-file data/raw/dolly-15k.json \
  --output-dir data/dolly-256 \
  --format alpaca \
  --max-length 256

# Train with minimal config
python pretrain.py \
  arch=trm_small \
  data_paths="[data/dolly-256]" \
  batch_size=1 \
  gradient_accumulation_steps=32 \
  arch.hidden_size=128 \
  arch.L_layers=1 \
  ema=false \
  +run_name=mps-tiny
```

## Monitoring Memory Usage

### Check current memory usage:
```bash
# Monitor in real-time
watch -n 1 'ps aux | grep python | grep pretrain'

# Or use Activity Monitor app
open -a "Activity Monitor"
```

### Check if MPS is being used:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

## Troubleshooting

### Still Out of Memory?

**1. Close other applications:**
- Chrome/Safari (browsers use lots of memory)
- Slack, Discord, etc.
- Docker containers
- Other development tools

**2. Reduce batch size further:**
```yaml
# In config/mps_training.yaml
batch_size: 1
gradient_accumulation_steps: 32  # Simulate batch_size=32
```

**3. Use even smaller sequences:**
```bash
# Train with 256 tokens instead of 512
python -m dataset.build_chat_dataset \
  --input-file data/raw/dolly-15k.json \
  --output-dir data/dolly-256 \
  --format alpaca \
  --max-length 256
```

**4. Reduce model size:**
Edit `config/arch/trm_small.yaml`:
```yaml
hidden_size: 128      # Instead of 256
L_layers: 1           # Instead of 2
```

**5. Disable compilation (if enabled):**
```bash
export DISABLE_COMPILE=1
```

### Training is Slow?

MPS training is typically slower than CUDA. Expected speeds:
- **With MPS:** ~2-5 steps/second (depending on config)
- **With CPU only:** ~0.1-0.5 steps/second
- **With CUDA GPU:** ~10-50 steps/second

To verify MPS is being used:
```python
# Add to pretrain.py or run in Python
import torch
x = torch.tensor([1.0])
print(f"Default device: {x.device}")
x_mps = x.to('mps')
print(f"MPS device: {x_mps.device}")
```

### Checkpoint Too Large?

Reduce checkpoint frequency:
```yaml
# In config/mps_training.yaml
save_interval: 2000      # Instead of 1000
keep_last_n_checkpoints: 2  # Instead of 3
```

## Generation After Training

Use the same memory optimizations for generation:

```bash
source .env_mps

python generate.py \
  --checkpoint checkpoints/mps-chat-model/TinyRecursiveReasoningModel_ACTV1_*/step_5000 \
  --prompt "Explain machine learning" \
  --max-new-tokens 100 \
  --temperature 0.7
```

## Expected Training Time

With MPS-optimized config on M1/M2 Mac:
- **Dolly-15k (14K examples):** ~6-12 hours for 10K steps
- **Alpaca (52K examples):** ~24-48 hours for 10K steps

Use a smaller subset for faster iteration:
```bash
python -m dataset.build_chat_dataset \
  --input-file data/raw/dolly-15k.json \
  --output-dir data/dolly-1k \
  --format alpaca \
  --max-length 512 \
  --subsample-size 1000  # Only use 1000 examples
```

## Summary

**For your Mac with 27GB memory, use:**

```bash
# 1. Setup
./setup_mps_training.sh

# 2. Train
source .env_mps
python pretrain.py --config-name mps_training \
  data_paths="[data/dolly-512]" \
  +run_name=mps-chat

# 3. Generate (after training)
python generate.py \
  --checkpoint checkpoints/mps-chat/TinyRecursiveReasoningModel_ACTV1_*/step_5000 \
  --prompt "Hello!" \
  --interactive
```

This should train successfully without OOM errors! 🎉
