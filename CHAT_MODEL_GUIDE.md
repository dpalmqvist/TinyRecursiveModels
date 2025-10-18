# Chat Model Training Guide

This guide explains how to use TinyRecursiveModels to train a conversational chat model instead of puzzle-solving.

## Overview

The repository has been adapted to support training causal language models for chat/conversation tasks. The key changes include:

- **Causal Attention**: Models now support autoregressive generation
- **Chat Dataset Loader**: Process conversational data in various formats
- **Generation Pipeline**: Inference script for interactive chat
- **Flexible Architecture**: Recursive reasoning can be preserved or disabled
- **Multi-Device Support**: CUDA, MPS (Apple Silicon), and CPU

> **🍎 Apple Silicon Users**: See [MPS_GUIDE.md](MPS_GUIDE.md) for Mac-specific training instructions.

## Quick Start

### 1. Prepare Your Chat Dataset

#### Option A: Use the Example Dataset (for testing)

```bash
cd examples
./prepare_example_chat_dataset.sh
```

This creates a tiny dataset in `data/example-chat-dataset/` with 5 conversations.

#### Option B: Build from Your Own Data

Supported formats:
- **ShareGPT**: Multi-turn conversations with `from` and `value` fields
- **Alpaca**: Instruction-following with `instruction`, `input`, `output` fields
- **OpenAssistant**: Conversations with `messages` containing `role` and `content`

Example command:

```bash
python -m dataset.build_chat_dataset \
    --input-file path/to/your/data.json \
    --output-dir data/my-chat-dataset \
    --format sharegpt \
    --tokenizer gpt2 \
    --max-length 2048 \
    --train-split 0.95
```

**Parameters:**
- `--input-file`: Path to your conversation data (JSON format)
- `--output-dir`: Where to save the processed dataset
- `--format`: Data format (`sharegpt`, `alpaca`, or `openassistant`)
- `--tokenizer`: HuggingFace tokenizer to use (e.g., `gpt2`, `facebook/opt-125m`)
- `--max-length`: Maximum sequence length (default: 2048)
- `--train-split`: Fraction of data for training (default: 0.95)
- `--subsample-size`: Optional, limit dataset size for testing

### 2. Train the Model

#### Basic Training

```bash
python pretrain.py --config-name cfg_chat \
    data_paths=[data/my-chat-dataset]
```

#### Custom Training Configuration

```bash
python pretrain.py --config-name cfg_chat \
    data_paths=[data/my-chat-dataset] \
    global_batch_size=64 \
    arch.hidden_size=512 \
    arch.L_layers=4 \
    lr=3e-4 \
    epochs=100
```

#### Multi-GPU Training

```bash
torchrun --nproc-per-node 4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    pretrain.py --config-name cfg_chat \
    data_paths=[data/my-chat-dataset] \
    global_batch_size=128
```

### 3. Generate Responses

#### Single Generation

```bash
python generate.py \
    --checkpoint checkpoints/Chat-dataset-ACT-torch/trm_chat-*/step_1000 \
    --prompt "What is recursion?" \
    --max-new-tokens 100 \
    --temperature 0.7
```

#### Interactive Chat

```bash
python generate.py \
    --checkpoint checkpoints/Chat-dataset-ACT-torch/trm_chat-*/step_1000 \
    --interactive
```

## Architecture Configuration

### Chat-Specific Settings (`config/arch/trm_chat.yaml`)

Key parameters for chat models:

```yaml
# Model size
hidden_size: 768        # Embedding dimension
num_heads: 12           # Attention heads
L_layers: 4             # Number of transformer layers

# Recursive reasoning (optional)
H_cycles: 1             # High-level reasoning iterations (1 = disabled)
L_cycles: 1             # Low-level reasoning iterations (1 = disabled)

# CRITICAL: Enable causal attention
causal: True            # Must be True for chat

# ACT (Adaptive Computation Time)
halt_max_steps: 1       # Set to 1 to disable ACT
halt_exploration_prob: 0.0

# Other settings
puzzle_emb_ndim: 0      # Disabled for chat
pos_encodings: rope     # RoPE positional encodings
forward_dtype: bfloat16
```

### Training Configuration (`config/cfg_chat.yaml`)

```yaml
dataset_type: "chat"    # Use chat dataset loader
data_paths: ['data/chat-dataset']

# Training hyperparameters
global_batch_size: 32   # Adjust based on GPU memory
epochs: 100
eval_interval: 10

# Learning rate schedule
lr: 3e-4                # Peak learning rate
lr_min_ratio: 0.1       # Final LR = lr * lr_min_ratio
lr_warmup_steps: 1000

# Optimizer (AdamATan2)
beta1: 0.9
beta2: 0.95
weight_decay: 0.1

# EMA for better evaluation
ema: True
ema_rate: 0.999
```

## Model Variants

### Standard Transformer (Recommended for Chat)

Disable recursive reasoning for a standard causal transformer:

```yaml
H_cycles: 1
L_cycles: 1
halt_max_steps: 1
```

### Recursive Reasoning for Chain-of-Thought

Enable recursive reasoning to allow the model to "think" multiple times:

```yaml
H_cycles: 3             # Outer reasoning loops
L_cycles: 4             # Inner reasoning loops per H_cycle
halt_max_steps: 16      # Enable adaptive computation
halt_exploration_prob: 0.1
```

This can potentially improve response quality for complex questions.

## Data Format Details

### ShareGPT Format

```json
[
  {
    "id": "conversation_1",
    "conversations": [
      {"from": "user", "value": "Hello!"},
      {"from": "assistant", "value": "Hi! How can I help?"},
      {"from": "user", "value": "What is AI?"},
      {"from": "assistant", "value": "AI stands for..."}
    ]
  }
]
```

### Alpaca Format

```json
[
  {
    "instruction": "Explain quantum computing",
    "input": "",
    "output": "Quantum computing is..."
  }
]
```

### Processed Format

The dataset builder converts conversations to token sequences with special markers:

```
<|user|>Hello!<|enduser|><|sep|><|assistant|>Hi!<|endassistant|><|endoftext|>
```

## Special Tokens

The following special tokens are added during dataset preparation:

- `<|user|>` / `<|enduser|>`: Mark user messages
- `<|assistant|>` / `<|endassistant|>`: Mark assistant responses
- `<|system|>` / `<|endsystem|>`: Mark system prompts (optional)
- `<|sep|>`: Separator between turns
- `<|endoftext|>`: End of conversation
- `<|pad|>`: Padding token

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or sequence length:

```bash
python pretrain.py --config-name cfg_chat \
    global_batch_size=16 \
    arch.seq_len=1024
```

### Slow Training

Enable compilation (default) and use bfloat16:

```bash
# Compilation is on by default
# To disable for debugging:
DISABLE_COMPILE=1 python pretrain.py --config-name cfg_chat
```

### Poor Generation Quality

1. Train longer (more epochs)
2. Increase model size (hidden_size, L_layers)
3. Use a better tokenizer
4. Collect more/better training data
5. Adjust sampling parameters (temperature, top_k, top_p)

### Loss Not Decreasing

1. Check learning rate (try 1e-4 to 5e-4)
2. Verify causal attention is enabled (`causal: True`)
3. Check dataset quality and preprocessing
4. Ensure labels are properly formatted (not all -100)

## Advanced Topics

### Custom Tokenizers

Use a different tokenizer during dataset building:

```bash
python -m dataset.build_chat_dataset \
    --tokenizer facebook/opt-125m \
    # or
    --tokenizer EleutherAI/gpt-neo-125M \
    # or path to saved tokenizer
    --tokenizer ./my_tokenizer/
```

### Instruction Masking

Currently, the model trains on all tokens. To train only on assistant responses, modify `dataset/build_chat_dataset.py`:

```python
# In format_conversation_for_training():
# Set labels to -100 for user tokens
# Keep labels for assistant tokens only
```

### Model Scaling

Approximate parameter counts:

| hidden_size | L_layers | Parameters |
|-------------|----------|------------|
| 512         | 2        | ~7M        |
| 768         | 4        | ~30M       |
| 1024        | 6        | ~80M       |
| 1024        | 12       | ~150M      |

Adjust based on your GPU memory:
- 8GB: ~30M parameters max
- 16GB: ~80M parameters
- 24GB+: 150M+ parameters

### Distributed Training

For multiple nodes:

```bash
# On each node:
torchrun --nproc-per-node 8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=<master_addr>:<master_port> \
    --nnodes=<total_nodes> \
    --node_rank=<this_node_rank> \
    pretrain.py --config-name cfg_chat \
    data_paths=[data/my-chat-dataset] \
    global_batch_size=512
```

## Next Steps

1. **Fine-tune for specific tasks**: Start with a pre-trained checkpoint and continue training on domain-specific data
2. **Experiment with recursive reasoning**: Try different H_cycles/L_cycles values
3. **Implement RLHF**: Add reinforcement learning from human feedback
4. **Create custom evaluators**: Implement chat-specific metrics in `evaluators/`

## References

- Original Paper: "Less is More: Recursive Reasoning with Tiny Networks" ([arxiv.org/abs/2510.04871](https://arxiv.org/abs/2510.04871))
- Original use case: ARC-AGI puzzle solving
- Adapted for: Conversational AI / Chat models
