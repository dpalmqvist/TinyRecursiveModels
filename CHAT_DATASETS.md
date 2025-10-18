# Chat Dataset Setup Guide

This guide shows you how to download and prepare standard chat datasets for training your TinyRecursiveModels chat model.

## Quick Start

### 1. Install Required Dependencies

```bash
pip install datasets  # Hugging Face datasets library
```

### 2. List Available Datasets

```bash
python download_chat_datasets.py --list
```

### 3. Download a Dataset

**Option A: Download Alpaca (recommended for getting started - 52K examples)**
```bash
python download_chat_datasets.py --dataset alpaca --output-dir data/raw
```

**Option B: Download Dolly-15k (high quality, smaller size)**
```bash
python download_chat_datasets.py --dataset dolly --output-dir data/raw
```

**Option C: Download OpenAssistant (larger, multi-turn conversations)**
```bash
python download_chat_datasets.py --dataset oasst1 --output-dir data/raw
```

**Option D: Download all datasets**
```bash
python download_chat_datasets.py --all --output-dir data/raw
```

### 4. Process Dataset for Training

After downloading, convert to training format:

```bash
# Process Alpaca
python -m dataset.build_chat_dataset \
  --input-file data/raw/alpaca.json \
  --output-dir data/alpaca-processed \
  --format alpaca \
  --tokenizer gpt2 \
  --max-length 2048 \
  --train-split 0.95
```

For faster testing with a subset:
```bash
python -m dataset.build_chat_dataset \
  --input-file data/raw/alpaca.json \
  --output-dir data/alpaca-1k \
  --format alpaca \
  --tokenizer gpt2 \
  --max-length 2048 \
  --train-split 0.95 \
  --subsample-size 1000
```

### 5. Train Your Model

```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/alpaca-processed]" \
  +run_name=chat-alpaca \
  ema=True
```

For multi-GPU training:
```bash
torchrun --nproc-per-node 4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/alpaca-processed]" \
  +run_name=chat-alpaca \
  ema=True
```

### 6. Test Your Model

```bash
python generate.py \
  --checkpoint checkpoints/chat-alpaca/TinyRecursiveReasoningModel_ACTV1_*/step_5000 \
  --prompt "What is machine learning?" \
  --max-new-tokens 100 \
  --temperature 0.7
```

Or use interactive mode:
```bash
python generate.py \
  --checkpoint checkpoints/chat-alpaca/TinyRecursiveReasoningModel_ACTV1_*/step_5000 \
  --interactive
```

## Dataset Comparison

| Dataset | Size | Format | Best For | Quality |
|---------|------|--------|----------|---------|
| **Alpaca** | 52K | Single-turn Q&A | Getting started, general instruction following | Good |
| **Dolly-15k** | 15K | Single-turn Q&A | High-quality responses, diverse tasks | Excellent |
| **OpenAssistant** | 161K msgs | Multi-turn conversations | Natural conversations, dialogue | Excellent |

## Recommended Training Strategy

### For Quick Experiments (1-2 hours on GPU)
```bash
# Use Dolly-15k with 1K subset
python -m dataset.build_chat_dataset \
  --input-file data/raw/dolly-15k.json \
  --output-dir data/dolly-1k \
  --format alpaca \
  --tokenizer gpt2 \
  --max-length 512 \
  --subsample-size 1000

# Train with smaller model
python pretrain.py \
  arch=trm \
  data_paths="[data/dolly-1k]" \
  arch.L_layers=2 \
  arch.H_cycles=2 \
  arch.L_cycles=3 \
  +run_name=quick-test
```

### For Best Results (several hours to days)
```bash
# Use full Alpaca dataset
python -m dataset.build_chat_dataset \
  --input-file data/raw/alpaca.json \
  --output-dir data/alpaca-full \
  --format alpaca \
  --tokenizer gpt2 \
  --max-length 2048

# Train with full model
python pretrain.py \
  arch=trm \
  data_paths="[data/alpaca-full]" \
  arch.L_layers=4 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  +run_name=chat-production \
  ema=True
```

## Using Your Own Dataset

If you have your own chat data, format it as JSON in one of these formats:

### ShareGPT Format
```json
[
  {
    "id": "conv_1",
    "conversations": [
      {"from": "human", "value": "Hello!"},
      {"from": "gpt", "value": "Hi! How can I help you?"},
      {"from": "human", "value": "What is 2+2?"},
      {"from": "gpt", "value": "2+2 equals 4."}
    ]
  }
]
```

### Alpaca Format
```json
[
  {
    "instruction": "Explain machine learning",
    "input": "",
    "output": "Machine learning is a subset of AI..."
  }
]
```

### OpenAssistant Format
```json
[
  {
    "messages": [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi there!"}
    ]
  }
]
```

Then process with:
```bash
python -m dataset.build_chat_dataset \
  --input-file your_data.json \
  --output-dir data/custom-dataset \
  --format [sharegpt|alpaca|openassistant] \
  --tokenizer gpt2 \
  --max-length 2048
```

## Troubleshooting

### Out of Memory
- Reduce `--max-length` (e.g., 512 or 1024 instead of 2048)
- Use `--subsample-size` to train on fewer examples
- Reduce model size with smaller `arch.L_layers`

### Slow Training
- Use multi-GPU training with `torchrun`
- Reduce `arch.H_cycles` and `arch.L_cycles`
- Use a smaller dataset (Dolly-15k instead of full Alpaca)

### Poor Generation Quality
- Train for more steps (at least 5,000+)
- Use lower temperature during generation (`--temperature 0.3`)
- Try a different dataset (Dolly-15k has higher quality)
- Ensure you're using a checkpoint after sufficient training

## Next Steps

After training on these datasets, you can:
1. Fine-tune on domain-specific data
2. Combine multiple datasets for better coverage
3. Use the model for evaluation on your own tasks
4. Experiment with different model architectures

For more information, see the main README and CLAUDE.md files.
