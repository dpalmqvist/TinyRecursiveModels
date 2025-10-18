# Chat Model Quick Start

## TL;DR - Get Started in 3 Steps

> **💡 Apple Silicon Users**: Training automatically uses MPS (Metal Performance Shaders) for GPU acceleration on M1/M2/M3/M4 Macs. See [MPS_GUIDE.md](MPS_GUIDE.md) for details.

### 1. Install Dependencies

```bash
pip install transformers
```

### 2. Prepare Example Dataset

```bash
cd examples
./prepare_example_chat_dataset.sh
```

### 3. Train & Test

```bash
# Train (quick test with small config)
python pretrain.py --config-name cfg_chat \
    data_paths=[data/example-chat-dataset] \
    global_batch_size=2 \
    epochs=10 \
    eval_interval=5

# Generate (after training)
python generate.py \
    --checkpoint checkpoints/Example-chat-dataset-ACT-torch/trm_chat*/step_* \
    --prompt "What is machine learning?" \
    --max-new-tokens 50
```

## Files Overview

### New Files for Chat Training

```
TinyRecursiveModels/
├── chat_dataset.py                          # Chat dataset loader
├── generate.py                              # Inference/generation script
├── CHAT_MODEL_GUIDE.md                      # Comprehensive guide
├── CHAT_QUICKSTART.md                       # This file
├── config/
│   ├── arch/trm_chat.yaml                   # Chat model architecture
│   └── cfg_chat.yaml                        # Chat training config
├── dataset/
│   └── build_chat_dataset.py                # Dataset builder
├── examples/
│   ├── example_chat_data.json               # Sample conversations
│   └── prepare_example_chat_dataset.sh      # Setup script
└── utils/
    └── tokenizer.py                         # Tokenizer utilities
```

### Modified Files

```
models/recursive_reasoning/trm.py            # Added causal attention support
pretrain.py                                  # Added chat dataset support
```

## Key Configuration Changes

### For Chat (vs Puzzle-Solving)

| Setting | Puzzle | Chat | Why? |
|---------|--------|------|------|
| `causal` | False | **True** | Autoregressive generation |
| `puzzle_emb_ndim` | 512 | **0** | No per-conversation embeddings |
| `halt_max_steps` | 16 | **1** | Disable ACT for simplicity |
| `H_cycles` | 3 | **1** | Standard forward pass |
| `L_cycles` | 6 | **1** | Standard forward pass |
| `dataset_type` | puzzle | **chat** | Use chat loader |
| `global_batch_size` | 768 | **32** | Longer sequences |
| `max_length` | 900 | **2048** | Full conversations |

## Common Commands

### Prepare Dataset

```bash
# ShareGPT format
python -m dataset.build_chat_dataset \
    --input-file data.json \
    --output-dir data/my-dataset \
    --format sharegpt \
    --tokenizer gpt2 \
    --max-length 2048
```

### Train

```bash
# Single GPU
python pretrain.py --config-name cfg_chat \
    data_paths=[data/my-dataset]

# Multi-GPU
torchrun --nproc-per-node 4 pretrain.py \
    --config-name cfg_chat \
    data_paths=[data/my-dataset] \
    global_batch_size=128
```

### Generate

```bash
# Single prompt
python generate.py \
    --checkpoint checkpoints/.../step_1000 \
    --prompt "Your question here"

# Interactive chat
python generate.py \
    --checkpoint checkpoints/.../step_1000 \
    --interactive
```

## Troubleshooting

**Q: ImportError: No module named 'transformers'**
A: `pip install transformers`

**Q: CUDA out of memory**
A: Reduce `global_batch_size` or use gradient accumulation

**Q: Model not generating coherent text**
A: Train longer, use more data, or increase model size

**Q: How do I use my own data?**
A: Format as ShareGPT JSON (see `examples/example_chat_data.json`) and run `build_chat_dataset.py`

**Q: Can I use recursive reasoning for chat?**
A: Yes! Set `H_cycles=3`, `L_cycles=4`, `halt_max_steps=16` in config. May improve complex reasoning.

## What Changed vs Original Codebase?

### Additions ✨
- Chat dataset builder and loader
- Generation/inference script
- Causal attention support in TRM
- Tokenizer integration
- Chat-specific configs

### Modifications 🔧
- `trm.py`: Added `causal` parameter
- `pretrain.py`: Added chat dataset support

### Preserved ✅
- Training loop
- Loss computation (works for both)
- Recursive reasoning architecture
- Multi-GPU training
- All original puzzle functionality

## Next Steps

1. Read [CHAT_MODEL_GUIDE.md](CHAT_MODEL_GUIDE.md) for detailed documentation
2. Prepare your own dataset
3. Experiment with model architecture (recursive reasoning on/off)
4. Scale up training (more data, bigger model, multi-GPU)

## Support

- Original Paper: https://arxiv.org/abs/2510.04871
- Questions? Check CHAT_MODEL_GUIDE.md or file an issue
