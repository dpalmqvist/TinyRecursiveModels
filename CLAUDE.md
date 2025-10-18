# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TinyRecursiveModels implements the Tiny Recursion Model (TRM), a recursive reasoning approach that achieves 45% accuracy on ARC-AGI-1 and 8% on ARC-AGI-2 using only 7M parameters. The core insight is that recursive reasoning with small models can solve hard problems without massive foundational models.

**Paper:** "Less is More: Recursive Reasoning with Tiny Networks" (https://arxiv.org/abs/2510.04871)

## Development Commands

### Environment Setup
```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2
wandb login YOUR-LOGIN  # Optional for logging
```

### Dataset Preparation
```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Maze-Hard
python dataset/build_maze_dataset.py
```

### Training

**Single GPU:**
```bash
python pretrain.py arch=trm data_paths="[data/arc1concept-aug-1000]" +run_name=my_experiment ema=True
```

**Multi-GPU (distributed training):**
```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=my_experiment ema=True
```

**Disable compilation (for debugging):**
```bash
DISABLE_COMPILE=1 python pretrain.py ...
```

## Code Architecture

### Core Recursive Reasoning Mechanism

The TRM architecture implements a hierarchical recursive reasoning loop:

1. **Embedding Layer** (`models/recursive_reasoning/trm.py:162-182`):
   - Token embeddings with learned/RoPE positional encodings
   - Optional sparse puzzle embeddings for task-specific conditioning
   - Puzzle embeddings are prepended to sequence before processing

2. **Recursive Loop** (`models/recursive_reasoning/trm.py:196-216`):
   - **H_cycles**: High-level reasoning iterations (outer loop)
   - **L_cycles**: Low-level reasoning iterations (inner loop)
   - Two latent states: `z_H` (high-level) and `z_L` (low-level)
   - Process: For each H_cycle, run L_cycles updating `z_L` using `z_H + input`, then update `z_H` using `z_L`
   - First H_cycles-1 iterations run without gradients for efficiency
   - Final iteration runs with gradients for training

3. **ACT (Adaptive Computation Time) Wrapper** (`models/recursive_reasoning/trm.py:225-297`):
   - Dynamically determines when to halt reasoning per example
   - Q-learning approach with halt/continue signals
   - Exploration during training forces variable computation steps
   - During evaluation, always uses maximum steps for consistent batching

4. **Loss Computation** (`models/losses.py:41-102`):
   - Primary loss: cross-entropy on token predictions (using stablemax or softmax)
   - Q-halt loss: learns when to stop reasoning based on correctness
   - Optional Q-continue loss: bootstrapping target for reinforcement learning

### Key Components

**Models:**
- `models/recursive_reasoning/trm.py`: Main TRM architecture with ACT
- `models/recursive_reasoning/transformers_baseline.py`: Baseline transformer without recursion
- `models/recursive_reasoning/hrm.py`: Original Hierarchical Reasoning Model implementation
- `models/recursive_reasoning/trm_hier6.py`, `trm_singlez.py`: TRM variants
- `models/layers.py`: Core building blocks (Attention, SwiGLU, RoPE, etc.)
- `models/sparse_embedding.py`: Sparse puzzle embeddings with custom optimizer
- `models/ema.py`: Exponential Moving Average for model weights

**Dataset Pipeline:**
- `puzzle_dataset.py`: Main dataset loader with distributed support
- `dataset/common.py`: Shared utilities, dihedral transformations for data augmentation
- `dataset/build_arc_dataset.py`: ARC-AGI dataset builder with augmentation
- `dataset/build_sudoku_dataset.py`, `build_maze_dataset.py`: Other puzzle datasets

**Training:**
- `pretrain.py`: Main training script with distributed support
- Uses Hydra for configuration management (config files in `config/`)
- Supports multi-node distributed training via torchrun
- Checkpoint saving and loading with automatic puzzle embedding resizing
- Integration with Weights & Biases for logging

### Configuration System

Uses Hydra with hierarchical configs:
- `config/cfg_pretrain.yaml`: Base training configuration
- `config/arch/`: Architecture-specific configs (trm.yaml, hrm.yaml, etc.)
- Override parameters via command line: `python pretrain.py arch=trm arch.L_layers=4`

### Important Implementation Details

1. **Puzzle Embeddings:** When loading checkpoints with different number of puzzles, embeddings are automatically resized by replicating the mean embedding (`pretrain.py:252-261`).

2. **Gradient Management:** The recursive loop runs H_cycles-1 iterations without gradients using `torch.no_grad()`, then one final iteration with gradients. This reduces memory and computation while maintaining learning signal.

3. **Distributed Training:** Full support for multi-GPU training with gradient synchronization via NCCL backend. Supports both CPU (GLOO) and GPU process groups.

4. **ACT Exploration:** During training, random exploration forces models to halt at different steps (controlled by `halt_exploration_prob`). This prevents the model from always halting at the same step.

5. **Optimizer Strategy:** Uses two separate optimizers:
   - `CastedSparseEmbeddingSignSGD_Distributed` for puzzle embeddings
   - `AdamATan2` for model parameters
   - Different learning rates and weight decay for each

6. **Data Augmentation:** Datasets apply dihedral transformations (8 symmetries via rotations/flips) to puzzle grids for data augmentation.

7. **Evaluation Mode:** Uses EMA (Exponential Moving Average) weights for evaluation when enabled. Creates a deep copy of the model before evaluation to avoid modifying training state.

## Testing and Evaluation

The codebase uses custom evaluators rather than standard unit tests:
- `evaluators/arc.py`: ARC-specific evaluation metrics
- Evaluators are run during training at intervals specified by `eval_interval`
- Evaluation results are logged to W&B and saved to checkpoint directories

## Model Variants

- **TRM** (default): Full recursive reasoning with both H and L cycles
- **TRM-MLP**: Uses MLP instead of attention for L-level reasoning (`mlp_t=True`)
- **TRM-Hier6**: Hierarchical variant with 6 levels
- **TRM-SingleZ**: Single latent state variant
- **HRM**: Original Hierarchical Reasoning Model baseline
- **Transformers Baseline**: Standard transformer without recursion

## Data Format

Datasets are stored as numpy arrays with memory-mapped loading:
- `inputs`: Input token sequences
- `labels`: Target token sequences
- `puzzle_identifiers`: IDs linking examples to specific puzzles
- `puzzle_indices`, `group_indices`: Indexing structures for efficient sampling

Special tokens:
- `pad_id`: Padding token
- `ignore_label_id`: Labels to ignore in loss computation (converted to -100)
- `blank_identifier_id`: Default puzzle identifier for padding
