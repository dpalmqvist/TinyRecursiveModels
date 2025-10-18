#!/bin/bash
# Setup script for training on MacBook with MPS (Apple Silicon GPU)
# This script prepares a memory-efficient dataset and training configuration

set -e  # Exit on error

echo "================================================"
echo "MPS Training Setup for TinyRecursiveModels"
echo "================================================"
echo ""

# Check if datasets library is installed
if ! python -c "import datasets" 2>/dev/null; then
    echo "Installing datasets library..."
    pip install datasets
fi

# 1. Download dataset (using smaller Dolly-15k for faster setup)
echo "Step 1: Downloading Dolly-15k dataset..."
if [ ! -f "data/raw/dolly-15k.json" ]; then
    python download_chat_datasets.py --dataset dolly --output-dir data/raw
else
    echo "Dataset already downloaded, skipping..."
fi

# 2. Process dataset with reduced sequence length
echo ""
echo "Step 2: Processing dataset with seq_len=512 for memory efficiency..."
if [ ! -d "data/dolly-512" ]; then
    python -m dataset.build_chat_dataset \
      --input-file data/raw/dolly-15k.json \
      --output-dir data/dolly-512 \
      --format alpaca \
      --tokenizer gpt2 \
      --max-length 512 \
      --train-split 0.95
else
    echo "Dataset already processed, skipping..."
fi

# 3. Set MPS environment variables
echo ""
echo "Step 3: Setting MPS environment variables..."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Allow using all available memory
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7   # Start releasing memory at 70%

# Save to a file for future use
cat > .env_mps << 'EOF'
# MPS Memory Management Settings
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7

# Optional: Disable compilation for debugging
# export DISABLE_COMPILE=1
EOF

echo "MPS environment variables saved to .env_mps"
echo "Run 'source .env_mps' before training in new terminal sessions"

# 4. Print training command
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To start training, run:"
echo ""
echo "  source .env_mps"
echo "  python pretrain.py --config-name mps_training \\"
echo "    data_paths=\"[data/dolly-512]\" \\"
echo "    +run_name=mps-chat-model"
echo ""
echo "Monitor memory usage with:"
echo "  watch -n 1 'ps aux | grep python'"
echo ""
echo "If you still run out of memory, try:"
echo "  - Reduce batch_size to 1 in config/mps_training.yaml"
echo "  - Reduce hidden_size to 128 in config/arch/trm_small.yaml"
echo "  - Use max-length 256 instead of 512"
echo ""
