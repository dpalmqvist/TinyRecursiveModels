#!/bin/bash
# Quick script to prepare Alpaca dataset with 512 max length for faster training

set -e

echo "Preparing Alpaca-512 dataset for training..."
echo ""

# Check if raw data exists
if [ ! -f "data/raw/alpaca.json" ]; then
    echo "Downloading Alpaca dataset..."
    python download_chat_datasets.py --dataset alpaca --output-dir data/raw
else
    echo "Alpaca raw data already exists"
fi

echo ""
echo "Processing dataset with max_length=512..."
python -m dataset.build_chat_dataset \
  --input-file data/raw/alpaca.json \
  --output-dir data/alpaca-512 \
  --format alpaca \
  --tokenizer gpt2 \
  --max-length 512 \
  --train-split 0.95

echo ""
echo "Dataset ready at data/alpaca-512"
echo "You can now train with:"
echo "  python pretrain.py --config-name cfg_chat"
