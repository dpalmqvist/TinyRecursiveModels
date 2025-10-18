#!/bin/bash

# Script to prepare the example chat dataset for training
# This creates a small dataset for testing the chat model training pipeline

set -e  # Exit on error

echo "========================================="
echo "Preparing Example Chat Dataset"
echo "========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent of examples/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"
echo "Working from: $PROJECT_ROOT"
echo ""

# Check if transformers is installed
python -c "import transformers" 2>/dev/null || {
    echo "Error: transformers library not found. Installing..."
    pip install transformers
}

# Build the dataset
echo ""
echo "Building chat dataset from examples/example_chat_data.json..."
python -m dataset.build_chat_dataset \
    --input-file examples/example_chat_data.json \
    --output-dir data/example-chat-dataset \
    --format sharegpt \
    --tokenizer gpt2 \
    --max-length 512 \
    --train-split 0.8 \
    --seed 42

echo ""
echo "========================================="
echo "Dataset preparation complete!"
echo "========================================="
echo ""
echo "Dataset saved to: data/example-chat-dataset/"
echo ""
echo "To train the model, run:"
echo "  python pretrain.py --config-name cfg_chat data_paths=[data/example-chat-dataset]"
echo ""
echo "Or for quick testing with minimal config:"
echo "  python pretrain.py --config-name cfg_chat \\"
echo "    data_paths=[data/example-chat-dataset] \\"
echo "    global_batch_size=2 \\"
echo "    epochs=10 \\"
echo "    eval_interval=5"
echo ""
