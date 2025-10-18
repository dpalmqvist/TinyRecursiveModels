"""
Download and prepare popular chat datasets for training.

This script downloads datasets from Hugging Face and converts them
to the format expected by build_chat_dataset.py

Usage:
    # Download a specific dataset
    python download_chat_datasets.py --dataset alpaca --output-dir data/raw

    # Download all datasets
    python download_chat_datasets.py --all --output-dir data/raw
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

try:
    from datasets import load_dataset
except ImportError:
    print("Please install the datasets library: pip install datasets")
    exit(1)


AVAILABLE_DATASETS = {
    "alpaca": {
        "hf_path": "tatsu-lab/alpaca",
        "format": "alpaca",
        "size": "52K examples",
        "description": "Stanford Alpaca instruction-following dataset"
    },
    "dolly": {
        "hf_path": "databricks/databricks-dolly-15k",
        "format": "alpaca",  # Similar format
        "size": "15K examples",
        "description": "Databricks Dolly-15k high-quality human-generated examples"
    },
    "oasst1": {
        "hf_path": "OpenAssistant/oasst1",
        "format": "openassistant",
        "size": "~161K messages",
        "description": "OpenAssistant Conversations - high-quality human conversations"
    },
}


def download_alpaca(output_dir: str):
    """Download and convert Alpaca dataset"""
    print("Downloading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca")

    # Convert to our format
    data = []
    for item in dataset["train"]:
        data.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"]
        })

    output_file = os.path.join(output_dir, "alpaca.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} examples to {output_file}")
    return output_file


def download_dolly(output_dir: str):
    """Download and convert Dolly-15k dataset"""
    print("Downloading Dolly-15k dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k")

    # Convert to Alpaca format
    data = []
    for item in dataset["train"]:
        data.append({
            "instruction": item["instruction"],
            "input": item.get("context", ""),
            "output": item["response"]
        })

    output_file = os.path.join(output_dir, "dolly-15k.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} examples to {output_file}")
    return output_file


def download_oasst1(output_dir: str):
    """Download and convert OpenAssistant dataset"""
    print("Downloading OpenAssistant dataset...")
    dataset = load_dataset("OpenAssistant/oasst1")

    # OASST1 is organized as message trees, need to reconstruct conversations
    # This is a simplified version - for production use, properly reconstruct conversation trees

    conversations = {}

    for split in ["train", "validation"]:
        if split not in dataset:
            continue

        for item in dataset[split]:
            message_tree_id = item.get("message_tree_id")
            parent_id = item.get("parent_id")
            message_id = item.get("message_id")
            role = item.get("role")  # "assistant" or "prompter"
            text = item.get("text", "")

            if message_tree_id not in conversations:
                conversations[message_tree_id] = []

            # Map role
            if role == "prompter":
                role = "user"
            elif role == "assistant":
                role = "assistant"

            conversations[message_tree_id].append({
                "message_id": message_id,
                "parent_id": parent_id,
                "role": role,
                "content": text
            })

    # Convert to simple format (flatten conversations)
    data = []
    for conv_id, messages in conversations.items():
        # Sort by parent relationship (simplified - just take first few turns)
        conversation_messages = []
        for msg in messages[:10]:  # Limit conversation length
            conversation_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        if conversation_messages:
            data.append({
                "id": conv_id,
                "messages": conversation_messages
            })

    output_file = os.path.join(output_dir, "oasst1.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} conversations to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Download chat datasets for training")
    parser.add_argument("--dataset", choices=list(AVAILABLE_DATASETS.keys()),
                       help="Which dataset to download")
    parser.add_argument("--all", action="store_true",
                       help="Download all available datasets")
    parser.add_argument("--output-dir", default="data/raw",
                       help="Directory to save raw datasets")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:\n")
        for name, info in AVAILABLE_DATASETS.items():
            print(f"  {name}:")
            print(f"    HuggingFace: {info['hf_path']}")
            print(f"    Size: {info['size']}")
            print(f"    Format: {info['format']}")
            print(f"    Description: {info['description']}")
            print()
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    downloaded_files = []

    if args.all:
        datasets_to_download = list(AVAILABLE_DATASETS.keys())
    elif args.dataset:
        datasets_to_download = [args.dataset]
    else:
        parser.print_help()
        return

    print(f"\nDownloading datasets: {', '.join(datasets_to_download)}\n")

    for dataset_name in datasets_to_download:
        try:
            if dataset_name == "alpaca":
                file_path = download_alpaca(args.output_dir)
                downloaded_files.append((file_path, "alpaca"))
            elif dataset_name == "dolly":
                file_path = download_dolly(args.output_dir)
                downloaded_files.append((file_path, "alpaca"))
            elif dataset_name == "oasst1":
                file_path = download_oasst1(args.output_dir)
                downloaded_files.append((file_path, "openassistant"))

            print()
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}\n")

    # Print next steps
    if downloaded_files:
        print("\n" + "="*70)
        print("Download complete! Next steps:")
        print("="*70)
        print("\nProcess datasets with build_chat_dataset.py:\n")

        for file_path, format_type in downloaded_files:
            dataset_name = Path(file_path).stem
            print(f"# Process {dataset_name}")
            print(f"python -m dataset.build_chat_dataset \\")
            print(f"  --input-file {file_path} \\")
            print(f"  --output-dir data/{dataset_name}-processed \\")
            print(f"  --format {format_type} \\")
            print(f"  --tokenizer gpt2 \\")
            print(f"  --max-length 2048 \\")
            print(f"  --train-split 0.95")
            print()

        print("\nThen train with:")
        print("python pretrain.py \\")
        print("  arch=trm \\")
        print(f"  data_paths=\"[data/{Path(downloaded_files[0][0]).stem}-processed]\" \\")
        print("  +run_name=my-chat-model \\")
        print("  ema=True")
        print()


if __name__ == "__main__":
    main()
