"""
Chat Dataset Builder for TinyRecursiveModels

Converts conversational datasets (ShareGPT, Alpaca, OpenAssistant formats)
into the format expected by the training pipeline.

Usage:
    python -m dataset.build_chat_dataset \
        --input-file data/chat_data.json \
        --output-dir data/chat-dataset \
        --format sharegpt \
        --tokenizer gpt2 \
        --max-length 2048 \
        --train-split 0.95
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
import json
import numpy as np
from pathlib import Path

from argdantic import ArgParser
from pydantic import BaseModel
from transformers import AutoTokenizer

from dataset.common import PuzzleDatasetMetadata


cli = ArgParser()


class ChatDataProcessConfig(BaseModel):
    input_file: str
    output_dir: str
    format: str = "sharegpt"  # sharegpt, alpaca, openassistant
    tokenizer: str = "gpt2"  # HuggingFace tokenizer name
    max_length: int = 2048
    train_split: float = 0.95
    seed: int = 42
    subsample_size: Optional[int] = None  # For testing with smaller datasets


# Special tokens
PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|endoftext|>"
BOS_TOKEN = "<|startoftext|>"
SEP_TOKEN = "<|sep|>"  # Separator between user/assistant turns

# Chat format markers
USER_START = "<|user|>"
USER_END = "<|enduser|>"
ASSISTANT_START = "<|assistant|>"
ASSISTANT_END = "<|endassistant|>"
SYSTEM_START = "<|system|>"
SYSTEM_END = "<|endsystem|>"


@dataclass
class ChatExample:
    """Single conversation example"""
    conversation_id: str
    turns: List[Tuple[str, str]]  # List of (role, content) tuples


def load_sharegpt_format(input_file: str) -> List[ChatExample]:
    """Load ShareGPT format conversations"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for idx, item in enumerate(data):
        conv_id = item.get('id', f'conv_{idx}')
        conversations = item.get('conversations', [])

        turns = []
        for msg in conversations:
            role = msg.get('from', 'unknown')
            content = msg.get('value', '')

            # Map ShareGPT roles to our format
            if role in ['human', 'user']:
                role = 'user'
            elif role in ['gpt', 'assistant']:
                role = 'assistant'
            elif role == 'system':
                role = 'system'

            turns.append((role, content))

        if turns:
            examples.append(ChatExample(conv_id, turns))

    return examples


def load_alpaca_format(input_file: str) -> List[ChatExample]:
    """Load Alpaca format (instruction, input, output)"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for idx, item in enumerate(data):
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')

        # Combine instruction and input
        user_message = instruction
        if input_text:
            user_message += f"\n\n{input_text}"

        turns = [
            ('user', user_message),
            ('assistant', output)
        ]

        examples.append(ChatExample(f'alpaca_{idx}', turns))

    return examples


def load_openassistant_format(input_file: str) -> List[ChatExample]:
    """Load OpenAssistant format"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for idx, item in enumerate(data):
        messages = item.get('messages', [])

        turns = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            turns.append((role, content))

        if turns:
            examples.append(ChatExample(f'oa_{idx}', turns))

    return examples


def format_conversation_for_training(
    example: ChatExample,
    tokenizer,
    max_length: int,
    include_system: bool = False
) -> Tuple[List[int], List[int]]:
    """
    Format conversation into input/label token sequences for causal LM training.

    Returns:
        inputs: Token IDs for input sequence
        labels: Token IDs shifted by 1 for next-token prediction
    """

    # Build the formatted conversation
    conversation_parts = []

    for role, content in example.turns:
        if role == 'system' and include_system:
            conversation_parts.append(f"{SYSTEM_START}{content}{SYSTEM_END}")
        elif role == 'user':
            conversation_parts.append(f"{USER_START}{content}{USER_END}")
        elif role == 'assistant':
            conversation_parts.append(f"{ASSISTANT_START}{content}{ASSISTANT_END}")

    # Join with separator
    full_text = SEP_TOKEN.join(conversation_parts) + EOS_TOKEN

    # Tokenize
    tokens = tokenizer.encode(full_text, add_special_tokens=False)

    # Truncate if too long (reserve 1 token for shifting)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    # Create inputs and labels for next-token prediction
    # For causal LM: model sees tokens[:-1] and predicts tokens[1:]
    # We pad to maintain max_length for both inputs and labels
    if len(tokens) > 0:
        inputs = tokens[:-1] + [tokenizer.pad_token_id]
        labels = tokens[1:] + [-100]  # -100 will be ignored in loss calculation
    else:
        # Handle empty sequence edge case
        inputs = [tokenizer.pad_token_id]
        labels = [-100]

    return inputs, labels


def prepare_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """Load and prepare tokenizer with special tokens"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Add special tokens
    special_tokens = {
        'pad_token': PAD_TOKEN,
        'eos_token': EOS_TOKEN,
        'bos_token': BOS_TOKEN,
        'additional_special_tokens': [
            SEP_TOKEN,
            USER_START, USER_END,
            ASSISTANT_START, ASSISTANT_END,
            SYSTEM_START, SYSTEM_END
        ]
    }

    tokenizer.add_special_tokens(special_tokens)

    return tokenizer


def convert_dataset(config: ChatDataProcessConfig):
    """Main conversion function"""
    np.random.seed(config.seed)

    print(f"Loading data from {config.input_file} (format: {config.format})")

    # Load conversations based on format
    if config.format == "sharegpt":
        examples = load_sharegpt_format(config.input_file)
    elif config.format == "alpaca":
        examples = load_alpaca_format(config.input_file)
    elif config.format == "openassistant":
        examples = load_openassistant_format(config.input_file)
    else:
        raise ValueError(f"Unknown format: {config.format}")

    print(f"Loaded {len(examples)} conversations")

    # Subsample if requested
    if config.subsample_size and config.subsample_size < len(examples):
        np.random.shuffle(examples)
        examples = examples[:config.subsample_size]
        print(f"Subsampled to {len(examples)} conversations")

    # Prepare tokenizer
    print(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = prepare_tokenizer(config.tokenizer)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # Split into train/test
    split_idx = int(len(examples) * config.train_split)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]

    print(f"Train: {len(train_examples)}, Test: {len(test_examples)}")

    # Process each split
    for split_name, split_examples in [("train", train_examples), ("test", test_examples)]:
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        # Convert conversations to token sequences
        inputs_list = []
        labels_list = []
        conversation_identifiers = []
        conversation_indices = [0]
        group_indices = [0]

        conversation_id = 0
        total_tokens = 0

        for example in split_examples:
            inputs, labels = format_conversation_for_training(
                example, tokenizer, config.max_length
            )

            # Pad to max_length
            pad_length = config.max_length - len(inputs)
            if pad_length > 0:
                inputs = inputs + [tokenizer.pad_token_id] * pad_length
                labels = labels + [-100] * pad_length  # -100 is ignored in loss

            inputs_list.append(inputs)
            labels_list.append(labels)
            conversation_identifiers.append(conversation_id)

            conversation_id += 1
            total_tokens += len([t for t in labels if t != -100])

            # For chat, each conversation is its own "puzzle" and "group"
            conversation_indices.append(conversation_id)
            group_indices.append(conversation_id)

        # Convert to numpy arrays
        inputs_array = np.array(inputs_list, dtype=np.int32)
        labels_array = np.array(labels_list, dtype=np.int32)
        conversation_identifiers = np.array(conversation_identifiers, dtype=np.int32)
        conversation_indices = np.array(conversation_indices, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int32)

        # Save arrays
        subset_name = "all"
        np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__inputs.npy"), inputs_array)
        np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__labels.npy"), labels_array)
        np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__puzzle_identifiers.npy"), conversation_identifiers)
        np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__puzzle_indices.npy"), conversation_indices)
        np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__group_indices.npy"), group_indices)

        # Create metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=config.max_length,
            vocab_size=vocab_size,
            pad_id=tokenizer.pad_token_id,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=len(split_examples) + 1,  # +1 for blank
            total_groups=len(split_examples),
            mean_puzzle_examples=1.0,  # Each conversation is one example
            total_puzzles=len(split_examples),
            sets=[subset_name]
        )

        # Save metadata
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

        print(f"{split_name}: {len(split_examples)} conversations, {total_tokens:,} tokens")

    # Save tokenizer for later use
    tokenizer_path = os.path.join(config.output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    # Save config
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    print(f"Dataset saved to {config.output_dir}")


@cli.command(singleton=True)
def main(config: ChatDataProcessConfig):
    """Build chat dataset from conversational data"""
    convert_dataset(config)


if __name__ == "__main__":
    cli()
