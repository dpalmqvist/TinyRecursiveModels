"""
Tokenizer utilities for chat model training

Provides tokenizer loading, caching, and special token management.
"""

from typing import Optional
import os
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer


# Cache for loaded tokenizers
_TOKENIZER_CACHE = {}


def load_tokenizer(
    tokenizer_path: str,
    cache: bool = True
) -> PreTrainedTokenizer:
    """
    Load a tokenizer from HuggingFace or local path.

    Args:
        tokenizer_path: Either a HuggingFace model name or path to saved tokenizer
        cache: Whether to cache the tokenizer in memory

    Returns:
        Loaded tokenizer
    """
    if cache and tokenizer_path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[tokenizer_path]

    # Check if it's a local path
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if cache:
        _TOKENIZER_CACHE[tokenizer_path] = tokenizer

    return tokenizer


def get_tokenizer_from_dataset(dataset_path: str) -> Optional[PreTrainedTokenizer]:
    """
    Load tokenizer that was saved with the dataset.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Loaded tokenizer or None if not found
    """
    tokenizer_path = os.path.join(dataset_path, "tokenizer")

    if os.path.exists(tokenizer_path):
        return load_tokenizer(tokenizer_path)

    return None


def get_vocab_size(tokenizer: PreTrainedTokenizer) -> int:
    """Get vocabulary size from tokenizer"""
    return len(tokenizer)


def clear_tokenizer_cache():
    """Clear the tokenizer cache"""
    global _TOKENIZER_CACHE
    _TOKENIZER_CACHE = {}
