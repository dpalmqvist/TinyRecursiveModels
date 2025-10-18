"""
Chat Dataset Loader for TinyRecursiveModels

Loads conversational datasets prepared by build_chat_dataset.py
Compatible with the existing training pipeline but optimized for chat data.
"""

import os
import json
from typing import List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata
from pydantic import BaseModel


class ChatDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead
    rank: int
    num_replicas: int


class ChatDataset(IterableDataset):
    """
    Dataset for chat/conversational data.

    Note: This reuses the "puzzle" terminology from the original codebase:
    - "puzzle" = conversation
    - "puzzle_identifier" = conversation_id
    - "group" = batch of conversations
    """

    def __init__(self, config: ChatDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Load metadata from all dataset paths
        self.metadata = self._merge_metadata()

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, \
            f"Global batch size {self.config.global_batch_size} must be divisible by world size {self.config.num_replicas}"

        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path: str) -> PuzzleDatasetMetadata:
        """Load metadata from a single dataset path"""
        metadata_path = os.path.join(dataset_path, self.split, "dataset.json")
        with open(metadata_path, "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _merge_metadata(self) -> PuzzleDatasetMetadata:
        """Merge metadata from multiple dataset paths"""
        merged_metadata = None

        for dataset_path in self.config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)

            if merged_metadata is None:
                merged_metadata = current_metadata
            else:
                # Verify compatibility
                assert merged_metadata.seq_len == current_metadata.seq_len
                assert merged_metadata.vocab_size == current_metadata.vocab_size
                assert merged_metadata.pad_id == current_metadata.pad_id
                assert merged_metadata.ignore_label_id == current_metadata.ignore_label_id

                # Accumulate counts
                merged_metadata.num_puzzle_identifiers += current_metadata.num_puzzle_identifiers
                merged_metadata.total_groups += current_metadata.total_groups
                merged_metadata.total_puzzles += current_metadata.total_puzzles

        return merged_metadata

    def _lazy_load_dataset(self):
        """Lazy load dataset arrays when first accessed"""
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_identifiers": None,  # Keep in memory
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data for each set
        self._data = {}
        for set_name in self.metadata.sets:
            for i, dataset_path in enumerate(self.config.dataset_paths):
                # Use unique names for multiple datasets
                set_key = f"{set_name}{i}" if i > 0 else set_name

                self._data[set_key] = {
                    field_name: np.load(
                        os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"),
                        mmap_mode=mmap_mode
                    )
                    for field_name, mmap_mode in field_mmap_modes.items()
                }

    def _collate_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert batch to tensors and handle padding"""
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad batch if needed
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {
                k: np.pad(
                    v,
                    ((0, pad_size),) + ((0, 0),) * (v.ndim - 1),
                    constant_values=pad_values[k]
                )
                for k, v in batch.items()
            }

        # Convert to tensors
        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self):
        """Iterate over test set (sequential, no shuffling)"""
        for set_name, dataset in self._data.items():
            total_examples = len(dataset["inputs"])

            start_index = 0
            while start_index < total_examples:
                # Compute batch indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)

                # Get local shard for this rank
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)

                # Extract batch
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start:local_end],
                    "labels": dataset["labels"][local_start:local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][local_start:local_end]
                })

                yield set_name, batch, end_index - start_index

                start_index += self.config.global_batch_size

    def _iter_train(self):
        """Iterate over training set (shuffled, with epochs)"""
        for set_name, dataset in self._data.items():
            self._iters += 1

            # Create shuffled indices for this epoch
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))
            num_examples = len(dataset["inputs"])

            # Repeat for epochs_per_iter
            all_indices = []
            for _ in range(self.config.epochs_per_iter):
                all_indices.append(rng.permutation(num_examples))
            all_indices = np.concatenate(all_indices)

            # Batch the data
            start_idx = 0
            while start_idx < len(all_indices):
                # Get batch indices
                end_idx = min(start_idx + self.config.global_batch_size, len(all_indices))
                batch_indices = all_indices[start_idx:end_idx]

                # Drop last incomplete batch
                if len(batch_indices) < self.config.global_batch_size:
                    break

                # Get local shard for this rank
                local_start = self.config.rank * self.local_batch_size
                local_end = (self.config.rank + 1) * self.local_batch_size
                local_indices = batch_indices[local_start:local_end]

                # Extract batch
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_indices],
                    "labels": dataset["labels"][local_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][local_indices]
                })

                yield set_name, batch, self.config.global_batch_size

                start_idx = end_idx

    def __iter__(self):
        """Main iterator"""
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, \
            "Multi-worker data loading not currently supported"

        self._lazy_load_dataset()

        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
