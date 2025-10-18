"""
Generation/Inference Script for Chat Model

Provides utilities for text generation with the trained chat model.

Usage:
    python generate.py \
        --checkpoint checkpoints/my-chat-model/step_1000 \
        --prompt "Hello, how are you?" \
        --max-length 100 \
        --temperature 0.7
"""

from typing import List, Dict, Optional
import os
import json
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from utils.functions import load_model_class
from utils.tokenizer import get_tokenizer_from_dataset, load_tokenizer


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Special tokens from dataset builder
USER_START = "<|user|>"
USER_END = "<|enduser|>"
ASSISTANT_START = "<|assistant|>"
ASSISTANT_END = "<|endassistant|>"
SYSTEM_START = "<|system|>"
SYSTEM_END = "<|endsystem|>"
SEP_TOKEN = "<|sep|>"
EOS_TOKEN = "<|endoftext|>"


def load_model_and_config(checkpoint_path: str, device: str = "cuda"):
    """Load model and configuration from checkpoint"""
    # Load the model state dict
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Load config from checkpoint directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")

    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract architecture config
    arch_config = config['arch']

    # Create model
    model_cls = load_model_class(arch_config['name'])
    loss_head_cls = load_model_class(arch_config['loss']['name'])

    # Prepare model config
    # Note: These need to match training config
    model_cfg = {
        **{k: v for k, v in arch_config.items() if k not in ['name', 'loss']},
        'batch_size': 1,  # For inference
        'vocab_size': None,  # Will be set from tokenizer
        'seq_len': None,  # Will be set from tokenizer
        'num_puzzle_identifiers': 1,  # Not used for chat
        'causal': True
    }

    return state_dict, model_cfg, arch_config, config


def create_inference_model(state_dict, model_cfg, arch_config, tokenizer, device="cuda"):
    """Create and initialize model for inference"""
    # Update vocab size and seq_len from tokenizer
    model_cfg['vocab_size'] = len(tokenizer)
    model_cfg['seq_len'] = 2048  # Default, adjust as needed

    # Create model
    model_cls = load_model_class(arch_config['name'])
    loss_head_cls = load_model_class(arch_config['loss']['name'])

    # Extract loss head kwargs (exclude 'name' key)
    loss_head_kwargs = {k: v for k, v in arch_config['loss'].items() if k != 'name'}

    with torch.device(device):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_head_kwargs)

        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    return model


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda"
) -> str:
    """
    Generate text using the chat model.

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device to run on

    Returns:
        Generated text
    """
    # Format prompt for chat model
    formatted_prompt = f"{USER_START}{prompt}{USER_END}{SEP_TOKEN}{ASSISTANT_START}"

    # Tokenize
    input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], dtype=torch.int32, device=device)

    # Prepare batch
    batch = {
        "inputs": input_ids,
        "labels": torch.full_like(input_ids, -100),  # Not used for generation
        "puzzle_identifiers": torch.zeros((1,), dtype=torch.int32, device=device)
    }

    # Initialize carry
    carry = model.initial_carry(batch)

    # Generated token IDs
    generated_ids = input_ids[0].tolist()

    # Generation loop
    for _ in range(max_new_tokens):
        # Forward pass
        carry, outputs = model.model(carry=carry, batch=batch)

        # Get logits for the last token
        logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        # Check for end of generation
        if next_token_id == tokenizer.eos_token_id:
            break

        # Add to generated sequence
        generated_ids.append(next_token_id)

        # Update batch for next iteration
        new_token = torch.tensor([[next_token_id]], dtype=torch.int32, device=device)
        batch["inputs"] = torch.cat([batch["inputs"], new_token], dim=1)
        batch["labels"] = torch.cat([batch["labels"], torch.full_like(new_token, -100)], dim=1)

        # Check if we need to truncate (avoid exceeding max sequence length)
        if batch["inputs"].shape[1] > model.model.config.seq_len:
            # Truncate from the left (keep most recent context)
            batch["inputs"] = batch["inputs"][:, -model.model.config.seq_len:]
            batch["labels"] = batch["labels"][:, -model.model.config.seq_len:]

    # Decode
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Extract assistant response
    if ASSISTANT_START in generated_text:
        response = generated_text.split(ASSISTANT_START)[-1]
        if ASSISTANT_END in response:
            response = response.split(ASSISTANT_END)[0]
        return response.strip()

    return generated_text


def interactive_chat(model, tokenizer, device="cuda"):
    """Run an interactive chat session"""
    print("=" * 50)
    print("TinyRecursiveModels Chat")
    print("=" * 50)
    print("Type 'quit' or 'exit' to end the session\n")

    conversation_history = []

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add to history
        conversation_history.append(("user", user_input))

        # Build prompt from history
        prompt_parts = []
        for role, content in conversation_history[-5:]:  # Keep last 5 turns
            if role == "user":
                prompt_parts.append(f"{USER_START}{content}{USER_END}")
            elif role == "assistant":
                prompt_parts.append(f"{ASSISTANT_START}{content}{ASSISTANT_END}")

        full_prompt = SEP_TOKEN.join(prompt_parts) + f"{SEP_TOKEN}{ASSISTANT_START}"

        # Generate response
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=user_input,
            max_new_tokens=200,
            temperature=0.7,
            device=device
        )

        print(f"Assistant: {response}\n")

        # Add to history
        conversation_history.append(("assistant", response))


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained chat model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer (will try to auto-detect)")
    parser.add_argument("--prompt", type=str, help="Input prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu, auto-detect if not specified)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat")

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        args.device = get_device()
        print(f"Auto-detected device: {args.device}")

    # Load tokenizer
    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer)
    else:
        # Try to auto-detect from checkpoint directory
        checkpoint_dir = os.path.dirname(args.checkpoint)
        # Go up to find data directory
        config_path = os.path.join(checkpoint_dir, "all_config.yaml")
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            data_path = config.get('data_paths', [None])[0]
            if data_path:
                tokenizer = get_tokenizer_from_dataset(data_path)
                if tokenizer is None:
                    raise ValueError(f"Could not load tokenizer from {data_path}")
            else:
                raise ValueError("Could not auto-detect tokenizer. Please specify --tokenizer")
        else:
            raise ValueError("Could not find config. Please specify --tokenizer")

    print(f"Loaded tokenizer with vocab size: {len(tokenizer)}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    state_dict, model_cfg, arch_config, config = load_model_and_config(args.checkpoint, args.device)
    model = create_inference_model(state_dict, model_cfg, arch_config, tokenizer, args.device)
    print("Model loaded successfully")

    # Generate or run interactive
    if args.interactive:
        interactive_chat(model, tokenizer, args.device)
    elif args.prompt:
        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}")
    else:
        print("Please specify either --prompt for single generation or --interactive for chat")


if __name__ == "__main__":
    main()
