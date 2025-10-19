#!/usr/bin/env python3
"""
Quick test script to verify model configuration doesn't produce NaN outputs.
Usage: python test_config.py --config-name cfg_chat_fast_fp32
"""
import torch
import hydra
from omegaconf import DictConfig
from pretrain import PretrainConfig, create_model, create_dataloader


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@hydra.main(config_path="config", config_name="cfg_chat_fast_fp32", version_base=None)
def test_model(hydra_config: DictConfig):
    """Test that model can be created and produces valid (non-NaN) outputs"""

    device = get_device()
    print(f"Using device: {device}")

    # Load config
    config = PretrainConfig(**hydra_config)
    print(f"\nConfig loaded:")
    print(f"  Architecture: {config.arch.name}")
    print(f"  Hidden size: {config.arch.__pydantic_extra__.get('hidden_size', 'default')}")
    print(f"  Forward dtype: {config.arch.__pydantic_extra__.get('forward_dtype', 'default')}")

    # Create dataloader
    print("\nLoading dataset...")
    train_loader, train_metadata = create_dataloader(
        config, "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1
    )

    print(f"  Vocab size: {train_metadata.vocab_size}")
    print(f"  Seq len: {train_metadata.seq_len}")

    # Create model
    print("\nCreating model...")
    model, optimizers, optimizer_lrs = create_model(
        config, train_metadata,
        rank=0, world_size=1, device=device
    )

    model.eval()

    # Get one batch
    print("\nTesting forward pass...")
    for set_name, batch, global_batch_size in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.inference_mode():
            carry = model.initial_carry(batch)
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, batch=batch, return_keys=[]
            )

        print(f"  Loss: {loss.item():.4f}")
        print(f"  Loss is NaN: {torch.isnan(loss).item()}")
        print(f"  Loss is Inf: {torch.isinf(loss).item()}")

        if "logits" in preds:
            logits = preds["logits"]
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits min: {logits.min().item():.4f}")
            print(f"  Logits max: {logits.max().item():.4f}")
            print(f"  Logits mean: {logits.mean().item():.4f}")
            print(f"  Logits has NaN: {torch.isnan(logits).any().item()}")
            print(f"  Logits has Inf: {torch.isinf(logits).any().item()}")

        # Test one more step to ensure it's stable
        print("\nTesting second forward pass...")
        with torch.inference_mode():
            carry, loss2, metrics2, preds2, all_finish2 = model(
                carry=carry, batch=batch, return_keys=[]
            )

        print(f"  Loss: {loss2.item():.4f}")
        print(f"  Loss is NaN: {torch.isnan(loss2).item()}")

        if torch.isnan(loss).item() or torch.isnan(loss2).item():
            print("\n❌ FAILED: Model produces NaN outputs!")
            return False
        else:
            print("\n✅ SUCCESS: Model produces valid outputs (no NaN)!")
            print("You can now train with: python pretrain.py --config-name cfg_chat_fast_fp32")
            return True

        break  # Only test one batch

    return False


if __name__ == "__main__":
    test_model()
