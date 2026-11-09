"""Utility functions for training, checkpointing, and hardware setup."""

import math
import os
import random
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import yaml

from gpt2.config import GPTConfig


def get_device() -> torch.device:
    """Detect available accelerator (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Calculate learning rate with linear warmup and cosine decay.

    1) Linear warmup for `warmup_steps`
    2) Cosine decay from `max_lr` down to `min_lr` until `max_steps`
    3) If step > max_steps, returns `min_lr`
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: GPTConfig,
    step: int,
    val_loss: float,
    save_path: str,
) -> None:
    """Save model checkpoint and training state."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        "step": step,
        "val_loss": val_loss,
        "config": config.__dict__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }
    torch.save(state, save_path)
    print(f"Saved checkpoint to {save_path} (step {step}, val_loss {val_loss:.4f})")


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, GPTConfig, Dict[str, Any]]:
    """Load model and state from checkpoint."""
    from gpt2.model import GPT

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint from {checkpoint_path} (step {checkpoint.get('step')})")
    return model, config, checkpoint


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
