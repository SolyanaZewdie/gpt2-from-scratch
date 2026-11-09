"""Evaluation utilities for calculating loss and perplexity."""

import math
import torch
from typing import Optional

from gpt2.dataset import DataLoaderLite


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataloader: DataLoaderLite,
    eval_steps: int = 20,
    device: Optional[torch.device] = None,
) -> float:
    """Estimate model loss over multiple batches from dataloader.

    Args:
        model: GPT model to evaluate
        dataloader: DataLoaderLite instance
        eval_steps: Number of batches to average over
        device: Device to place tensors on

    Returns:
        Average loss across evaluated batches
    """
    model.eval()
    losses = torch.zeros(eval_steps)
    for k in range(eval_steps):
        x, y = dataloader.next_batch()
        if device is not None:
            x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity given cross-entropy loss: PPL = exp(loss)."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")
