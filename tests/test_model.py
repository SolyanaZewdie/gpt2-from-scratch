"""Tests for full GPT-2 model architecture."""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gpt2.config import GPTConfig
from gpt2.model import GPT


def test_weight_tying():
    """Verify weight tying between token embedding and language model output projection."""
    config = GPTConfig(
        block_size=64,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_forward_pass_with_loss():
    """Verify forward pass computes loss and produces correct shapes."""
    config = GPTConfig(
        block_size=64,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)
    model.eval()

    batch_size = 2
    seq_len = 16
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_autoregressive_generate():
    """Verify model generation expands input sequence by max_new_tokens."""
    config = GPTConfig(
        block_size=64,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (1, 5))
    max_new_tokens = 10
    out = model.generate(idx, max_new_tokens=max_new_tokens)

    assert out.shape == (1, 5 + max_new_tokens)
