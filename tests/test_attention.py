"""Tests for Causal Self-Attention mechanism."""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gpt2.config import GPTConfig
from gpt2.attention import CausalSelfAttention


def test_causal_self_attention_shape():
    """Verify output tensor preserves input dimensions (B, T, C)."""
    config = GPTConfig(block_size=128, n_embd=64, n_head=4, dropout=0.0)
    attn = CausalSelfAttention(config)
    attn.eval()

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.n_embd)
    out = attn(x)

    assert out.shape == (batch_size, seq_len, config.n_embd)


def test_causal_masking_property():
    """Verify that earlier token representations are invariant to changes in later tokens."""
    config = GPTConfig(block_size=64, n_embd=32, n_head=2, dropout=0.0)
    attn = CausalSelfAttention(config)
    attn.eval()

    seq_len = 8
    x1 = torch.randn(1, seq_len, config.n_embd)
    x2 = x1.clone()
    # Modify future tokens (positions 4 to 7)
    x2[:, 4:, :] = torch.randn(1, 4, config.n_embd)

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Positions 0 through 3 should be identical within numerical precision
    torch.testing.assert_close(out1[:, :4, :], out2[:, :4, :], atol=1e-5, rtol=1e-5)
