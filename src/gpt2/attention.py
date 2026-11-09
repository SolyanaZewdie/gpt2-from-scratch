"""Causal Self-Attention mechanism for GPT-2."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention.
    
    Computes query, key, and value vectors for all heads in a single matrix
    multiplication, performs scaled dot-product attention with an autoregressive
    lower-triangular mask, and projects the concatenated head outputs back
    to the embedding dimension.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        # Key, Query, Value projections combined into a single linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer: ensures tokens can only attend to previous tokens
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension (n_embd)

        # Calculate Query, Key, Value for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Fast causal self-attention using PyTorch 2.0 SDPA if available
        if hasattr(F, "scaled_dot_product_attention"):
            # is_causal=True automatically handles the lower triangular masking
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Fallback manual self-attention calculation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Re-assemble head outputs side by side: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection & residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y
