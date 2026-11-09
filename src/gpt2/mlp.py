"""Multi-Layer Perceptron (Feed-Forward Network) for GPT-2."""

import torch
import torch.nn as nn

from gpt2.config import GPTConfig


class MLP(nn.Module):
    """Position-wise Feed-Forward Network.

    Consists of two linear transformations with a GeLU activation in between:
        FFN(x) = GELU(x * W1 + b1) * W2 + b2
    where the hidden layer dimension is 4 * n_embd.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Expansion layer: n_embd -> 4 * n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # OpenAI GPT-2 uses the tanh approximation for GELU
        self.gelu = nn.GELU(approximate="tanh")
        # Projection back: 4 * n_embd -> n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
