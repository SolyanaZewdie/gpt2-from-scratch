"""Configuration classes for GPT-2 models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """Hyperparameters for GPT-2 architecture."""

    block_size: int = 1024       # Maximum sequence length (context window)
    vocab_size: int = 50257      # Size of vocabulary (50,000 BPE merges + 256 byte tokens + 1 <|endoftext|>)
    n_layer: int = 12            # Number of transformer blocks
    n_head: int = 12             # Number of attention heads
    n_embd: int = 768            # Embedding dimension
    dropout: float = 0.0         # Dropout probability
    bias: bool = True            # Use bias in Linears and LayerNorms, like OpenAI GPT-2

    @classmethod
    def gpt2_124m(cls, **kwargs) -> "GPTConfig":
        """GPT-2 Small (124M parameters)."""
        return cls(n_layer=12, n_head=12, n_embd=768, **kwargs)

    @classmethod
    def gpt2_medium_355m(cls, **kwargs) -> "GPTConfig":
        """GPT-2 Medium (355M parameters)."""
        return cls(n_layer=24, n_head=16, n_embd=1024, **kwargs)

    @classmethod
    def gpt2_large_774m(cls, **kwargs) -> "GPTConfig":
        """GPT-2 Large (774M parameters)."""
        return cls(n_layer=36, n_head=20, n_embd=1280, **kwargs)

    @classmethod
    def gpt2_xl_1558m(cls, **kwargs) -> "GPTConfig":
        """GPT-2 XL (1558M parameters)."""
        return cls(n_layer=48, n_head=25, n_embd=1600, **kwargs)

    @classmethod
    def shakespeare_tiny(cls, **kwargs) -> "GPTConfig":
        """Lightweight configuration suitable for fast local training on Shakespeare."""
        return cls(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2, **kwargs)
