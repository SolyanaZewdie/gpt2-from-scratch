"""Dataset and DataLoader implementations for GPT-2 training."""

import os
from typing import Tuple, Union, Optional
import torch
from torch.utils.data import Dataset, DataLoader

from gpt2.tokenizer import GPT2Tokenizer


class TextDataset(Dataset):
    """PyTorch Dataset for causal language modeling from a text file or string."""

    def __init__(
        self,
        data_or_path: str,
        block_size: int = 1024,
        tokenizer: Optional[GPT2Tokenizer] = None,
    ):
        """Initialize dataset.

        Args:
            data_or_path: File path containing raw text, or a raw text string.
            block_size: Context sequence length.
            tokenizer: GPT2Tokenizer instance. If None, instantiates a new one.
        """
        self.block_size = block_size
        self.tokenizer = tokenizer or GPT2Tokenizer()

        if os.path.isfile(data_or_path):
            with open(data_or_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = data_or_path

        tokens = self.tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        if len(self.tokens) <= self.block_size:
            raise ValueError(
                f"Dataset size ({len(self.tokens)} tokens) must be greater than block_size ({self.block_size})."
            )

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Chunk of block_size + 1 tokens
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class DataLoaderLite:
    """Lightweight, zero-overhead sequential batch loader for language modeling.

    Reads a text file or pre-tokenized tensor and iterates continuously through
    batches of size (batch_size, block_size) with targets shifted by 1.
    """

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        data_path: Optional[str] = None,
        text: Optional[str] = None,
        device: str = "cpu",
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.current_position = 0

        tokenizer = GPT2Tokenizer()
        if data_path is not None and os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        elif text is not None:
            raw_text = text
        else:
            raise ValueError("Either data_path or text must be provided.")

        tokens = tokenizer.encode(raw_text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens):,} tokens into DataLoaderLite.")
        assert len(self.tokens) > batch_size * block_size + 1, (
            "Not enough tokens for at least one batch."
        )

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch next batch of (x, y) tensors."""
        B, T = self.batch_size, self.block_size
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets

        # Advance position in the tensor
        self.current_position += B * T

        # Wrap around if next batch will exceed dataset length
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        if self.device != "cpu":
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

        return x, y

    def reset(self) -> None:
        """Reset stream position."""
        self.current_position = 0
