"""Tokenizer wrapper for GPT-2 using tiktoken."""

from typing import List, Union


class GPT2Tokenizer:
    """GPT-2 Tokenizer wrapper using OpenAI's tiktoken BPE tokenizer."""

    def __init__(self):
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding("gpt2")
            self._use_tiktoken = True
        except ImportError:
            self._enc = None
            self._use_tiktoken = False
            print(
                "Warning: tiktoken is not installed. Using rudimentary UTF-8 byte tokenizer. "
                "Install tiktoken (`pip install tiktoken`) for exact GPT-2 tokenization."
            )

    @property
    def eot_token(self) -> int:
        """The <|endoftext|> token ID in GPT-2 is 50256."""
        return 50256

    @property
    def vocab_size(self) -> int:
        """GPT-2 vocabulary size."""
        return 50257

    def encode(self, text: str, allowed_special: Union[str, set] = {"<|endoftext|>"}) -> List[int]:
        """Encode string text into a list of token integers.

        Args:
            text: Input string
            allowed_special: Special tokens allowed in encoding

        Returns:
            List of integer token IDs
        """
        if self._use_tiktoken:
            return self._enc.encode(text, allowed_special=allowed_special)
        else:
            # Fallback: simple byte encoding (values 0-255)
            return list(text.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        """Decode a list of token integers back into a string.

        Args:
            tokens: List of integer token IDs

        Returns:
            Decoded string
        """
        if self._use_tiktoken:
            return self._enc.decode(tokens)
        else:
            # Fallback byte decoding
            valid_bytes = [t for t in tokens if 0 <= t < 256]
            return bytes(valid_bytes).decode("utf-8", errors="replace")
