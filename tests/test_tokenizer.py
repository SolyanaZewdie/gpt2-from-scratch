"""Tests for tokenizer wrapper."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gpt2.tokenizer import GPT2Tokenizer


def test_tokenizer_roundtrip():
    """Verify text is encoded and decoded without data loss."""
    tokenizer = GPT2Tokenizer()
    text = "Hello world! Building GPT-2 from scratch in PyTorch."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_tokenizer_eot_token():
    """Verify special endoftext token is recognized."""
    tokenizer = GPT2Tokenizer()
    eot_id = tokenizer.eot_token
    assert eot_id == 50256
