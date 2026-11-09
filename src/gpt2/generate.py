"""Text generation inference utilities."""

from typing import Optional
import torch

from gpt2.model import GPT
from gpt2.tokenizer import GPT2Tokenizer


def generate_text(
    model: GPT,
    prompt: str,
    tokenizer: Optional[GPT2Tokenizer] = None,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    device: Optional[torch.device] = None,
) -> str:
    """Generate completion for a text prompt using a GPT model.

    Args:
        model: GPT model
        prompt: Initial conditioning text string
        tokenizer: GPT2Tokenizer instance
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (1.0 = standard, <1.0 = more deterministic)
        top_k: Top-k filtering threshold
        device: Device to place tensors on

    Returns:
        Generated full text string (prompt + generated tokens)
    """
    model.eval()
    if tokenizer is None:
        tokenizer = GPT2Tokenizer()

    if device is None:
        device = next(model.parameters()).device

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eot_token]

    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        out_tokens = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    out_indices = out_tokens[0].tolist()
    return tokenizer.decode(out_indices)
