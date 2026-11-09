"""Inference CLI script for generating text with GPT-2."""

import argparse
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
from gpt2.config import GPTConfig
from gpt2.generate import generate_text
from gpt2.model import GPT
from gpt2.tokenizer import GPT2Tokenizer
from gpt2.utils import get_device, load_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a GPT-2 model")
    parser.add_argument(
        "--prompt",
        type=str,
        default="To be, or not to be, that is the question:",
        help="Input text prompt to condition text generation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="Load official OpenAI pretrained GPT-2 weights from Hugging Face",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=60,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Softmax temperature for sampling (lower = more deterministic)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Filter logits to only the top K highest probability tokens",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer()

    if args.pretrained is not None:
        print(f"Loading pretrained model: {args.pretrained}")
        model = GPT.from_pretrained(args.pretrained)
    elif args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model, _, _ = load_checkpoint(args.checkpoint, device)
    else:
        print("No pretrained weight or checkpoint specified; using randomly initialized GPT-2 Small.")
        config = GPTConfig.gpt2_124m()
        model = GPT(config)

    model.to(device)

    print(f"\nConditioning prompt:\n'{args.prompt}'\n")
    output = generate_text(
        model=model,
        prompt=args.prompt,
        tokenizer=tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print("--- Generated Output ---")
    print(output)
    print("------------------------")


if __name__ == "__main__":
    main()
