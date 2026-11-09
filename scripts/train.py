"""Main training script for GPT-2 from scratch."""

import argparse
import os
import sys
import time

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
from gpt2.config import GPTConfig
from gpt2.dataset import DataLoaderLite
from gpt2.evaluate import estimate_loss, calculate_perplexity
from gpt2.generate import generate_text
from gpt2.model import GPT
from gpt2.tokenizer import GPT2Tokenizer
from gpt2.utils import (
    get_device,
    get_lr,
    load_yaml_config,
    save_checkpoint,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 from scratch")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_shakespeare.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/train.txt",
        help="Path to training text file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/val.txt",
        help="Path to validation text file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Hardware device to use (cuda/mps/cpu)",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # Load configuration
    cfg = load_yaml_config(args.config)
    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("training", {})

    model_config = GPTConfig(**m_cfg)
    model = GPT(model_config).to(device)

    print(f"Model parameters: {model.get_num_params():,}")

    # Training hyperparameters
    batch_size = t_cfg.get("batch_size", 8)
    grad_accum_steps = t_cfg.get("gradient_accumulation_steps", 4)
    max_steps = t_cfg.get("max_steps", 2000)
    warmup_steps = t_cfg.get("warmup_steps", 200)
    max_lr = float(t_cfg.get("learning_rate", 6e-4))
    min_lr = float(t_cfg.get("min_lr", 6e-5))
    weight_decay = float(t_cfg.get("weight_decay", 0.1))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    eval_interval = t_cfg.get("eval_interval", 200)
    eval_steps = t_cfg.get("eval_steps", 20)
    save_interval = t_cfg.get("save_interval", 500)

    # Prepare DataLoaders
    if not os.path.exists(args.train_data):
        print(f"Training data not found at {args.train_data}. Running data/prepare_shakespeare.py...")
        from data.prepare_shakespeare import download_shakespeare
        download_shakespeare()

    train_loader = DataLoaderLite(
        batch_size=batch_size,
        block_size=model_config.block_size,
        data_path=args.train_data,
        device=str(device),
    )
    val_loader = DataLoaderLite(
        batch_size=batch_size,
        block_size=model_config.block_size,
        data_path=args.val_data if os.path.exists(args.val_data) else args.train_data,
        device=str(device),
    )

    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=max_lr,
        betas=(t_cfg.get("beta1", 0.9), t_cfg.get("beta2", 0.95)),
        device_type=device.type,
    )

    tokenizer = GPT2Tokenizer()
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")

    use_autocast = device.type == "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"\n--- Starting Training ({max_steps} steps) ---")
    model.train()
    start_time = time.time()

    for step in range(max_steps):
        t0 = time.time()

        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate periodically
        if step % eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, eval_steps, device)
            val_ppl = calculate_perplexity(val_loss)
            print(f"\n[Step {step:05d} / {max_steps}] Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.out_dir, "best_model.pt")
                save_checkpoint(model, optimizer, model_config, step, val_loss, best_path)

            # Sample text to inspect generation quality
            sample = generate_text(model, "The kingdom of", tokenizer, max_new_tokens=25, device=device)
            print(f"Sample generation:\n>>> {sample.strip()}\n")

        # Forward & backward pass with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()

            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits, loss = model(x, y)
                    loss = loss / grad_accum_steps
            else:
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps

            accum_loss += loss.item()
            loss.backward()

        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        dt = time.time() - t0
        tokens_per_sec = (batch_size * model_config.block_size * grad_accum_steps) / dt

        if step % 10 == 0:
            print(
                f"step {step:05d} | loss: {accum_loss:.4f} | lr: {lr:.2e} | "
                f"grad_norm: {norm:.2f} | dt: {dt*1000:.1f}ms | tok/sec: {tokens_per_sec:.0f}"
            )

        # Periodic checkpoint save
        if step > 0 and step % save_interval == 0:
            save_path = os.path.join(args.out_dir, f"model_step_{step}.pt")
            save_checkpoint(model, optimizer, model_config, step, accum_loss, save_path)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    final_path = os.path.join(args.out_dir, "final_model.pt")
    save_checkpoint(model, optimizer, model_config, max_steps, accum_loss, final_path)


if __name__ == "__main__":
    main()
