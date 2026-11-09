"""Download and prepare the Tiny Shakespeare dataset for GPT-2 training."""

import os
import requests

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download_shakespeare():
    """Download the Tiny Shakespeare dataset if it doesn't already exist."""
    input_file_path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(input_file_path):
        print(f"Downloading Tiny Shakespeare dataset from {DATA_URL}...")
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Saved dataset to {input_file_path}")
    else:
        print(f"Dataset already exists at {input_file_path}")

    # Read data
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()

    # Split into 90% train and 10% validation
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    train_path = os.path.join(DATA_DIR, "train.txt")
    val_path = os.path.join(DATA_DIR, "val.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_data)

    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_data)

    print(f"Train split saved to {train_path} ({len(train_data):,} characters)")
    print(f"Val split saved to {val_path} ({len(val_data):,} characters)")


if __name__ == "__main__":
    download_shakespeare()
