# GPT-2 From Scratch

A clean, modular, and educational PyTorch implementation of OpenAI's **GPT-2** (Generative Pre-trained Transformer 2) architecture built entirely from scratch.

This repository provides clear, self-contained implementations of every component of GPT-2—from multi-head causal self-attention and pre-LayerNorm residual blocks to autoregressive text generation, training loops, and pretrained Hugging Face weight loading.

---

## Architecture Overview

```
Input Tokens + Positional Encodings
             │
             ▼
     ┌────────────────┐
     │  LayerNorm 1   │
     └───────┬────────┘
             ▼
     ┌────────────────┐
     │  Causal Self-  │
     │   Attention    │ ───┐ (Residual Add)
     └───────┬────────┘    │
             ├─────────────┘
             ▼
     ┌────────────────┐
     │  LayerNorm 2   │
     └───────┬────────┘
             ▼
     ┌────────────────┐
     │  MLP / GeLU    │ ───┐ (Residual Add)
     └───────┬────────┘    │
             ├─────────────┘
             ▼
     [ Repeat x N Layers ]
             │
             ▼
     ┌────────────────┐
     │ Final LayerNorm│
     └───────┬────────┘
             ▼
     ┌────────────────┐
     │ Language Model │ (Weight-tied with Token Embeddings)
     │   Head (LM)    │
     └───────┬────────┘
             ▼
       Output Logits
```

### Supported GPT-2 Model Sizes

| Model Variant | Layers ($n_{layer}$) | Heads ($n_{head}$) | Embedding Dim ($n_{embd}$) | Context Window | Parameters |
|:--------------|:--------------------:|:------------------:|:--------------------------:|:--------------:|:----------:|
| **GPT-2 Small**  | 12                   | 12                 | 768                        | 1024           | 124M       |
| **GPT-2 Medium** | 24                   | 16                 | 1024                       | 1024           | 355M       |
| **GPT-2 Large**  | 36                   | 20                 | 1280                       | 1024           | 774M       |
| **GPT-2 XL**     | 48                   | 25                 | 1600                       | 1024           | 1.55B      |

---

## Repository Structure

```
gpt2-from-scratch/
├── configs/                     # YAML training and model configurations
│   ├── gpt2_124m.yaml           # Full GPT-2 (124M) configuration
│   ├── gpt2_medium_355m.yaml    # GPT-2 Medium (355M) configuration
│   └── train_shakespeare.yaml   # Fast training configuration for tiny Shakespeare
├── data/                        # Datasets and download scripts
│   ├── prepare_shakespeare.py   # Automated data downloader and train/val split
│   └── .gitkeep
├── checkpoints/                 # Saved model weights (.pt)
│   └── .gitkeep
├── notebooks/                   # Interactive Jupyter walkthroughs
│   └── gpt2_walkthrough.ipynb
├── scripts/                     # Executable CLI workflows
│   ├── train.py                 # Training script with mixed precision & cosine LR
│   └── generate.py              # Text generation CLI (supports checkpoints & pretrained)
├── src/
│   └── gpt2/                    # Core library package
│       ├── __init__.py          # Top-level exports
│       ├── attention.py         # Multi-Head Causal Self-Attention with PyTorch SDPA
│       ├── config.py            # GPTConfig dataclass and preset factory methods
│       ├── dataset.py           # TextDataset and high-throughput DataLoaderLite
│       ├── evaluate.py          # Validation loss and perplexity computation
│       ├── generate.py          # Autoregressive generation with temperature & top-k
│       ├── mlp.py               # Position-wise feed-forward network with GeLU
│       ├── model.py             # Full GPT model, weight tying & Hugging Face weight loader
│       ├── tokenizer.py         # Tiktoken GPT-2 BPE tokenizer wrapper
│       └── utils.py             # Device detection, checkpoints, cosine decay scheduler
├── tests/                       # Unit and integration test suite
│   ├── test_attention.py        # Validates causal masking and tensor shapes
│   ├── test_model.py            # Validates forward loss, weight tying & sampling
│   └── test_tokenizer.py        # Validates tokenization roundtrip
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Quickstart

### 1. Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/SolyanaZewdie/gpt2-from-scratch.git
cd gpt2-from-scratch

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
# Or install in editable mode:
pip install -e .
```

---

### 2. Prepare Data

Download and prepare the Tiny Shakespeare dataset:

```bash
python data/prepare_shakespeare.py
```

This will create `data/train.txt` and `data/val.txt`.

---

### 3. Train from Scratch

To launch training using the local Shakespeare configuration:

```bash
python scripts/train.py --config configs/train_shakespeare.yaml
```

Training features:
- **PyTorch 2.0 SDPA**: Uses FlashAttention kernels where available for accelerated training.
- **Cosine Decay with Warmup**: Dynamically updates learning rates according to standard transformer training schedules.
- **Gradient Accumulation**: Simulates larger effective batch sizes without running out of GPU VRAM.
- **Mixed Precision**: Automatic FP16 / BF16 mixed-precision training.

---

### 4. Text Generation

#### Generate from Official Pretrained Weights:
You can load official OpenAI GPT-2 weights directly from Hugging Face into our custom architecture:

```bash
python scripts/generate.py --pretrained gpt2 --prompt "Artificial intelligence will" --max_tokens 50
```

#### Generate from a Trained Checkpoint:
```bash
python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "ROMEO:" --temperature 0.8 --top_k 40
```

---

### 5. Running Tests

Run the test suite to verify causal masking, model shapes, weight tying, and tokenization:

```bash
pytest tests/ -v
```

---

## Architectural Highlights

- **Weight Tying**: In accordance with the GPT-2 paper, the token embedding matrix $W_{te}$ and the output language modeling head projection $W_{lm}$ share identical parameters:
  $$\text{weight}_{lm\_head} = \text{weight}_{wte}$$
- **Scaled Initialization**: Residual projections (`c_proj`) are initialized with a standard deviation scaled down by $\frac{0.02}{\sqrt{2 \cdot n_{layer}}}$ to maintain stable gradient dynamics across deep networks.
- **Pre-LayerNorm Residual Formulation**: Normalization is applied at the input of each sub-layer (prior to attention and MLP) rather than on the output.

---

## License

This project is open-source under the [MIT License](LICENSE).
