"""GPT-2 Model Architecture and Pretrained Weight Loading."""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2.config import GPTConfig
from gpt2.attention import CausalSelfAttention
from gpt2.mlp import MLP


class Block(nn.Module):
    """Transformer Decoder Block with pre-LayerNorm architecture."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm residual stream
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Full GPT-2 Language Model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share parameters between token embeddings and output projection
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all model weights
        self.apply(self._init_weights)

        # Apply special scaled initialization to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with normal distribution N(0, 0.02)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, subtract position embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for training and inference.

        Args:
            idx: Tensor of token indices, shape (batch_size, seq_len)
            targets: Optional ground-truth targets, shape (batch_size, seq_len)

        Returns:
            Tuple of (logits, loss). If targets is None, loss is None.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        )

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward through embeddings
        tok_emb = self.transformer.wte(idx)  # token embeddings (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through all transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If targets are provided, compute cross-entropy loss over entire sequence
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # For inference optimization, compute logits only for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """Configure AdamW optimizer with weight decay applied only to 2D weight tensors."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Any parameter that is 2D will be decayed; 1D biases and layernorms will not
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW if CUDA is available and supported
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generate new tokens from conditioning tokens.

        Args:
            idx: Conditioning tokens, shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Softmax temperature (higher = more random)
            top_k: If set, filter logits to only top k options

        Returns:
            Tensor of generated token indices, shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop current context if it exceeds block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Forward model to get logits
            logits, _ = self(idx_cond)

            # Pluck logits at final step and scale by temperature
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Optionally crop probabilities to top-k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2") -> "GPT":
        """Load pretrained OpenAI GPT-2 weights from Hugging Face.

        Supported types: 'gpt2' (124M), 'gpt2-medium' (355M),
                         'gpt2-large' (774M), 'gpt2-xl' (1558M).
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}, (
            f"Unrecognized model_type: {model_type}"
        )
        try:
            from transformers import GPT2LMHeadModel
        except ImportError as e:
            raise ImportError(
                "Loading pretrained GPT-2 requires transformers package. "
                "Install it with `pip install transformers`."
            ) from e

        print(f"Loading weights from pretrained Hugging Face GPT-2: {model_type}...")

        # Configuration dictionary mapping model names to dimensions
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True

        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        # Load HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Filter out attention bias buffer if present
        sd_keys_hf = [
            k
            for k in sd_hf.keys()
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]

        # In HuggingFace, OpenAI Conv1D weights are transposed relative to standard nn.Linear
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for transposed Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        print(f"Successfully loaded {model_type} ({model.get_num_params():,} parameters)")
        return model
