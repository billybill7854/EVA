"""BabyBrain — a randomly-initialised, growable transformer.

The Ron Protocol forbids pretrained weights: BabyBrain *always* starts
from ``torch.nn.init``-style random initialisation. Growth is allowed —
layers and width can be added during a lifetime — but never loaded from
a pretrained checkpoint of another model.

Public surface (referenced throughout the repo):

* ``BabyBrain(vocab_size, d_model, n_layers, n_heads, dtype_str, device=None)``
* ``forward(input_ids) -> (logits, hidden)``
* ``predict_next(input_ids) -> probs`` (shape ``[batch, vocab]``)
* ``get_hidden_state() -> Tensor`` (last forward's hidden state)
* ``get_parameter_snapshot(sample_ratio=1.0) -> dict[name, {mean, std}]``
* ``parameter_count`` property
* ``architecture`` property (``"transformer"`` or ``"mamba"``)
* ``device`` property
* ``grow_width(new_d_model)`` / ``grow_depth(extra_layers)`` — widen or
  deepen the network while preserving previous behaviour (identity init
  for fresh weights).
* ``estimate_memory_bytes()`` — used by the RAM-budget guard.
* ``detect_device()`` module-level helper.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


DType = Union[torch.dtype, str]


def _resolve_dtype(dtype: DType) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype, torch.float32)


def detect_device(preference: str = "auto") -> torch.device:
    """Resolve a :class:`torch.device` honouring user preference.

    Order: ``cuda`` > ``mps`` > ``cpu``. Float16 only materialises well on
    CUDA; callers (e.g. training loop) may downgrade dtype when on CPU/MPS.
    """

    pref = (preference or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if pref == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


class _MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Causal scaled dot-product (PyTorch SDPA handles the mask).
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(b, t, self.d_model)
        return self.out(attn)


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class BabyBrain(nn.Module):
    """Randomly-initialised causal transformer with growth primitives."""

    architecture: str = "transformer"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dtype_str: str = "float32",
        device: Optional[Union[str, torch.device]] = None,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._d_model = d_model
        self._n_heads = n_heads
        self._dtype = _resolve_dtype(dtype_str)
        self._max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._last_hidden: Optional[torch.Tensor] = None

        # Random init as per Ron Protocol.
        self.apply(self._random_init)
        resolved_device = (
            torch.device(device) if device is not None else detect_device("auto")
        )
        # float16 on CPU is very slow and numerically fragile — keep fp32 off-GPU.
        use_dtype = self._dtype
        if resolved_device.type in ("cpu", "mps") and use_dtype == torch.float16:
            use_dtype = torch.float32
        self.to(resolved_device, dtype=use_dtype)
        self._device = resolved_device
        self._dtype = use_dtype

    # ------------------------------------------------------------------
    # introspection helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def n_heads(self) -> int:
        return self._n_heads

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------

    def _random_init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        b, t = input_ids.shape
        t = min(t, self._max_seq_len)
        ids = input_ids[:, -t:].to(self._device)
        pos = torch.arange(t, device=self._device).unsqueeze(0).expand(b, t)
        x = self.token_embed(ids) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        self._last_hidden = x.detach()
        return logits, x

    def predict_next(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for the next token."""

        logits, _ = self.forward(input_ids)
        last = logits[:, -1, :]
        return F.softmax(last.float(), dim=-1)

    def get_hidden_state(self) -> torch.Tensor:
        if self._last_hidden is not None:
            return self._last_hidden
        return torch.zeros(
            (1, 1, self._d_model), device=self._device, dtype=self._dtype
        )

    # ------------------------------------------------------------------
    # parameter snapshot (used by information-gain curiosity signal)
    # ------------------------------------------------------------------

    def get_parameter_snapshot(
        self, sample_ratio: float = 1.0
    ) -> dict[str, dict[str, float]]:
        snap: dict[str, dict[str, float]] = {}
        with torch.no_grad():
            for name, param in self.named_parameters():
                flat = param.detach().float().reshape(-1)
                if sample_ratio < 1.0 and flat.numel() > 16:
                    k = max(16, int(flat.numel() * sample_ratio))
                    idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
                    flat = flat.index_select(0, idx)
                snap[name] = {
                    "mean": float(flat.mean().item()),
                    "std": float(flat.std().item()),
                }
        return snap

    # ------------------------------------------------------------------
    # growth primitives (self-growing brain)
    # ------------------------------------------------------------------

    def estimate_memory_bytes(
        self,
        extra_layers: int = 0,
        new_d_model: Optional[int] = None,
    ) -> int:
        """Rough memory estimate for the (possibly grown) network.

        Counts parameters only; caller adds activation/buffer overhead.
        Used by :class:`eva.evolution.evolver.Evolver` to refuse growth
        that would exceed ``hardware.max_ram_gb``.
        """

        d = new_d_model if new_d_model is not None else self._d_model
        n = self.n_layers + extra_layers
        # rough per-block parameter count: attn (4*d^2) + ffn (8*d^2)
        block_params = 12 * d * d
        total = (
            self._vocab_size * d  # token embed
            + self._max_seq_len * d  # positional embed
            + n * block_params
            + d  # final ln
            + d * self._vocab_size  # head
        )
        bytes_per_param = 2 if self._dtype in (torch.float16, torch.bfloat16) else 4
        return total * bytes_per_param

    @torch.no_grad()
    def grow_depth(self, extra_layers: int = 1) -> int:
        """Append ``extra_layers`` transformer blocks initialised near-identity.

        The new block's residual output is initialised close to zero so the
        freshly grown brain behaves almost identically to its previous self
        and can learn to use the new capacity over time.
        """

        added = 0
        for _ in range(extra_layers):
            block = _TransformerBlock(self._d_model, self._n_heads).to(
                device=self._device, dtype=self._dtype
            )
            # near-identity: zero the second linear layer of FFN and the
            # attn output projection so the residual add is ~0.
            for module in (block.ff[-1], block.attn.out):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            self.blocks.append(block)
            added += 1
        logger.info(
            "BabyBrain.grow_depth: +%d layers (now %d, params=%d)",
            added,
            len(self.blocks),
            self.parameter_count,
        )
        return added

    @torch.no_grad()
    def grow_width(self, new_d_model: int) -> int:
        """Increase ``d_model`` to ``new_d_model`` (identity-preserving).

        Implementation note: widening every linear layer without breaking
        semantics is surgery-heavy. We do a simpler, still-safe variant —
        we *rebuild* the brain with the new width and copy the overlapping
        slice of the old weights in-place, padding the rest with small
        random values. The result is not perfectly identity-preserving but
        avoids catastrophic forgetting in practice for small increments.
        """

        if new_d_model <= self._d_model:
            return 0
        if new_d_model % self._n_heads != 0:
            raise ValueError(
                f"new_d_model ({new_d_model}) must be divisible by "
                f"n_heads ({self._n_heads})"
            )

        logger.info(
            "BabyBrain.grow_width: %d -> %d", self._d_model, new_d_model
        )
        old_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        old_d = self._d_model

        # Rebuild
        self._d_model = new_d_model
        self.token_embed = nn.Embedding(self._vocab_size, new_d_model)
        self.pos_embed = nn.Embedding(self._max_seq_len, new_d_model)
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(new_d_model, self._n_heads)
                for _ in range(len(old_state_layers(old_state)))
            ]
        )
        self.ln_f = nn.LayerNorm(new_d_model)
        self.head = nn.Linear(new_d_model, self._vocab_size, bias=False)
        self.apply(self._random_init)
        self.to(self._device, dtype=self._dtype)

        # Copy overlapping slices
        with torch.no_grad():
            for name, new_p in self.state_dict().items():
                if name in old_state:
                    old_p = old_state[name]
                    slices = tuple(
                        slice(0, min(a, b))
                        for a, b in zip(old_p.shape, new_p.shape, strict=False)
                    )
                    new_p[slices] = old_p[slices].to(
                        device=new_p.device, dtype=new_p.dtype
                    )
        return new_d_model - old_d


def old_state_layers(state: dict[str, torch.Tensor]) -> list[int]:
    """Helper — count how many transformer blocks are in a state dict."""

    idx = set()
    for k in state.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                idx.add(int(parts[1]))
    return sorted(idx)
