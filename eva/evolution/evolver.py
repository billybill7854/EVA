"""Lifetime evolver — grow the brain, mutate hyperparameters.

``Evolver.step(...)`` is called once per training step (or any
convenient interval) with the current curiosity reward. When the
signal plateaus for ``plateau_window`` steps the evolver:

1. Proposes either a width or depth growth (whichever is cheaper under
   the RAM budget).
2. Checks the estimate against ``max_ram_gb``.
3. Applies the growth via :meth:`BabyBrain.grow_depth` /
   :meth:`BabyBrain.grow_width`.
4. Mutates a handful of training hyperparameters (lr, temperature,
   gradient clip) within bounded ranges.
5. Returns an :class:`EvolutionEvent` so callers can log / display it.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from eva.core.baby_brain import BabyBrain

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureGenome:
    """Runtime-mutable architectural genes."""

    d_model: int
    n_layers: int
    n_heads: int
    lr: float = 1e-4
    exploration_temperature: float = 1.0
    grad_clip: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "lr": self.lr,
            "exploration_temperature": self.exploration_temperature,
            "grad_clip": self.grad_clip,
        }


@dataclass
class EvolutionEvent:
    step: int
    kind: str  # "grow_depth" | "grow_width" | "mutate_hparam" | "none"
    detail: str
    genome_after: dict[str, Any] = field(default_factory=dict)


class Evolver:
    """Per-lifetime evolution controller."""

    def __init__(
        self,
        brain: BabyBrain,
        genome: ArchitectureGenome,
        max_ram_gb: float = 8.0,
        plateau_window: int = 200,
        plateau_threshold: float = 1e-3,
        max_d_model: int = 1024,
        max_layers: int = 32,
        seed: Optional[int] = None,
        on_event: Optional[Callable[[EvolutionEvent], None]] = None,
    ) -> None:
        self.brain = brain
        self.genome = genome
        self.max_ram_gb = max_ram_gb
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.max_d_model = max_d_model
        self.max_layers = max_layers
        self._rewards: deque[float] = deque(maxlen=plateau_window)
        self._rng = random.Random(seed)
        self._on_event = on_event

    # ------------------------------------------------------------------

    def _plateaued(self) -> bool:
        if len(self._rewards) < self.plateau_window:
            return False
        first_half = list(self._rewards)[: self.plateau_window // 2]
        second_half = list(self._rewards)[self.plateau_window // 2 :]
        if not first_half or not second_half:
            return False
        delta = abs(
            sum(second_half) / len(second_half)
            - sum(first_half) / len(first_half)
        )
        return delta < self.plateau_threshold

    def _would_fit(
        self,
        *,
        extra_layers: int = 0,
        new_d_model: Optional[int] = None,
    ) -> bool:
        bytes_needed = self.brain.estimate_memory_bytes(
            extra_layers=extra_layers,
            new_d_model=new_d_model,
        )
        # Roughly triple the param footprint to account for activations + optim.
        budget = int(self.max_ram_gb * (1024 ** 3))
        return bytes_needed * 3 < budget

    # ------------------------------------------------------------------

    def step(self, step: int, reward: float) -> Optional[EvolutionEvent]:
        self._rewards.append(reward)
        if not self._plateaued():
            return None
        # Clear the window so we don't fire again immediately.
        self._rewards.clear()

        # Prefer depth growth (cheaper) first, then width, then hparam mutation.
        grew = False
        if (
            self.genome.n_layers < self.max_layers
            and self._would_fit(extra_layers=1)
        ):
            added = self.brain.grow_depth(1)
            self.genome.n_layers += added
            event = EvolutionEvent(
                step=step,
                kind="grow_depth",
                detail=f"+{added} layer(s); now {self.genome.n_layers}",
                genome_after=self.genome.to_dict(),
            )
            grew = True
        elif (
            self.genome.d_model + self.genome.n_heads <= self.max_d_model
            and self._would_fit(
                new_d_model=self.genome.d_model + self.genome.n_heads
            )
        ):
            # Grow width by one head's worth — keeps divisibility invariant.
            new_width = self.genome.d_model + self.genome.n_heads
            self.brain.grow_width(new_width)
            self.genome.d_model = new_width
            event = EvolutionEvent(
                step=step,
                kind="grow_width",
                detail=f"d_model -> {new_width}",
                genome_after=self.genome.to_dict(),
            )
            grew = True

        if not grew:
            # Mutate a hyperparameter instead.
            target = self._rng.choice(("lr", "exploration_temperature"))
            factor = self._rng.choice((0.9, 1.1))
            if target == "lr":
                self.genome.lr = max(1e-6, min(1e-2, self.genome.lr * factor))
                detail = f"lr -> {self.genome.lr:.6f}"
            else:
                self.genome.exploration_temperature = max(
                    0.3,
                    min(2.0, self.genome.exploration_temperature * factor),
                )
                detail = (
                    f"exploration_temperature -> "
                    f"{self.genome.exploration_temperature:.3f}"
                )
            event = EvolutionEvent(
                step=step,
                kind="mutate_hparam",
                detail=detail,
                genome_after=self.genome.to_dict(),
            )

        logger.info("Evolver: %s — %s", event.kind, event.detail)
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception as exc:  # pragma: no cover — best-effort
                logger.exception("Evolver on_event callback raised: %s", exc)
        return event
