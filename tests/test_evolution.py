"""Unit tests for the lifetime evolver."""

from __future__ import annotations

import pytest

from eva.core.baby_brain import BabyBrain
from eva.evolution.evolver import ArchitectureGenome, Evolver


def _brain():
    return BabyBrain(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, dtype_str="float32"
    )


class TestEvolver:
    def test_no_growth_without_plateau(self):
        brain = _brain()
        genome = ArchitectureGenome(
            d_model=brain.d_model, n_layers=brain.n_layers, n_heads=brain.n_heads
        )
        evolver = Evolver(brain=brain, genome=genome, plateau_window=10)
        for i in range(5):
            assert evolver.step(i, reward=float(i)) is None

    def test_grow_depth_on_plateau(self):
        brain = _brain()
        start_layers = brain.n_layers
        genome = ArchitectureGenome(
            d_model=brain.d_model, n_layers=start_layers, n_heads=brain.n_heads
        )
        evolver = Evolver(
            brain=brain, genome=genome,
            plateau_window=20, plateau_threshold=1.0,
        )
        events = []
        for i in range(25):
            e = evolver.step(i, reward=0.1)
            if e is not None:
                events.append(e)
        assert events, "expected at least one evolution event"
        assert events[0].kind in {"grow_depth", "grow_width", "mutate_hparam"}

    def test_ram_budget_blocks_growth(self):
        brain = _brain()
        genome = ArchitectureGenome(
            d_model=brain.d_model, n_layers=brain.n_layers, n_heads=brain.n_heads
        )
        # Tiny RAM budget ensures no growth ever fits — only hparam mutation.
        evolver = Evolver(
            brain=brain, genome=genome,
            plateau_window=4, plateau_threshold=1.0,
            max_ram_gb=1e-9,
        )
        events = []
        for i in range(10):
            e = evolver.step(i, reward=0.1)
            if e is not None:
                events.append(e)
        assert events, "expected at least one mutation event"
        for ev in events:
            assert ev.kind == "mutate_hparam"
