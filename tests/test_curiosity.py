"""Tests for eva/curiosity modules."""

import pytest
import torch

from eva.core.baby_brain import BabyBrain
from eva.curiosity.empowerment import EmpowermentModule
from eva.curiosity.information_gain import InformationGainModule
from eva.curiosity.novelty import NoveltyModule
from eva.curiosity.prediction_error import PredictionErrorModule
from eva.curiosity.reward import CuriosityEngine


class TestPredictionError:
    def test_compute(self):
        mod = PredictionErrorModule()
        dist = torch.softmax(torch.randn(128), dim=0)
        error = mod.compute(dist, 5)
        assert error > 0

    def test_relative_surprise(self):
        mod = PredictionErrorModule()
        dist = torch.softmax(torch.randn(128), dim=0)
        error = mod.compute(dist, 5)
        surprise = mod.get_relative_surprise(error)
        assert surprise > 0

    def test_reset(self):
        mod = PredictionErrorModule()
        dist = torch.softmax(torch.randn(128), dim=0)
        mod.compute(dist, 5)
        mod.reset()
        assert not mod._initialized


class TestInformationGain:
    def test_snapshot_and_compute(self):
        brain = BabyBrain(vocab_size=128, d_model=64, n_layers=2, n_heads=4, dtype_str="float32")
        mod = InformationGainModule()
        mod.snapshot_before(brain)
        # Without weight update, gain should be ~0
        gain = mod.compute(brain)
        assert gain == pytest.approx(0.0, abs=1e-6)

    def test_compute_without_snapshot(self):
        brain = BabyBrain(vocab_size=128, d_model=64, n_layers=2, n_heads=4, dtype_str="float32")
        mod = InformationGainModule()
        gain = mod.compute(brain)
        assert gain == 0.0


class TestNovelty:
    def test_novel_state_high_score(self):
        mod = NoveltyModule()
        score = mod.compute("new_state")
        assert score > 0.5

    def test_familiar_state_lower_score(self):
        mod = NoveltyModule()
        mod.compute("state_a")
        score2 = mod.compute("state_a")
        assert score2 < 0.7  # Second visit = lower novelty

    def test_hash_state(self):
        mod = NoveltyModule()
        state = torch.randn(1, 10, 64)
        h = mod.hash_state(state)
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex

    def test_reset(self):
        mod = NoveltyModule()
        mod.compute("x")
        mod.reset()
        score = mod.compute("x")
        assert score > 0.5  # Fresh after reset


class TestEmpowerment:
    def test_insufficient_data(self):
        mod = EmpowermentModule()
        score = mod.compute([])
        assert score == 0.5

    def test_with_outcomes(self):
        mod = EmpowermentModule()
        outcomes = [torch.randn(64) for _ in range(10)]
        score = mod.compute(outcomes)
        assert 0.0 <= score <= 1.0

    def test_add_outcome(self):
        mod = EmpowermentModule()
        mod.add_outcome(torch.randn(64))
        assert len(mod.get_recent_outcomes()) == 1


class TestCuriosityEngine:
    def test_compute_reward(self):
        brain = BabyBrain(vocab_size=128, d_model=64, n_layers=2, n_heads=4, dtype_str="float32")
        engine = CuriosityEngine(alpha=0.3, beta=0.3, gamma=0.2, delta=0.2)
        engine.prepare(brain)

        input_ids = torch.tensor([[1, 2, 3]])
        predicted = brain.predict_next(input_ids)
        hidden = brain.get_hidden_state()

        reward, breakdown = engine.compute_reward(
            predicted, 4, brain, hidden, [torch.randn(64) for _ in range(5)]
        )
        assert isinstance(reward, float)
        assert "total" in breakdown
        assert "prediction_error" in breakdown
        assert "info_gain" in breakdown
        assert "novelty" in breakdown
        assert "empowerment" in breakdown
