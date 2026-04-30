"""Tests for eva/core modules: config, baby_brain, tokenizer."""

import os
import tempfile

import pytest
import torch
import yaml

from eva.core.baby_brain import BabyBrain
from eva.core.config import EVAConfig
from eva.core.tokenizer import (
    ANCESTOR_ID,
    BOS_ID,
    EOS_ID,
    HUMAN_ID,
    PAD_ID,
    SCAFFOLD_ID,
    SELF_ID,
    UNK_ID,
    EVATokenizer,
)

# --- Config Tests ---

class TestEVAConfig:
    def _make_config_file(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(data, f)
        f.close()
        return f.name

    def _default_data(self) -> dict:
        return {
            "model": {
                "type": "transformer",
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "vocab_size": 128,
                "random_init": True,
                "dtype": "float16",
            },
            "hardware": {"max_ram_gb": 4},
            "curiosity": {"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.2},
            "emotions": {"enabled": True, "circuit_breakers": {"valence_floor": -0.8, "arousal_ceiling": 0.95}},
            "guidance": {"ai_scaffold": {"always_available": True}, "covenant": {}},
            "legacy": {"contradiction": {"prioritize": None}, "fading_presence": {}, "ancestor_archive": {}},
            "identity": {"naming": {}, "lineage": {}, "clan": {}},
            "reproduction": {"inheritance": {"weights": False}},
            "portage": {},
            "training": {"phase": "prenatal", "learning_rate": 0.0001},
            "developmental_emotions": {},
        }

    def test_load_from_yaml(self):
        data = self._default_data()
        path = self._make_config_file(data)
        try:
            config = EVAConfig.from_yaml(path)
            assert config.model.d_model == 64
            assert config.model.n_layers == 2
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            EVAConfig.from_yaml("/nonexistent/path.yaml")

    def test_random_init_violation(self):
        data = self._default_data()
        data["model"]["random_init"] = False
        path = self._make_config_file(data)
        try:
            with pytest.raises(ValueError, match="RON PROTOCOL"):
                EVAConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_contradiction_must_be_null(self):
        data = self._default_data()
        data["legacy"]["contradiction"]["prioritize"] = "archive"
        path = self._make_config_file(data)
        try:
            with pytest.raises(ValueError, match="RON PROTOCOL"):
                EVAConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_children_weights_must_be_false(self):
        data = self._default_data()
        data["reproduction"]["inheritance"]["weights"] = True
        path = self._make_config_file(data)
        try:
            with pytest.raises(ValueError, match="RON PROTOCOL"):
                EVAConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_estimate_memory(self):
        data = self._default_data()
        path = self._make_config_file(data)
        try:
            config = EVAConfig.from_yaml(path)
            mem = config.estimate_memory_gb()
            assert mem > 0
        finally:
            os.unlink(path)


# --- BabyBrain Tests ---

class TestBabyBrain:
    def test_creation(self):
        brain = BabyBrain(vocab_size=128, d_model=64, n_layers=2, n_heads=4, dtype_str="float32")
        assert brain.parameter_count > 0
        assert brain.architecture in ("transformer", "mamba")

    def test_forward(self):
        brain = BabyBrain(vocab_size=128, d_model=64, n_layers=2, n_heads=4, dtype_str="float32")
        input_ids = torch.tensor([[1, 2, 3, 4]])
        logits, hidden = brain(input_ids)
        assert logits.shape == (1, 4, 128)
        assert hidden.shape == (1, 4, 64)

    def test_predict_next(self):
        brain = BabyBrain(vocab_size=128, d_model=64, n_layers=2, n_heads=4, dtype_str="float32")
        input_ids = torch.tensor([[1, 2, 3]])
        probs = brain.predict_next(input_ids)
        assert probs.shape == (1, 128)
        assert abs(probs.sum().item() - 1.0) < 0.01

    def test_get_hidden_state(self):
        brain = BabyBrain(vocab_size=128, d_model=64, n_layers=2, n_heads=4, dtype_str="float32")
        # Before any forward pass
        h = brain.get_hidden_state()
        assert h.shape == (1, 1, 64)

        # After forward pass
        brain(torch.tensor([[1, 2, 3]]))
        h = brain.get_hidden_state()
        assert h.shape[2] == 64

    def test_parameter_snapshot(self):
        brain = BabyBrain(vocab_size=128, d_model=64, n_layers=2, n_heads=4, dtype_str="float32")
        snap = brain.get_parameter_snapshot()
        assert len(snap) > 0
        for name, stats in snap.items():
            assert "mean" in stats
            assert "std" in stats


# --- Tokenizer Tests ---

class TestEVATokenizer:
    def test_creation(self):
        tok = EVATokenizer()
        assert tok.vocab_size > 8  # At least special + printable ASCII

    def test_special_tokens(self):
        tok = EVATokenizer()
        assert PAD_ID == 0
        assert UNK_ID == 1
        assert BOS_ID == 2
        assert EOS_ID == 3

    def test_encode_decode_roundtrip(self):
        tok = EVATokenizer()
        text = "hello world"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_source_tagging(self):
        tok = EVATokenizer()
        ids = tok.encode("test", source="human")
        assert HUMAN_ID in ids
        ids = tok.encode("test", source="scaffold")
        assert SCAFFOLD_ID in ids
        ids = tok.encode("test", source="ancestor")
        assert ANCESTOR_ID in ids

    def test_add_token(self):
        tok = EVATokenizer()
        old_size = tok.vocab_size
        new_id = tok.add_token("<NEW>")
        assert tok.vocab_size == old_size + 1
        # Adding same token returns same ID
        same_id = tok.add_token("<NEW>")
        assert same_id == new_id

    def test_get_source_tag(self):
        tok = EVATokenizer()
        ids = tok.encode("test", source="human")
        assert tok.get_source_tag(ids) == "human"
