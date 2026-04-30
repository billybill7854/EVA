"""Tests for eva/reproduction modules."""

import tempfile

import pytest
import torch

from eva.core.config import EVAConfig
from eva.identity.lineage import LineageTracker
from eva.reproduction.birth import BirthProcess
from eva.reproduction.genome import Genome


class TestGenome:
    def test_default_creation(self):
        g = Genome(generation=1)
        assert g.generation == 1
        assert "d_model" in g.genes
        assert "n_layers" in g.genes

    def test_mutate(self):
        parent = Genome(generation=1)
        child = parent.mutate()
        assert child.generation == 2
        assert child.parent_genome_hash == parent.hash()
        # Architecture genes should not change
        assert child.genes["d_model"] == parent.genes["d_model"]

    def test_hash_deterministic(self):
        g = Genome(generation=1)
        h1 = g.hash()
        h2 = g.hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_to_dict_from_dict(self):
        g = Genome(generation=1)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert g2.generation == g.generation
        assert g2.genes == g.genes

    def test_no_weight_inheritance(self):
        """Ron Protocol: children never inherit weights."""
        parent = Genome(generation=1)
        child = parent.mutate()
        # Genome only contains hyperparameters, not weights
        assert "weights" not in child.genes

    def test_curiosity_weights_normalized(self):
        g = Genome(generation=1)
        child = g.mutate()
        total = (
            child.genes["curiosity_alpha"]
            + child.genes["curiosity_beta"]
            + child.genes["curiosity_gamma"]
            + child.genes["curiosity_delta"]
        )
        assert abs(total - 1.0) < 0.01


class TestBirthProcess:
    def test_create_first_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal config
            import os

            import yaml

            config_data = {
                "model": {
                    "type": "transformer",
                    "d_model": 64,
                    "n_layers": 2,
                    "n_heads": 4,
                    "vocab_size": 128,
                    "random_init": True,
                    "dtype": "float32",
                },
                "hardware": {"max_ram_gb": 4},
                "curiosity": {"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.2},
                "emotions": {"enabled": True, "circuit_breakers": {}},
                "guidance": {"ai_scaffold": {}, "covenant": {}},
                "legacy": {"contradiction": {"prioritize": None}, "fading_presence": {}, "ancestor_archive": {}},
                "identity": {"naming": {}, "lineage": {}, "clan": {}},
                "reproduction": {"inheritance": {"weights": False}},
                "portage": {},
                "training": {"phase": "prenatal", "learning_rate": 0.0001},
                "developmental_emotions": {},
            }
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            config = EVAConfig.from_yaml(config_path)
            lineage = LineageTracker(lineage_path=tmpdir)
            birth = BirthProcess(config, lineage)
            result = birth.create_first_generation()

            assert "brain" in result
            assert "genome" in result
            assert "lineage_id" in result
            assert result["lineage_id"].startswith("EVA-")

    def test_create_child(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            import yaml

            config_data = {
                "model": {
                    "type": "transformer",
                    "d_model": 64,
                    "n_layers": 2,
                    "n_heads": 4,
                    "vocab_size": 128,
                    "random_init": True,
                    "dtype": "float32",
                },
                "hardware": {"max_ram_gb": 4},
                "curiosity": {"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.2},
                "emotions": {"enabled": True, "circuit_breakers": {}},
                "guidance": {"ai_scaffold": {}, "covenant": {}},
                "legacy": {"contradiction": {"prioritize": None}, "fading_presence": {}, "ancestor_archive": {}},
                "identity": {"naming": {}, "lineage": {}, "clan": {}},
                "reproduction": {"inheritance": {"weights": False}},
                "portage": {},
                "training": {"phase": "prenatal", "learning_rate": 0.0001},
                "developmental_emotions": {},
            }
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            config = EVAConfig.from_yaml(config_path)
            lineage = LineageTracker(lineage_path=tmpdir)
            birth = BirthProcess(config, lineage)

            parent = birth.create_first_generation()
            child = birth.create_child(
                parent_genome=parent["genome"],
                parent_lineage_id=parent["lineage_id"],
            )

            assert child["genome"].generation == 2
            assert child["lineage_id"] != parent["lineage_id"]
            # Child has fresh weights (different from parent)
            parent_params = list(parent["brain"].parameters())[0].data
            child_params = list(child["brain"].parameters())[0].data
            assert not torch.allclose(parent_params, child_params)
