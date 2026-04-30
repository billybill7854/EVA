"""Tests for eva/memory modules."""

import pytest
import torch

from eva.memory.episodic import Episode, EpisodicMemory


class TestEpisodicMemory:
    def _make_episode(self, timestamp: int = 0, importance: float = 0.5) -> Episode:
        return Episode(
            state_embedding=torch.randn(64),
            action=1,
            outcome=2,
            surprise=0.5,
            emotional_importance=importance,
            source_tag="self",
            timestamp=timestamp,
        )

    def test_store_and_size(self):
        mem = EpisodicMemory(capacity=100)
        for i in range(10):
            mem.store(self._make_episode(timestamp=i))
        assert mem.size() == 10

    def test_capacity_eviction(self):
        mem = EpisodicMemory(capacity=5)
        for i in range(10):
            mem.store(self._make_episode(timestamp=i, importance=float(i)))
        assert mem.size() == 5

    def test_recall(self):
        mem = EpisodicMemory(capacity=100)
        ep = self._make_episode()
        mem.store(ep)
        results = mem.recall(ep.state_embedding, k=1)
        assert len(results) == 1

    def test_recall_empty(self):
        mem = EpisodicMemory(capacity=100)
        results = mem.recall(torch.randn(64), k=5)
        assert len(results) == 0

    def test_consolidate(self):
        mem = EpisodicMemory(capacity=100)
        embedding = torch.randn(64)
        ep1 = Episode(embedding, 1, 2, 0.1, 0.05, "self", 0)
        ep2 = Episode(embedding + torch.randn(64) * 0.001, 1, 2, 0.1, 0.05, "self", 1)
        mem.store(ep1)
        mem.store(ep2)
        old_size = mem.size()
        mem.consolidate()
        assert mem.size() <= old_size

    def test_clear(self):
        mem = EpisodicMemory(capacity=100)
        for i in range(5):
            mem.store(self._make_episode(timestamp=i))
        mem.clear()
        assert mem.size() == 0
