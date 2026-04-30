"""Unit tests for PersistentMemoryStore."""

from __future__ import annotations

from pathlib import Path

from eva.memory.persistent import Insight, PersistentMemoryStore, StoredEpisode


def test_round_trip_episodes(tmp_path: Path) -> None:
    store = PersistentMemoryStore(tmp_path / "mem.db")
    store.record_episode(
        StoredEpisode(
            step=1, action=2, outcome=3, surprise=0.5,
            emotional_importance=0.4, source_tag="self",
        )
    )
    episodes = store.recent_episodes()
    assert len(episodes) == 1
    assert episodes[0]["action"] == 2


def test_insights_signal_vs_noise(tmp_path: Path) -> None:
    store = PersistentMemoryStore(tmp_path / "mem.db")
    store.record_insight(
        Insight(step=1, kind="pattern", description="signal", confidence=0.9)
    )
    store.record_insight(
        Insight(step=2, kind="pattern", description="noise", confidence=0.1)
    )
    insights = store.recent_insights()
    assert len(insights) == 2
    # recent_insights returns newest-first; noise was inserted last.
    assert insights[0]["confidence"] < insights[1]["confidence"]


def test_thoughts_and_genome_history(tmp_path: Path) -> None:
    store = PersistentMemoryStore(tmp_path / "mem.db")
    store.record_thought(step=1, category="input", content="hi")
    store.record_genome(step=1, genes={"d_model": 64}, parameter_count=1000)
    store.record_genome(step=2, genes={"d_model": 128}, parameter_count=4000)
    thoughts = store.recent_thoughts()
    assert thoughts[0]["content"] == "hi"
    hist = store.genome_history()
    assert [h["parameter_count"] for h in hist] == [1000, 4000]


def test_persistence_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    store = PersistentMemoryStore(db)
    store.record_thought(step=1, category="test", content="persist me")
    store.close()

    store2 = PersistentMemoryStore(db)
    thoughts = store2.recent_thoughts()
    assert any(t["content"] == "persist me" for t in thoughts)
    store2.close()
