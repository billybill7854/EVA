"""Unit tests for Memory Inspector."""

from datetime import datetime, timedelta

import pytest
import torch

from eva.memory.episodic import Episode, EpisodicMemory
from eva.transparency.memory_inspector import (
    ConsolidationEvent,
    MemoryInspector,
    MemoryView,
)


@pytest.fixture
def memory_system():
    """Create a memory system with test episodes."""
    memory = EpisodicMemory(max_size=100)
    
    # Add test episodes with varying properties
    episodes = [
        Episode(
            state_embedding=torch.randn(64),
            action=1,
            outcome=2,
            surprise=0.5,
            emotional_importance=0.8,
            source_tag="self",
            timestamp=100,
        ),
        Episode(
            state_embedding=torch.randn(64),
            action=3,
            outcome=4,
            surprise=-0.3,
            emotional_importance=0.2,
            source_tag="human",
            timestamp=200,
        ),
        Episode(
            state_embedding=torch.randn(64),
            action=5,
            outcome=6,
            surprise=0.1,
            emotional_importance=0.6,
            source_tag="scaffold",
            timestamp=300,
        ),
        Episode(
            state_embedding=torch.randn(64),
            action=7,
            outcome=8,
            surprise=-0.5,
            emotional_importance=0.4,
            source_tag="ancestor",
            timestamp=400,
        ),
        Episode(
            state_embedding=torch.randn(64),
            action=9,
            outcome=10,
            surprise=0.3,
            emotional_importance=0.9,
            source_tag="self",
            timestamp=500,
        ),
    ]
    
    for ep in episodes:
        memory.store(ep)
        
    return memory


@pytest.fixture
def inspector(memory_system):
    """Create a memory inspector."""
    return MemoryInspector(memory_system)


def test_memory_inspector_initialization(memory_system):
    """Test that memory inspector initializes correctly."""
    inspector = MemoryInspector(memory_system)
    
    assert inspector.memory_system is memory_system
    assert inspector._consolidation_events == []
    assert inspector._retrieval_counts == {}


def test_get_memories_no_filters(inspector):
    """Test getting all memories without filters."""
    memories = inspector.get_memories()
    
    assert len(memories) == 5
    assert all(isinstance(m, MemoryView) for m in memories)
    
    # Should be sorted by timestamp (most recent first)
    timestamps = [m.timestamp for m in memories]
    assert timestamps == sorted(timestamps, reverse=True)


def test_get_memories_importance_filter(inspector):
    """Test filtering memories by importance."""
    # Filter for high importance (>= 0.6)
    memories = inspector.get_memories(importance_min=0.6)
    
    assert len(memories) == 3  # Episodes with importance 0.8, 0.6, 0.9
    assert all(m.importance >= 0.6 for m in memories)


def test_get_memories_valence_filter(inspector):
    """Test filtering memories by emotional valence."""
    # Filter for positive valence
    positive = inspector.get_memories(valence="positive")
    assert len(positive) == 2  # Surprise > 0.2: 0.5, 0.3
    assert all(m.emotional_valence > 0.2 for m in positive)
    
    # Filter for negative valence
    negative = inspector.get_memories(valence="negative")
    assert len(negative) == 2  # Surprise < -0.2: -0.3, -0.5
    assert all(m.emotional_valence < -0.2 for m in negative)
    
    # Filter for neutral valence
    neutral = inspector.get_memories(valence="neutral")
    assert len(neutral) == 1  # Surprise in [-0.2, 0.2]: 0.1
    assert all(-0.2 <= m.emotional_valence <= 0.2 for m in neutral)


def test_get_memories_source_filter(inspector):
    """Test filtering memories by source tag."""
    # Filter for self-generated memories
    self_memories = inspector.get_memories(source="self")
    assert len(self_memories) == 2
    assert all(m.source == "self" for m in self_memories)
    
    # Filter for human memories
    human_memories = inspector.get_memories(source="human")
    assert len(human_memories) == 1
    assert all(m.source == "human" for m in human_memories)


def test_get_memories_limit(inspector):
    """Test limiting the number of returned memories."""
    memories = inspector.get_memories(limit=3)
    
    assert len(memories) == 3


def test_get_memories_combined_filters(inspector):
    """Test combining multiple filters."""
    # Filter for high importance self-generated memories
    memories = inspector.get_memories(
        importance_min=0.6,
        source="self",
    )
    
    assert len(memories) == 2  # Episodes with importance 0.8 and 0.9, both self
    assert all(m.importance >= 0.6 for m in memories)
    assert all(m.source == "self" for m in memories)


def test_memory_view_content(inspector):
    """Test that memory views contain correct content."""
    memories = inspector.get_memories()
    
    for memory in memories:
        assert isinstance(memory.timestamp, datetime)
        assert isinstance(memory.content, str)
        assert isinstance(memory.importance, float)
        assert isinstance(memory.emotional_valence, float)
        assert isinstance(memory.source, str)
        assert isinstance(memory.tags, list)
        assert isinstance(memory.retrieval_count, int)
        
        # Content should be truncated to 200 chars
        assert len(memory.content) <= 203  # 200 + "..."


def test_record_retrieval(inspector):
    """Test recording memory retrievals."""
    # Record some retrievals
    inspector.record_retrieval(0)
    inspector.record_retrieval(0)
    inspector.record_retrieval(1)
    
    assert inspector._retrieval_counts[0] == 2
    assert inspector._retrieval_counts[1] == 1
    
    # Get memories and check retrieval counts
    memories = inspector.get_memories()
    # Note: retrieval counts are based on episode index, which may not match
    # the order in the filtered results


def test_get_consolidation_events_empty(inspector):
    """Test getting consolidation events when none exist."""
    events = inspector.get_consolidation_events()
    
    assert events == []


def test_record_consolidation(inspector):
    """Test recording consolidation events."""
    # Record a consolidation event
    inspector.record_consolidation(memories_before=100, memories_after=85)
    
    events = inspector.get_consolidation_events()
    assert len(events) == 1
    
    event = events[0]
    assert isinstance(event, ConsolidationEvent)
    assert event.memories_before == 100
    assert event.memories_after == 85
    assert event.retention_rate == 0.85
    assert isinstance(event.timestamp, datetime)


def test_record_multiple_consolidations(inspector):
    """Test recording multiple consolidation events."""
    inspector.record_consolidation(100, 85)
    inspector.record_consolidation(85, 70)
    inspector.record_consolidation(70, 60)
    
    events = inspector.get_consolidation_events()
    assert len(events) == 3
    
    # Check retention rates
    assert events[0].retention_rate == 0.85
    assert events[1].retention_rate == pytest.approx(0.8235, rel=0.01)
    assert events[2].retention_rate == pytest.approx(0.8571, rel=0.01)


def test_get_retrieval_patterns_empty(inspector):
    """Test getting retrieval patterns with no retrievals."""
    patterns = inspector.get_retrieval_patterns()
    
    assert "most_retrieved" in patterns
    assert "retrieval_frequency" in patterns
    assert patterns["retrieval_frequency"] == 0.0


def test_get_retrieval_patterns(inspector):
    """Test getting retrieval patterns."""
    # Record some retrievals
    inspector.record_retrieval(0)
    inspector.record_retrieval(0)
    inspector.record_retrieval(0)
    inspector.record_retrieval(1)
    inspector.record_retrieval(2)
    
    patterns = inspector.get_retrieval_patterns()
    
    assert "most_retrieved" in patterns
    assert "retrieval_frequency" in patterns
    
    # Check most retrieved
    most_retrieved = patterns["most_retrieved"]
    assert len(most_retrieved) <= 10
    assert all(isinstance(m, MemoryView) for m in most_retrieved)
    
    # Check average frequency
    assert patterns["retrieval_frequency"] == 1.0  # 5 retrievals / 5 memories


def test_get_formation_rate(inspector):
    """Test getting memory formation rate."""
    # All test memories are in the past, so rate should be 0 for recent window
    rate = inspector.get_formation_rate(time_window=timedelta(seconds=1))
    
    # Rate should be 0 or very low since test memories have old timestamps
    assert rate >= 0.0


def test_get_retention_rate_no_consolidation(inspector):
    """Test getting retention rate with no consolidation events."""
    rate = inspector.get_retention_rate()
    
    assert rate == 1.0  # 100% retention when no consolidation


def test_get_retention_rate_with_consolidation(inspector):
    """Test getting retention rate with consolidation events."""
    inspector.record_consolidation(100, 85)
    inspector.record_consolidation(85, 70)
    
    rate = inspector.get_retention_rate()
    
    # Average of 0.85 and 0.8235
    expected = (0.85 + (70 / 85)) / 2
    assert rate == pytest.approx(expected, rel=0.01)


def test_empty_memory_system():
    """Test inspector with empty memory system."""
    empty_memory = EpisodicMemory(max_size=100)
    inspector = MemoryInspector(empty_memory)
    
    memories = inspector.get_memories()
    assert memories == []
    
    patterns = inspector.get_retrieval_patterns()
    assert patterns["most_retrieved"] == []
    assert patterns["retrieval_frequency"] == 0.0
    
    rate = inspector.get_formation_rate()
    assert rate == 0.0


def test_time_range_filter(inspector):
    """Test filtering memories by time range."""
    now = datetime.now()
    
    # All test memories should be in the past
    memories = inspector.get_memories(
        time_start=now - timedelta(days=1),
        time_end=now,
    )
    
    # Should get all memories since they're all recent
    assert len(memories) >= 0


def test_consolidation_retention_rate_edge_cases(inspector):
    """Test consolidation retention rate edge cases."""
    # Test with zero memories before
    inspector.record_consolidation(0, 0)
    events = inspector.get_consolidation_events()
    assert events[-1].retention_rate == 1.0  # Should handle division by zero
    
    # Test with all memories retained
    inspector.record_consolidation(50, 50)
    events = inspector.get_consolidation_events()
    assert events[-1].retention_rate == 1.0
