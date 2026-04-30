"""Memory Inspector — detailed views of episodic memories with filtering.

Provides human-readable inspection of EVA's episodic memories with
comprehensive filtering capabilities, consolidation event tracking,
and memory analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from eva.memory.episodic import Episode, EpisodicMemory


@dataclass
class MemoryView:
    """Human-readable view of a memory.

    Attributes:
        timestamp: When the memory was formed.
        content: Truncated content for display.
        importance: Emotional importance score.
        emotional_valence: Emotional valence of the memory.
        source: Source tag (self, human, scaffold, ancestor).
        tags: Additional tags for categorization.
        retrieval_count: Number of times this memory was retrieved.
    """

    timestamp: datetime
    content: str
    importance: float
    emotional_valence: float
    source: str
    tags: list[str]
    retrieval_count: int


@dataclass
class ConsolidationEvent:
    """Record of a memory consolidation event.

    Attributes:
        timestamp: When consolidation occurred.
        memories_before: Number of memories before consolidation.
        memories_after: Number of memories after consolidation.
        retention_rate: Percentage of memories retained.
    """

    timestamp: datetime
    memories_before: int
    memories_after: int
    retention_rate: float


class MemoryInspector:
    """Inspect and analyze episodic memories.

    Provides detailed views of EVA's episodic memories with filtering
    capabilities, consolidation tracking, and memory analytics.

    Args:
        memory_system: The episodic memory system to inspect.
    """

    def __init__(self, memory_system: EpisodicMemory):
        self.memory_system = memory_system
        self._consolidation_events: list[ConsolidationEvent] = []
        self._retrieval_counts: dict[int, int] = {}  # episode index -> count

    def get_memories(
        self,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        importance_min: float = 0.0,
        valence: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> list[MemoryView]:
        """Get memories with filtering.

        Args:
            time_start: Filter memories after this time.
            time_end: Filter memories before this time.
            importance_min: Minimum importance threshold.
            valence: Filter by emotional valence (positive, negative, neutral).
            source: Filter by source tag (self, human, scaffold, ancestor).
            limit: Maximum number of memories to return.

        Returns:
            List of memory views matching the filters.
        """
        memories = self._get_all_episodes()

        # Apply filters
        if time_start:
            memories = [m for m in memories if self._episode_to_datetime(m) >= time_start]
        if time_end:
            memories = [m for m in memories if self._episode_to_datetime(m) <= time_end]
        if importance_min > 0:
            memories = [m for m in memories if m.emotional_importance >= importance_min]
        if valence:
            memories = [m for m in memories if self._match_valence(m, valence)]
        if source:
            memories = [m for m in memories if m.source_tag == source]

        # Sort by timestamp (most recent first)
        memories = sorted(memories, key=lambda m: m.timestamp, reverse=True)

        # Convert to views
        views = [self._create_view(m, idx) for idx, m in enumerate(memories[:limit])]
        return views

    def _get_all_episodes(self) -> list[Episode]:
        """Get all episodes from the memory system."""
        # Access the internal buffer directly
        return list(self.memory_system._buffer)

    def _episode_to_datetime(self, episode: Episode) -> datetime:
        """Convert episode timestamp (step number) to datetime.

        For now, we use a simple conversion assuming each step is 1 second.
        In a real system, this would use actual timestamps.
        """
        # Use a fixed reference time and add steps to maintain consistent ordering
        # This ensures that higher timestamps are more recent
        reference_time = datetime(2020, 1, 1)
        return reference_time + timedelta(seconds=episode.timestamp)

    def _match_valence(self, episode: Episode, valence: str) -> bool:
        """Check if episode matches valence filter.

        Args:
            episode: The episode to check.
            valence: The valence filter (positive, negative, neutral).

        Returns:
            True if the episode matches the valence filter.
        """
        # Use surprise as a proxy for emotional valence
        # Positive surprise = positive valence, negative surprise = negative valence
        val = episode.surprise
        if valence == "positive":
            return val > 0.2
        elif valence == "negative":
            return val < -0.2
        elif valence == "neutral":
            return -0.2 <= val <= 0.2
        return True

    def _create_view(self, episode: Episode, idx: int) -> MemoryView:
        """Create human-readable memory view.

        Args:
            episode: The episode to convert.
            idx: The index of the episode in the buffer.

        Returns:
            A memory view for display.
        """
        # Create content string from action and outcome
        content = f"Action: {episode.action}, Outcome: {episode.outcome}, Surprise: {episode.surprise:.3f}"

        # Truncate for display
        if len(content) > 200:
            content = content[:200] + "..."

        return MemoryView(
            timestamp=self._episode_to_datetime(episode),
            content=content,
            importance=episode.emotional_importance,
            emotional_valence=episode.surprise,  # Using surprise as valence proxy
            source=episode.source_tag,
            tags=[],  # Episodes don't have tags yet, placeholder for future
            retrieval_count=self._retrieval_counts.get(idx, 0),
        )

    def record_retrieval(self, episode_idx: int) -> None:
        """Record that an episode was retrieved.

        Args:
            episode_idx: The index of the retrieved episode.
        """
        self._retrieval_counts[episode_idx] = self._retrieval_counts.get(episode_idx, 0) + 1

    def get_consolidation_events(self) -> list[ConsolidationEvent]:
        """Get memory consolidation events.

        Returns:
            List of consolidation events that have occurred.
        """
        return list(self._consolidation_events)

    def record_consolidation(self, memories_before: int, memories_after: int) -> None:
        """Record a consolidation event.

        Args:
            memories_before: Number of memories before consolidation.
            memories_after: Number of memories after consolidation.
        """
        retention_rate = memories_after / memories_before if memories_before > 0 else 1.0

        event = ConsolidationEvent(
            timestamp=datetime.now(),
            memories_before=memories_before,
            memories_after=memories_after,
            retention_rate=retention_rate,
        )

        self._consolidation_events.append(event)

    def get_retrieval_patterns(self) -> dict[str, Any]:
        """Get patterns of memory retrieval.

        Returns:
            Dictionary containing retrieval statistics and patterns.
        """
        memories = self._get_all_episodes()

        if not memories:
            return {
                "most_retrieved": [],
                "retrieval_frequency": 0.0,
            }

        # Get episodes with their retrieval counts
        episodes_with_counts = [
            (idx, ep, self._retrieval_counts.get(idx, 0))
            for idx, ep in enumerate(memories)
        ]

        # Sort by retrieval count
        most_retrieved = sorted(
            episodes_with_counts,
            key=lambda x: x[2],
            reverse=True
        )[:10]

        # Convert to views
        most_retrieved_views = [
            self._create_view(ep, idx)
            for idx, ep, count in most_retrieved
        ]

        # Calculate average retrieval frequency
        total_retrievals = sum(self._retrieval_counts.values())
        avg_frequency = total_retrievals / len(memories) if memories else 0.0

        return {
            "most_retrieved": most_retrieved_views,
            "retrieval_frequency": avg_frequency,
        }

    def get_formation_rate(self, time_window: timedelta = timedelta(hours=1)) -> float:
        """Get memory formation rate.

        Args:
            time_window: Time window to measure formation rate.

        Returns:
            Memory formation rate (memories per hour).
        """
        now = datetime.now()
        memories = self._get_all_episodes()

        # Filter to recent memories
        recent = [
            m for m in memories
            if now - self._episode_to_datetime(m) <= time_window
        ]

        # Calculate rate per hour
        rate = len(recent) / time_window.total_seconds() * 3600
        return rate

    def get_retention_rate(self) -> float:
        """Get memory retention rate after consolidation.

        Returns:
            Average retention rate across all consolidation events.
        """
        if not self._consolidation_events:
            return 1.0  # No consolidation yet, 100% retention

        # Calculate average retention rate
        total_rate = sum(event.retention_rate for event in self._consolidation_events)
        avg_rate = total_rate / len(self._consolidation_events)

        return avg_rate
