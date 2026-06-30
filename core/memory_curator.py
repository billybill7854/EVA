from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any


@dataclass
class CuratedMemory:
    memory_type: str
    content: str
    importance_score: int
    tags: List[str]
    expires_at: Optional[datetime] = None


class MemoryCurator:
    """
    Lightweight, heuristic memory curation.
    Goal: store only stable, reusable user facts/preferences/goals.
    """

    PREFERENCE_PATTERNS = [
        re.compile(r"\b(i (?:really )?(?:like|love|prefer|enjoy))\b", re.I),
        re.compile(r"\b(my favorite)\b", re.I),
    ]
    FACT_PATTERNS = [
        re.compile(r"\b(i am|i'm|my name is|call me)\b", re.I),
        re.compile(r"\b(my (?:phone|email|address) is)\b", re.I),
        re.compile(r"\b(i live in|i work at|i study at)\b", re.I),
    ]
    GOAL_PATTERNS = [
        re.compile(r"\b(i want to|my goal is|i'm trying to|i plan to)\b", re.I),
    ]

    def extract(self, text: str) -> List[CuratedMemory]:
        t = (text or "").strip()
        if not t:
            return []

        # Avoid storing commands or one-off operational messages
        if len(t) < 12:
            return []
        if t.startswith(("/", "http://", "https://")):
            return []

        lowered = t.lower()
        candidates: List[CuratedMemory] = []

        def tag_topic() -> List[str]:
            tags = []
            if any(w in lowered for w in ["work", "job", "career"]):
                tags.append("topic:work")
            if any(w in lowered for w in ["study", "school", "exam", "class"]):
                tags.append("topic:study")
            if any(w in lowered for w in ["health", "gym", "sleep", "diet"]):
                tags.append("topic:health")
            return tags

        tags = tag_topic()

        if any(p.search(t) for p in self.PREFERENCE_PATTERNS):
            candidates.append(
                CuratedMemory(
                    memory_type="preference",
                    content=t,
                    importance_score=5,
                    tags=tags + ["kind:preference"],
                    expires_at=datetime.utcnow() + timedelta(days=365),
                )
            )

        if any(p.search(t) for p in self.GOAL_PATTERNS):
            candidates.append(
                CuratedMemory(
                    memory_type="goal",
                    content=t,
                    importance_score=6,
                    tags=tags + ["kind:goal"],
                    expires_at=datetime.utcnow() + timedelta(days=180),
                )
            )

        if any(p.search(t) for p in self.FACT_PATTERNS):
            candidates.append(
                CuratedMemory(
                    memory_type="fact",
                    content=t,
                    importance_score=7,
                    tags=tags + ["kind:fact"],
                    expires_at=None,  # facts don't expire by default
                )
            )

        # De-dup by memory_type (keep highest importance)
        best: Dict[str, CuratedMemory] = {}
        for c in candidates:
            if c.memory_type not in best or c.importance_score > best[c.memory_type].importance_score:
                best[c.memory_type] = c
        return list(best.values())

