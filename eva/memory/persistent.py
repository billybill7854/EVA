"""Permanent memory — SQLite-backed episodic + insight store.

The in-memory :class:`eva.memory.episodic.EpisodicMemory` vanishes when
the process exits. :class:`PersistentMemoryStore` writes every episode
to a SQLite database so an EVA genuinely has a *life*, not just a
session. The store also records "insights" flagged by
:class:`eva.transparency.EmergenceEventDetector` separately from raw
episodes so the UI can distinguish signal from noise.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER NOT NULL,
    action INTEGER NOT NULL,
    outcome INTEGER NOT NULL,
    surprise REAL NOT NULL,
    emotional_importance REAL NOT NULL,
    source_tag TEXT NOT NULL,
    state_embedding BLOB,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER NOT NULL,
    kind TEXT NOT NULL,
    description TEXT NOT NULL,
    confidence REAL NOT NULL,
    data TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS thoughts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    context TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS genome_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER NOT NULL,
    genes TEXT NOT NULL,
    parameter_count INTEGER NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS self_modifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER NOT NULL,
    target TEXT NOT NULL,
    diff TEXT NOT NULL,
    outcome TEXT NOT NULL,
    reason TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_step ON episodes(step);
CREATE INDEX IF NOT EXISTS idx_insights_step ON insights(step);
CREATE INDEX IF NOT EXISTS idx_thoughts_step ON thoughts(step);
"""


@dataclass
class StoredEpisode:
    step: int
    action: int
    outcome: int
    surprise: float
    emotional_importance: float
    source_tag: str
    state_embedding: Optional[bytes] = None


@dataclass
class Insight:
    step: int
    kind: str
    description: str
    confidence: float
    data: dict[str, Any] = field(default_factory=dict)


class PersistentMemoryStore:
    """Thread-safe SQLite persistence for EVA's long-term memory."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None
        )
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # episodes
    # ------------------------------------------------------------------

    def record_episode(self, ep: StoredEpisode) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO episodes
                    (step, action, outcome, surprise,
                     emotional_importance, source_tag, state_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ep.step,
                    int(ep.action),
                    int(ep.outcome),
                    float(ep.surprise),
                    float(ep.emotional_importance),
                    ep.source_tag,
                    ep.state_embedding,
                ),
            )
            return int(cur.lastrowid or 0)

    def recent_episodes(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT step, action, outcome, surprise, emotional_importance,
                       source_tag, created_at
                FROM episodes ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # insights (signal vs noise)
    # ------------------------------------------------------------------

    def record_insight(self, insight: Insight) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO insights (step, kind, description, confidence, data)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    insight.step,
                    insight.kind,
                    insight.description,
                    float(insight.confidence),
                    json.dumps(insight.data),
                ),
            )
            return int(cur.lastrowid or 0)

    def recent_insights(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT step, kind, description, confidence, data, created_at "
                "FROM insights ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row, strict=False))
                try:
                    d["data"] = json.loads(d["data"]) if d["data"] else {}
                except json.JSONDecodeError:
                    d["data"] = {}
                rows.append(d)
            return rows

    # ------------------------------------------------------------------
    # thoughts (raw stream — can be noisy)
    # ------------------------------------------------------------------

    def record_thought(
        self,
        step: int,
        category: str,
        content: str,
        context: Optional[dict[str, Any]] = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO thoughts (step, category, content, context)
                VALUES (?, ?, ?, ?)
                """,
                (
                    step,
                    category,
                    content,
                    json.dumps(context or {}),
                ),
            )
            return int(cur.lastrowid or 0)

    def recent_thoughts(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT step, category, content, context, created_at "
                "FROM thoughts ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row, strict=False))
                try:
                    d["context"] = json.loads(d["context"]) if d["context"] else {}
                except json.JSONDecodeError:
                    d["context"] = {}
                rows.append(d)
            return rows

    # ------------------------------------------------------------------
    # genome history (for evolution UI)
    # ------------------------------------------------------------------

    def record_genome(
        self, step: int, genes: dict[str, Any], parameter_count: int
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO genome_history (step, genes, parameter_count)
                VALUES (?, ?, ?)
                """,
                (step, json.dumps(genes), int(parameter_count)),
            )
            return int(cur.lastrowid or 0)

    def genome_history(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT step, genes, parameter_count, created_at "
                "FROM genome_history ORDER BY id ASC"
            )
            cols = [c[0] for c in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row, strict=False))
                try:
                    d["genes"] = json.loads(d["genes"]) if d["genes"] else {}
                except json.JSONDecodeError:
                    d["genes"] = {}
                rows.append(d)
            return rows

    # ------------------------------------------------------------------
    # self-modification history
    # ------------------------------------------------------------------

    def record_self_modification(
        self,
        step: int,
        target: str,
        diff: str,
        outcome: str,
        reason: str = "",
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO self_modifications
                    (step, target, diff, outcome, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (step, target, diff, outcome, reason),
            )
            return int(cur.lastrowid or 0)

    def self_modifications(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT step, target, diff, outcome, reason, created_at "
                "FROM self_modifications ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def path(self) -> Path:
        return self._path


__all__ = ["PersistentMemoryStore", "StoredEpisode", "Insight"]
