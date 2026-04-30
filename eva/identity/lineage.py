"""Lineage — generation tracking and family trees."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LineageTracker:
    """Tracks generational lineage and family relationships.

    Args:
        lineage_path: Directory for lineage storage.
    """

    def __init__(self, lineage_path: str = "lineage/") -> None:
        self._path = Path(lineage_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._tree_file = self._path / "tree.json"
        self._tree: dict[str, Any] = self._load_tree()

    def _load_tree(self) -> dict[str, Any]:
        if self._tree_file.exists():
            with open(self._tree_file) as f:
                return json.load(f)
        return {"individuals": {}, "next_id": 1}

    def _save_tree(self) -> None:
        with open(self._tree_file, "w") as f:
            json.dump(self._tree, f, indent=2)

    def register_birth(
        self,
        name: Optional[str] = None,
        parent_id: Optional[str] = None,
        generation: int = 1,
        genome_hash: Optional[str] = None,
    ) -> str:
        lineage_id = f"EVA-{self._tree['next_id']:06d}"
        self._tree["next_id"] += 1
        self._tree["individuals"][lineage_id] = {
            "name": name,
            "parent_id": parent_id,
            "generation": generation,
            "children": [],
            "genome_hash": genome_hash,
        }
        if parent_id and parent_id in self._tree["individuals"]:
            self._tree["individuals"][parent_id]["children"].append(lineage_id)
        self._save_tree()
        logger.info("Birth registered: %s (gen %d, parent=%s)", lineage_id, generation, parent_id)
        return lineage_id

    def get_individual(self, lineage_id: str) -> Optional[dict[str, Any]]:
        return self._tree["individuals"].get(lineage_id)

    def get_ancestors(self, lineage_id: str) -> list[str]:
        ancestors: list[str] = []
        current = lineage_id
        while current:
            individual = self._tree["individuals"].get(current)
            if individual and individual.get("parent_id"):
                ancestors.append(individual["parent_id"])
                current = individual["parent_id"]
            else:
                break
        return ancestors

    def get_generation(self, lineage_id: str) -> int:
        individual = self._tree["individuals"].get(lineage_id)
        if individual:
            return individual.get("generation", 1)
        return 1

    def update_name(self, lineage_id: str, name: str) -> None:
        if lineage_id in self._tree["individuals"]:
            self._tree["individuals"][lineage_id]["name"] = name
            self._save_tree()

    @property
    def population(self) -> int:
        return len(self._tree["individuals"])

    def to_dict(self) -> dict[str, Any]:
        return self._tree.copy()
