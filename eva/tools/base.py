"""Abstract tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract base for every EVA tool.

    Concrete subclasses must implement :meth:`execute`, :meth:`get_name`
    and :meth:`is_available`. Tools are intentionally kept narrow —
    each one solves a single, well-described capability gap.
    """

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """Run the tool and return a serialisable result."""

    @abstractmethod
    def get_name(self) -> str:  # pragma: no cover
        """Return the tool's stable name."""

    @abstractmethod
    def is_available(self) -> bool:  # pragma: no cover
        """Return ``True`` if the tool can currently run on this host."""
