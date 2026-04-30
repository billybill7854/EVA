"""Base Environment — abstract interface for EVA environments.

Environments provide stimuli for EVA to learn from. Each environment
produces observations (token sequences) and accepts actions (token
predictions). The base class defines the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnvironment(ABC):
    """Abstract base class for EVA environments.

    An environment provides a stream of stimuli for EVA to learn from.
    Each step, the environment provides an observation and EVA responds
    with a prediction. The environment then reveals the actual next
    token, enabling learning from prediction errors.
    """

    def __init__(self, name: str = "base") -> None:
        self.name = name
        self._step_count: int = 0

    @abstractmethod
    def reset(self) -> list[int]:
        """Reset the environment and return initial observation.

        Returns:
            Initial token sequence.
        """

    @abstractmethod
    def step(self, action: int) -> tuple[int, dict[str, Any]]:
        """Take one step in the environment.

        Args:
            action: EVA's predicted next token.

        Returns:
            Tuple of (actual_next_token, info_dict).
        """

    @abstractmethod
    def get_current_sequence(self) -> list[int]:
        """Return the current token sequence context.

        Returns:
            List of token IDs representing current context.
        """

    @property
    def step_count(self) -> int:
        """Total steps taken in this environment."""
        return self._step_count

    def get_info(self) -> dict[str, Any]:
        """Return environment information."""
        return {
            "name": self.name,
            "step_count": self._step_count,
        }
