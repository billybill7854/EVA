"""CuriosityEngine — combines all intrinsic motivation signals.

Reward = alpha*surprise + beta*info_gain + gamma*novelty + delta*empowerment
       + epsilon*tool_reward

Tool reward contract (exercised in tests/test_tool_learning_integration.py):
* +0.5 on first *successful* tool use.
* +0.2 on subsequent successful uses that introduce a new pattern.
* -0.1 on successful uses once diversity drops below 0.5.
* 0.0 for failures or when no tool tracker is attached.
"""

from __future__ import annotations

from typing import Optional

import torch

from eva.core.baby_brain import BabyBrain
from eva.curiosity.empowerment import EmpowermentModule
from eva.curiosity.information_gain import InformationGainModule
from eva.curiosity.novelty import NoveltyModule
from eva.curiosity.prediction_error import PredictionErrorModule
from eva.tools.usage_tracker import ToolUsageTracker


class CuriosityEngine:
    """Combines all intrinsic motivation signals into one reward."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.2,
        epsilon: float = 1.0,
        tool_tracker: Optional[ToolUsageTracker] = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        self.prediction_error = PredictionErrorModule()
        self.information_gain = InformationGainModule()
        self.novelty = NoveltyModule()
        self.empowerment = EmpowermentModule()
        self.tool_tracker = tool_tracker

    # ------------------------------------------------------------------
    def prepare(
        self, brain: BabyBrain, sample_ratio: float = 1.0
    ) -> None:
        self.information_gain.snapshot_before(
            brain, sample_ratio=sample_ratio
        )

    # ------------------------------------------------------------------
    def _compute_tool_reward(
        self, tool_name: str, tool_success: bool
    ) -> float:
        """Reward shaping for tool usage (see module docstring)."""

        if not tool_success or self.tool_tracker is None:
            return 0.0
        if self.tool_tracker.is_first_success(tool_name):
            return 0.5
        diversity = self.tool_tracker.get_usage_diversity(tool_name)
        if self.tool_tracker.has_new_pattern(tool_name):
            return 0.2
        if diversity < 0.5:
            return -0.1
        return 0.0

    # ------------------------------------------------------------------
    def compute_reward(
        self,
        predicted: torch.Tensor,
        actual: int,
        brain: BabyBrain,
        hidden_state: torch.Tensor,
        recent_outcomes: list[torch.Tensor],
        sample_ratio: float = 1.0,
        tool_name: Optional[str] = None,
        tool_success: Optional[bool] = None,
    ) -> tuple[float, dict[str, float]]:
        pred_error = self.prediction_error.compute(predicted, actual)
        relative_surprise = self.prediction_error.get_relative_surprise(
            pred_error
        )
        info_gain = self.information_gain.compute(
            brain, sample_ratio=sample_ratio
        )
        state_hash = self.novelty.hash_state(hidden_state)
        novelty = self.novelty.compute(state_hash)
        empowerment = self.empowerment.compute(recent_outcomes)

        tool_reward = 0.0
        if tool_name is not None and tool_success is not None:
            tool_reward = self._compute_tool_reward(tool_name, tool_success)

        total = (
            self.alpha * relative_surprise
            + self.beta * info_gain
            + self.gamma * novelty
            + self.delta * empowerment
            + self.epsilon * tool_reward
        )

        breakdown = {
            "prediction_error": pred_error,
            "relative_surprise": relative_surprise,
            "info_gain": info_gain,
            "novelty": novelty,
            "empowerment": empowerment,
            "tool_reward": tool_reward,
            "total": total,
        }
        return total, breakdown
