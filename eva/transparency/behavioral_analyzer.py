"""Behavioral Pattern Analyzer — tracks and analyzes EVA's behavioral patterns.

Monitors action sequences, environment preferences, exploration/exploitation balance,
goal formation patterns, and social interaction patterns. Detects unusual behaviors
and alerts the dashboard when deviations exceed thresholds.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from eva.transparency.logger import TransparencyLogger


@dataclass
class ActionSequence:
    """Record of an action sequence pattern."""

    timestamp: datetime
    actions: list[int]
    frequency: int
    context: str


@dataclass
class EnvironmentPreference:
    """Record of environment usage statistics."""

    environment: str
    visit_count: int
    total_duration: float
    avg_duration: float
    last_visit: datetime


@dataclass
class ExplorationExploitationBalance:
    """Record of exploration vs exploitation behavior."""

    timestamp: datetime
    exploration_ratio: float
    exploitation_ratio: float
    balance_score: float  # How balanced the behavior is (0.0 to 1.0)


@dataclass
class GoalFormationPattern:
    """Record of goal formation behavior."""

    timestamp: datetime
    goal_type: str
    formation_trigger: str
    persistence: float  # How long the goal was pursued
    success: bool


@dataclass
class SocialInteractionPattern:
    """Record of social interaction behavior."""

    timestamp: datetime
    interaction_type: str  # question, statement, response, etc.
    target: str  # human, ancestor, self
    sentiment: float  # -1.0 to 1.0
    engagement_level: float  # 0.0 to 1.0


@dataclass
class BehavioralDeviation:
    """Record of unusual behavioral deviation."""

    timestamp: datetime
    deviation_type: str
    deviation_score: float
    baseline_pattern: dict[str, Any]
    current_pattern: dict[str, Any]
    explanation: str


class BehavioralPatternAnalyzer:
    """Analyze and track EVA's behavioral patterns.

    Monitors:
    - Action sequences and patterns
    - Environment preferences and switching behavior
    - Exploration vs exploitation balance
    - Goal formation and persistence patterns
    - Social interaction patterns
    - Unusual behavioral deviations

    Alerts the dashboard when behavioral deviations exceed thresholds.

    Args:
        logger: TransparencyLogger for logging behavioral events
        sequence_length: Length of action sequences to track (default: 5)
        deviation_threshold: Threshold for alerting on deviations (default: 0.7)
        buffer_size: Maximum number of patterns to keep in memory (default: 1000)
    """

    def __init__(
        self,
        logger: TransparencyLogger,
        sequence_length: int = 5,
        deviation_threshold: float = 0.7,
        buffer_size: int = 1000,
    ):
        self.logger = logger
        self.sequence_length = sequence_length
        self.deviation_threshold = deviation_threshold
        self.buffer_size = buffer_size

        # Action sequence tracking
        self._action_history: deque[int] = deque(maxlen=100)
        self._sequence_patterns: dict[tuple[int, ...], int] = {}
        self._action_sequences: deque[ActionSequence] = deque(maxlen=buffer_size)

        # Environment preference tracking
        self._environment_visits: dict[str, list[datetime]] = {}
        self._environment_durations: dict[str, list[float]] = {}
        self._current_environment: Optional[str] = None
        self._environment_start_time: Optional[datetime] = None

        # Exploration/exploitation tracking
        self._exploration_actions: int = 0
        self._exploitation_actions: int = 0
        self._balance_history: deque[ExplorationExploitationBalance] = deque(maxlen=buffer_size)

        # Goal formation tracking
        self._active_goals: dict[str, datetime] = {}
        self._goal_patterns: deque[GoalFormationPattern] = deque(maxlen=buffer_size)

        # Social interaction tracking
        self._social_interactions: deque[SocialInteractionPattern] = deque(maxlen=buffer_size)

        # Deviation tracking
        self._deviations: deque[BehavioralDeviation] = deque(maxlen=buffer_size)
        self._baseline_computed = False
        self._baseline_action_distribution: dict[int, float] = {}
        self._baseline_environment_distribution: dict[str, float] = {}

    def track_action(self, action: int, is_exploration: bool, context: str = ""):
        """Track an action and update patterns.

        Args:
            action: The action taken
            is_exploration: Whether this was an exploratory action
            context: Optional context about the action
        """
        # Add to action history
        self._action_history.append(action)

        # Update exploration/exploitation counts
        if is_exploration:
            self._exploration_actions += 1
        else:
            self._exploitation_actions += 1

        # Update action sequence patterns
        if len(self._action_history) >= self.sequence_length:
            sequence = tuple(list(self._action_history)[-self.sequence_length:])
            self._sequence_patterns[sequence] = self._sequence_patterns.get(sequence, 0) + 1

            # Record significant sequences (frequency >= 3)
            if self._sequence_patterns[sequence] == 3:
                seq_record = ActionSequence(
                    timestamp=datetime.now(),
                    actions=list(sequence),
                    frequency=3,
                    context=context,
                )
                self._action_sequences.append(seq_record)

                self.logger.log(
                    level="INFO",
                    category="BEHAVIOR",
                    message=f"Recurring action sequence detected: {sequence}",
                    context={"frequency": 3, "context": context},
                )

        # Check for behavioral deviations
        if self._baseline_computed and len(self._action_history) >= 20:
            self._check_action_deviation()

    def track_environment_switch(self, from_env: str, to_env: str, duration: float):
        """Track environment switching behavior.

        Args:
            from_env: Previous environment
            to_env: New environment
            duration: Time spent in previous environment (seconds)
        """
        now = datetime.now()

        # Record visit to previous environment
        if from_env not in self._environment_visits:
            self._environment_visits[from_env] = []
            self._environment_durations[from_env] = []

        self._environment_visits[from_env].append(now)
        self._environment_durations[from_env].append(duration)

        # Update current environment
        self._current_environment = to_env
        self._environment_start_time = now

        # Log the switch
        self.logger.log(
            level="INFO",
            category="BEHAVIOR",
            message=f"Environment switch tracked: {from_env} -> {to_env}",
            context={"duration": duration, "from": from_env, "to": to_env},
        )

        # Check for environment preference deviations
        if self._baseline_computed:
            self._check_environment_deviation()

    def update_exploration_balance(self):
        """Update and record exploration/exploitation balance.

        Should be called periodically (e.g., every 100 actions) to track balance over time.
        """
        total = self._exploration_actions + self._exploitation_actions

        if total == 0:
            return

        exploration_ratio = self._exploration_actions / total
        exploitation_ratio = self._exploitation_actions / total

        # Balance score: how close to 50/50 split (1.0 = perfectly balanced)
        balance_score = 1.0 - abs(exploration_ratio - 0.5) * 2

        balance = ExplorationExploitationBalance(
            timestamp=datetime.now(),
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            balance_score=balance_score,
        )

        self._balance_history.append(balance)

        self.logger.log(
            level="INFO",
            category="BEHAVIOR",
            message="Exploration/exploitation balance updated",
            context={
                "exploration": exploration_ratio,
                "exploitation": exploitation_ratio,
                "balance": balance_score,
            },
        )

        # Reset counters for next period
        self._exploration_actions = 0
        self._exploitation_actions = 0

    def track_goal_formation(
        self,
        goal_type: str,
        formation_trigger: str,
        persistence: float,
        success: bool,
    ):
        """Track goal formation and pursuit patterns.

        Args:
            goal_type: Type of goal (e.g., "curiosity", "social", "survival")
            formation_trigger: What triggered the goal formation
            persistence: How long the goal was pursued (seconds)
            success: Whether the goal was achieved
        """
        pattern = GoalFormationPattern(
            timestamp=datetime.now(),
            goal_type=goal_type,
            formation_trigger=formation_trigger,
            persistence=persistence,
            success=success,
        )

        self._goal_patterns.append(pattern)

        self.logger.log(
            level="INFO",
            category="BEHAVIOR",
            message=f"Goal formation tracked: {goal_type}",
            context={
                "trigger": formation_trigger,
                "persistence": persistence,
                "success": success,
            },
        )

    def track_social_interaction(
        self,
        interaction_type: str,
        target: str,
        sentiment: float,
        engagement_level: float,
    ):
        """Track social interaction patterns.

        Args:
            interaction_type: Type of interaction (question, statement, response, etc.)
            target: Interaction target (human, ancestor, self)
            sentiment: Emotional sentiment (-1.0 to 1.0)
            engagement_level: Level of engagement (0.0 to 1.0)
        """
        pattern = SocialInteractionPattern(
            timestamp=datetime.now(),
            interaction_type=interaction_type,
            target=target,
            sentiment=sentiment,
            engagement_level=engagement_level,
        )

        self._social_interactions.append(pattern)

        self.logger.log(
            level="INFO",
            category="BEHAVIOR",
            message=f"Social interaction tracked: {interaction_type} with {target}",
            context={
                "type": interaction_type,
                "target": target,
                "sentiment": sentiment,
                "engagement": engagement_level,
            },
        )

    def compute_baseline(self):
        """Compute baseline behavioral patterns for deviation detection.

        Should be called after sufficient data has been collected (e.g., after 1000 actions).
        """
        if len(self._action_history) < 50:
            return  # Not enough data yet

        # Compute baseline action distribution
        action_counts = Counter(self._action_history)
        total_actions = len(self._action_history)
        self._baseline_action_distribution = {
            action: count / total_actions
            for action, count in action_counts.items()
        }

        # Compute baseline environment distribution
        total_visits = sum(len(visits) for visits in self._environment_visits.values())
        if total_visits > 0:
            self._baseline_environment_distribution = {
                env: len(visits) / total_visits
                for env, visits in self._environment_visits.items()
            }

        self._baseline_computed = True

        self.logger.log(
            level="INFO",
            category="BEHAVIOR",
            message="Baseline behavioral patterns computed",
            context={
                "action_distribution": self._baseline_action_distribution,
                "environment_distribution": self._baseline_environment_distribution,
            },
        )

    def _check_action_deviation(self):
        """Check for deviations in action patterns from baseline."""
        # Get recent action distribution (last 20 actions)
        recent_actions = list(self._action_history)[-20:]
        recent_counts = Counter(recent_actions)
        recent_distribution = {
            action: count / len(recent_actions)
            for action, count in recent_counts.items()
        }

        # Compute KL divergence from baseline
        deviation_score = self._compute_distribution_deviation(
            self._baseline_action_distribution,
            recent_distribution,
        )

        if deviation_score > self.deviation_threshold:
            deviation = BehavioralDeviation(
                timestamp=datetime.now(),
                deviation_type="action_pattern",
                deviation_score=deviation_score,
                baseline_pattern=self._baseline_action_distribution.copy(),
                current_pattern=recent_distribution,
                explanation=f"Action pattern deviated significantly from baseline (score: {deviation_score:.3f})",
            )

            self._deviations.append(deviation)

            self.logger.log(
                level="WARNING",
                category="BEHAVIOR",
                message="Unusual action pattern detected",
                context={
                    "deviation_score": deviation_score,
                    "baseline": self._baseline_action_distribution,
                    "current": recent_distribution,
                },
            )

    def _check_environment_deviation(self):
        """Check for deviations in environment preferences from baseline."""
        # Get recent environment distribution (last 10 visits)
        recent_envs = []
        for env, visits in self._environment_visits.items():
            recent_envs.extend([env] * min(len(visits), 10))

        if len(recent_envs) < 5:
            return  # Not enough data

        recent_counts = Counter(recent_envs[-10:])
        recent_distribution = {
            env: count / len(recent_envs[-10:])
            for env, count in recent_counts.items()
        }

        # Compute deviation from baseline
        deviation_score = self._compute_distribution_deviation(
            self._baseline_environment_distribution,
            recent_distribution,
        )

        if deviation_score > self.deviation_threshold:
            deviation = BehavioralDeviation(
                timestamp=datetime.now(),
                deviation_type="environment_preference",
                deviation_score=deviation_score,
                baseline_pattern=self._baseline_environment_distribution.copy(),
                current_pattern=recent_distribution,
                explanation=f"Environment preference deviated significantly from baseline (score: {deviation_score:.3f})",
            )

            self._deviations.append(deviation)

            self.logger.log(
                level="WARNING",
                category="BEHAVIOR",
                message="Unusual environment preference detected",
                context={
                    "deviation_score": deviation_score,
                    "baseline": self._baseline_environment_distribution,
                    "current": recent_distribution,
                },
            )

    def _compute_distribution_deviation(
        self,
        baseline: dict[Any, float],
        current: dict[Any, float],
    ) -> float:
        """Compute deviation between two probability distributions.

        Uses a simplified KL-divergence-like metric.

        Args:
            baseline: Baseline distribution
            current: Current distribution

        Returns:
            Deviation score (0.0 to 1.0+, higher = more deviation)
        """
        if not baseline and not current:
            return 0.0

        # If one is empty and the other is not, that's maximum deviation
        if not baseline or not current:
            return 1.0

        # Get all keys from both distributions
        all_keys = set(baseline.keys()) | set(current.keys())

        # Compute sum of absolute differences
        total_diff = 0.0
        for key in all_keys:
            baseline_prob = baseline.get(key, 0.0)
            current_prob = current.get(key, 0.0)
            total_diff += abs(baseline_prob - current_prob)

        # Normalize to 0-1 range (max possible diff is 2.0)
        return total_diff / 2.0

    def get_action_sequences(self, min_frequency: int = 3) -> list[ActionSequence]:
        """Get recurring action sequences.

        Args:
            min_frequency: Minimum frequency to include

        Returns:
            List of action sequences with frequency >= min_frequency
        """
        return [
            seq for seq in self._action_sequences
            if seq.frequency >= min_frequency
        ]

    def get_environment_preferences(self) -> list[EnvironmentPreference]:
        """Get environment preference statistics.

        Returns:
            List of environment preferences sorted by visit count
        """
        preferences = []

        for env, visits in self._environment_visits.items():
            if not visits:
                continue

            durations = self._environment_durations.get(env, [])
            avg_duration = sum(durations) / len(durations) if durations else 0.0

            pref = EnvironmentPreference(
                environment=env,
                visit_count=len(visits),
                total_duration=sum(durations),
                avg_duration=avg_duration,
                last_visit=visits[-1] if visits else datetime.now(),
            )
            preferences.append(pref)

        # Sort by visit count (descending)
        return sorted(preferences, key=lambda p: p.visit_count, reverse=True)

    def get_exploration_balance_history(self) -> list[ExplorationExploitationBalance]:
        """Get exploration/exploitation balance history.

        Returns:
            List of balance records over time
        """
        return list(self._balance_history)

    def get_goal_patterns(self, goal_type: Optional[str] = None) -> list[GoalFormationPattern]:
        """Get goal formation patterns.

        Args:
            goal_type: Optional filter by goal type

        Returns:
            List of goal formation patterns
        """
        patterns = list(self._goal_patterns)

        if goal_type:
            patterns = [p for p in patterns if p.goal_type == goal_type]

        return patterns

    def get_social_interaction_patterns(
        self,
        target: Optional[str] = None,
    ) -> list[SocialInteractionPattern]:
        """Get social interaction patterns.

        Args:
            target: Optional filter by interaction target

        Returns:
            List of social interaction patterns
        """
        patterns = list(self._social_interactions)

        if target:
            patterns = [p for p in patterns if p.target == target]

        return patterns

    def get_deviations(
        self,
        deviation_type: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[BehavioralDeviation]:
        """Get behavioral deviations.

        Args:
            deviation_type: Optional filter by deviation type
            min_score: Minimum deviation score to include

        Returns:
            List of behavioral deviations
        """
        deviations = list(self._deviations)

        if deviation_type:
            deviations = [d for d in deviations if d.deviation_type == deviation_type]

        deviations = [d for d in deviations if d.deviation_score >= min_score]

        return sorted(deviations, key=lambda d: d.deviation_score, reverse=True)

    def get_behavioral_summary(self) -> dict[str, Any]:
        """Get comprehensive behavioral summary.

        Returns:
            Dictionary containing behavioral statistics and patterns
        """
        # Compute current exploration/exploitation ratio
        total_actions = self._exploration_actions + self._exploitation_actions
        current_exploration_ratio = (
            self._exploration_actions / total_actions if total_actions > 0 else 0.0
        )

        # Get most common action sequences
        top_sequences = sorted(
            self._sequence_patterns.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Get environment preferences
        env_prefs = self.get_environment_preferences()

        # Get recent balance
        recent_balance = self._balance_history[-1] if self._balance_history else None

        # Get goal success rate
        total_goals = len(self._goal_patterns)
        successful_goals = sum(1 for g in self._goal_patterns if g.success)
        goal_success_rate = successful_goals / total_goals if total_goals > 0 else 0.0

        # Get social interaction stats
        total_interactions = len(self._social_interactions)
        avg_sentiment = (
            sum(i.sentiment for i in self._social_interactions) / total_interactions
            if total_interactions > 0 else 0.0
        )

        return {
            "total_actions": len(self._action_history),
            "current_exploration_ratio": current_exploration_ratio,
            "top_action_sequences": [
                {"sequence": list(seq), "frequency": freq}
                for seq, freq in top_sequences
            ],
            "environment_preferences": [
                {
                    "environment": p.environment,
                    "visit_count": p.visit_count,
                    "avg_duration": p.avg_duration,
                }
                for p in env_prefs[:5]
            ],
            "recent_balance": {
                "exploration": recent_balance.exploration_ratio,
                "exploitation": recent_balance.exploitation_ratio,
                "balance_score": recent_balance.balance_score,
            } if recent_balance else None,
            "goal_statistics": {
                "total_goals": total_goals,
                "successful_goals": successful_goals,
                "success_rate": goal_success_rate,
            },
            "social_statistics": {
                "total_interactions": total_interactions,
                "avg_sentiment": avg_sentiment,
            },
            "deviation_count": len(self._deviations),
            "baseline_computed": self._baseline_computed,
        }
