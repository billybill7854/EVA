"""Self-Model System — tracks internal state history for identity continuity.

Maintains snapshots of EVA's internal states over time, enabling:
- Self-consistency rewards for maintaining coherent identity
- Prediction accuracy rewards for anticipating own future states
- Self-recognition rewards for identifying past patterns
- Self-query capabilities for introspection
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SelfStateSnapshot:
    """A snapshot of EVA's internal state at a point in time.

    Attributes:
        timestamp: Step number when snapshot was taken.
        emotional_state: 5D affective state vector.
        drive_levels: Dictionary of homeostatic drive values.
        behavioral_pattern: Recent action sequence hash or embedding.
        hidden_state: Neural network hidden state.
        identity_markers: Key identity-related observations.
    """

    timestamp: int
    emotional_state: np.ndarray
    drive_levels: dict[str, float]
    behavioral_pattern: torch.Tensor
    hidden_state: torch.Tensor
    identity_markers: dict[str, Any]


class SelfModelSystem:
    """Tracks internal state history and computes self-model rewards.

    Maintains a history of state snapshots and provides:
    - Consistency rewards for maintaining stable identity
    - Prediction accuracy rewards for anticipating future states
    - Recognition rewards for identifying past patterns
    - Self-query for introspection

    Args:
        history_size: Maximum number of snapshots to retain.
        snapshot_interval: How many steps between snapshots.
        consistency_weight: Weight for consistency reward.
        prediction_weight: Weight for prediction accuracy reward.
        recognition_weight: Weight for recognition reward.
    """

    def __init__(
        self,
        history_size: int = 1000,
        snapshot_interval: int = 10,
        consistency_weight: float = 0.3,
        prediction_weight: float = 0.4,
        recognition_weight: float = 0.3,
    ) -> None:
        self.history_size = history_size
        self.snapshot_interval = snapshot_interval
        self.consistency_weight = consistency_weight
        self.prediction_weight = prediction_weight
        self.recognition_weight = recognition_weight

        # State history as circular buffer
        self._history: deque[SelfStateSnapshot] = deque(
            maxlen=history_size
        )

        # Prediction tracking
        self._pending_predictions: dict[int, dict[str, Any]] = {}

        # Identity metrics
        self.identity_consistency_score: float = 0.5
        self.temporal_continuity_score: float = 0.5

        # Step counter
        self._step_count: int = 0

    def update(
        self,
        emotional_state: np.ndarray,
        drive_levels: dict[str, float],
        behavioral_pattern: torch.Tensor,
        hidden_state: torch.Tensor,
        identity_markers: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update the self-model with current state.

        Takes a snapshot if interval has elapsed.

        Args:
            emotional_state: Current 5D affective state.
            drive_levels: Current homeostatic drive values.
            behavioral_pattern: Recent action sequence embedding.
            hidden_state: Current neural hidden state.
            identity_markers: Optional identity-related observations.
        """
        self._step_count += 1

        # Take snapshot at intervals
        if self._step_count % self.snapshot_interval == 0:
            snapshot = SelfStateSnapshot(
                timestamp=self._step_count,
                emotional_state=emotional_state.copy(),
                drive_levels=drive_levels.copy(),
                behavioral_pattern=behavioral_pattern.clone().detach(),
                hidden_state=hidden_state.clone().detach(),
                identity_markers=identity_markers or {},
            )
            self._history.append(snapshot)

            # Update identity metrics
            self._update_identity_metrics()

    def _update_identity_metrics(self) -> None:
        """Update identity consistency and temporal continuity scores."""
        if len(self._history) < 2:
            return

        # Consistency: how similar are recent emotional states?
        recent_emotions = [
            snap.emotional_state for snap in list(self._history)[-10:]
        ]
        if len(recent_emotions) >= 2:
            emotion_matrix = np.stack(recent_emotions)
            mean_emotion = emotion_matrix.mean(axis=0)
            deviations = np.linalg.norm(
                emotion_matrix - mean_emotion, axis=1
            )
            avg_deviation = deviations.mean()
            # Lower deviation = higher consistency
            self.identity_consistency_score = max(
                0.0, 1.0 - avg_deviation / 2.0
            )

        # Temporal continuity: how smooth are state transitions?
        if len(self._history) >= 2:
            recent_snaps = list(self._history)[-10:]
            continuity_scores = []
            for i in range(len(recent_snaps) - 1):
                curr = recent_snaps[i]
                next_snap = recent_snaps[i + 1]

                # Emotional continuity
                emotion_diff = np.linalg.norm(
                    curr.emotional_state - next_snap.emotional_state
                )
                emotion_continuity = max(0.0, 1.0 - emotion_diff)

                # Drive continuity
                drive_diffs = [
                    abs(curr.drive_levels.get(k, 0.5)
                        - next_snap.drive_levels.get(k, 0.5))
                    for k in set(curr.drive_levels.keys())
                    | set(next_snap.drive_levels.keys())
                ]
                avg_drive_diff = (
                    np.mean(drive_diffs) if drive_diffs else 0.0
                )
                drive_continuity = max(0.0, 1.0 - avg_drive_diff)

                continuity_scores.append(
                    (emotion_continuity + drive_continuity) / 2.0
                )

            self.temporal_continuity_score = (
                np.mean(continuity_scores) if continuity_scores else 0.5
            )

    def compute_consistency_reward(self) -> float:
        """Compute reward for maintaining consistent identity.

        Returns higher reward when recent states show coherent patterns.

        Returns:
            Consistency reward value [0, 1].
        """
        if len(self._history) < 2:
            return 0.0

        # Reward based on identity consistency score
        return self.identity_consistency_score * self.consistency_weight

    def predict_future_state(
        self,
        steps_ahead: int,
        current_emotional_state: np.ndarray,
        current_drives: dict[str, float],
    ) -> int:
        """Record a prediction about future internal state.

        Args:
            steps_ahead: How many steps in the future to predict.
            current_emotional_state: Current emotional state.
            current_drives: Current drive levels.

        Returns:
            Prediction ID for later verification.
        """
        prediction_id = self._step_count + steps_ahead
        self._pending_predictions[prediction_id] = {
            "predicted_emotion": current_emotional_state.copy(),
            "predicted_drives": current_drives.copy(),
            "made_at_step": self._step_count,
        }
        return prediction_id

    def verify_prediction(
        self,
        prediction_id: int,
        actual_emotional_state: np.ndarray,
        actual_drives: dict[str, float],
    ) -> float:
        """Verify a previous prediction and return accuracy.

        Args:
            prediction_id: ID of the prediction to verify.
            actual_emotional_state: Actual emotional state observed.
            actual_drives: Actual drive levels observed.

        Returns:
            Prediction accuracy [0, 1].
        """
        if prediction_id not in self._pending_predictions:
            return 0.0

        pred = self._pending_predictions.pop(prediction_id)

        # Emotional prediction accuracy
        emotion_error = np.linalg.norm(
            pred["predicted_emotion"] - actual_emotional_state
        )
        emotion_accuracy = max(0.0, 1.0 - emotion_error / 2.0)

        # Drive prediction accuracy
        drive_errors = [
            abs(pred["predicted_drives"].get(k, 0.5)
                - actual_drives.get(k, 0.5))
            for k in set(pred["predicted_drives"].keys())
            | set(actual_drives.keys())
        ]
        avg_drive_error = np.mean(drive_errors) if drive_errors else 0.0
        drive_accuracy = max(0.0, 1.0 - avg_drive_error)

        # Combined accuracy
        return (emotion_accuracy + drive_accuracy) / 2.0

    def compute_prediction_accuracy_reward(
        self,
        actual_emotional_state: np.ndarray,
        actual_drives: dict[str, float],
    ) -> float:
        """Compute reward for accurate self-prediction.

        Checks if any pending predictions match current step.

        Args:
            actual_emotional_state: Current emotional state.
            actual_drives: Current drive levels.

        Returns:
            Prediction accuracy reward [0, 1].
        """
        if self._step_count not in self._pending_predictions:
            return 0.0

        accuracy = self.verify_prediction(
            self._step_count, actual_emotional_state, actual_drives
        )
        return accuracy * self.prediction_weight

    def compute_recognition_reward(
        self, current_hidden_state: torch.Tensor
    ) -> float:
        """Compute reward for recognizing past patterns.

        Returns higher reward when current state matches historical patterns.

        Args:
            current_hidden_state: Current neural hidden state.

        Returns:
            Recognition reward [0, 1].
        """
        if len(self._history) < 5:
            return 0.0

        # Find most similar past state
        query = current_hidden_state.float().flatten()
        emb_dim = query.shape[0]

        # Batch similarity computation
        past_states = [
            snap.hidden_state.float().flatten()[:emb_dim]
            for snap in list(self._history)[:-1]  # Exclude most recent
        ]

        if not past_states:
            return 0.0

        past_embeddings = torch.stack(past_states)
        query_vec = query[:emb_dim].unsqueeze(0)

        # Cosine similarity
        similarities = F.cosine_similarity(
            query_vec, past_embeddings, dim=1
        )
        max_similarity = similarities.max().item()

        # Reward for recognizing familiar patterns
        recognition_score = max(0.0, max_similarity)
        return recognition_score * self.recognition_weight

    def compute_total_reward(
        self,
        current_emotional_state: np.ndarray,
        current_drives: dict[str, float],
        current_hidden_state: torch.Tensor,
    ) -> tuple[float, dict[str, float]]:
        """Compute total self-model reward.

        Args:
            current_emotional_state: Current emotional state.
            current_drives: Current drive levels.
            current_hidden_state: Current neural hidden state.

        Returns:
            Tuple of (total_reward, breakdown_dict).
        """
        consistency = self.compute_consistency_reward()
        prediction = self.compute_prediction_accuracy_reward(
            current_emotional_state, current_drives
        )
        recognition = self.compute_recognition_reward(
            current_hidden_state
        )

        total = consistency + prediction + recognition

        breakdown = {
            "consistency": consistency,
            "prediction_accuracy": prediction,
            "recognition": recognition,
            "total": total,
            "identity_consistency_score": self.identity_consistency_score,
            "temporal_continuity_score": self.temporal_continuity_score,
        }

        return total, breakdown

    def self_query(
        self, query_type: str, context: dict[str, Any]
    ) -> Optional[SelfStateSnapshot]:
        """Query past internal states for introspection.

        Supports queries like "What did I do when X happened before?"

        Args:
            query_type: Type of query ("emotional_match", "drive_match", etc.)
            context: Context for the query (e.g., emotional state to match).

        Returns:
            Most relevant past snapshot, or None if no match.
        """
        if not self._history:
            return None

        if query_type == "emotional_match":
            # Find snapshot with most similar emotional state
            target_emotion = context.get("emotional_state")
            if target_emotion is None:
                return None

            best_match = None
            best_similarity = -1.0

            for snap in self._history:
                diff = np.linalg.norm(
                    snap.emotional_state - target_emotion
                )
                similarity = max(0.0, 1.0 - diff / 2.0)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = snap

            return best_match

        elif query_type == "drive_match":
            # Find snapshot with most similar drive levels
            target_drives = context.get("drive_levels")
            if target_drives is None:
                return None

            best_match = None
            best_similarity = -1.0

            for snap in self._history:
                diffs = [
                    abs(snap.drive_levels.get(k, 0.5)
                        - target_drives.get(k, 0.5))
                    for k in set(snap.drive_levels.keys())
                    | set(target_drives.keys())
                ]
                avg_diff = np.mean(diffs) if diffs else 1.0
                similarity = max(0.0, 1.0 - avg_diff)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = snap

            return best_match

        elif query_type == "recent":
            # Return most recent snapshot
            return self._history[-1] if self._history else None

        return None

    def get_history_size(self) -> int:
        """Return current number of snapshots in history."""
        return len(self._history)

    def clear_history(self) -> None:
        """Clear all state history. Used for portage."""
        self._history.clear()
        self._pending_predictions.clear()
        self.identity_consistency_score = 0.5
        self.temporal_continuity_score = 0.5
        self._step_count = 0
