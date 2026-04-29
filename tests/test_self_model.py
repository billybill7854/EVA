"""Tests for eva/autonomy/self_model module."""

import numpy as np
import pytest
import torch

from eva.autonomy.self_model import SelfModelSystem, SelfStateSnapshot


class TestSelfModelSystem:
    def _make_emotional_state(self) -> np.ndarray:
        """Create a sample 5D emotional state."""
        return np.array([0.0, 0.5, 0.5, 0.5, 0.5])

    def _make_drive_levels(self) -> dict[str, float]:
        """Create sample drive levels."""
        return {
            "hunger": 0.3,
            "curiosity": 0.7,
            "social": 0.5,
        }

    def _make_behavioral_pattern(self) -> torch.Tensor:
        """Create sample behavioral pattern embedding."""
        return torch.randn(32)

    def _make_hidden_state(self) -> torch.Tensor:
        """Create sample hidden state."""
        return torch.randn(64)

    def test_initialization(self):
        """Test that SelfModelSystem initializes correctly."""
        system = SelfModelSystem(
            history_size=100,
            snapshot_interval=5,
        )
        assert system.history_size == 100
        assert system.snapshot_interval == 5
        assert system.get_history_size() == 0
        assert system.identity_consistency_score == 0.5
        assert system.temporal_continuity_score == 0.5

    def test_update_creates_snapshots(self):
        """Test that update creates snapshots at correct intervals."""
        system = SelfModelSystem(snapshot_interval=3)

        # First two updates should not create snapshots
        for i in range(2):
            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )
        assert system.get_history_size() == 0

        # Third update should create first snapshot
        system.update(
            emotional_state=self._make_emotional_state(),
            drive_levels=self._make_drive_levels(),
            behavioral_pattern=self._make_behavioral_pattern(),
            hidden_state=self._make_hidden_state(),
        )
        assert system.get_history_size() == 1

        # Sixth update should create second snapshot
        for i in range(3):
            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )
        assert system.get_history_size() == 2

    def test_consistency_reward_computation(self):
        """Test that consistency reward is computed correctly."""
        system = SelfModelSystem(snapshot_interval=1)

        # Create snapshots with similar emotional states
        base_emotion = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        for i in range(10):
            # Add small noise to maintain similarity
            emotion = base_emotion + np.random.randn(5) * 0.05
            system.update(
                emotional_state=emotion,
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )

        reward = system.compute_consistency_reward()
        # Should have high consistency due to similar states
        assert 0.0 <= reward <= 1.0
        assert reward > 0.1  # Should be reasonably high

    def test_prediction_and_verification(self):
        """Test prediction recording and verification."""
        system = SelfModelSystem(snapshot_interval=1)

        # Make a prediction
        current_emotion = self._make_emotional_state()
        current_drives = self._make_drive_levels()

        pred_id = system.predict_future_state(
            steps_ahead=5,
            current_emotional_state=current_emotion,
            current_drives=current_drives,
        )

        # Advance to prediction time
        for i in range(5):
            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )

        # Verify prediction with similar state (should have high accuracy)
        accuracy = system.verify_prediction(
            pred_id,
            actual_emotional_state=current_emotion + np.random.randn(5) * 0.1,
            actual_drives=current_drives,
        )

        assert 0.0 <= accuracy <= 1.0
        assert accuracy > 0.5  # Should be reasonably accurate

    def test_prediction_accuracy_reward(self):
        """Test that prediction accuracy reward is computed correctly."""
        system = SelfModelSystem(snapshot_interval=1)

        # Make a prediction for current step + 3
        current_emotion = self._make_emotional_state()
        current_drives = self._make_drive_levels()

        system.predict_future_state(
            steps_ahead=3,
            current_emotional_state=current_emotion,
            current_drives=current_drives,
        )

        # Advance 2 steps (no reward yet)
        for i in range(2):
            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )
            reward = system.compute_prediction_accuracy_reward(
                self._make_emotional_state(),
                self._make_drive_levels(),
            )
            assert reward == 0.0

        # Third step should trigger reward
        system.update(
            emotional_state=current_emotion,
            drive_levels=current_drives,
            behavioral_pattern=self._make_behavioral_pattern(),
            hidden_state=self._make_hidden_state(),
        )
        reward = system.compute_prediction_accuracy_reward(
            current_emotion,
            current_drives,
        )
        assert reward > 0.0

    def test_recognition_reward(self):
        """Test that recognition reward identifies past patterns."""
        system = SelfModelSystem(snapshot_interval=1)

        # Create snapshots with a specific hidden state
        reference_hidden = torch.randn(64)

        for i in range(10):
            # Use similar hidden states
            hidden = reference_hidden + torch.randn(64) * 0.1
            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=hidden,
            )

        # Recognition reward should be high for similar state
        reward = system.compute_recognition_reward(
            reference_hidden + torch.randn(64) * 0.1
        )
        assert 0.0 <= reward <= 1.0
        assert reward > 0.0

    def test_recognition_reward_insufficient_history(self):
        """Test that recognition reward returns 0 with insufficient history."""
        system = SelfModelSystem(snapshot_interval=1)

        # Only 2 snapshots (need at least 5)
        for i in range(2):
            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )

        reward = system.compute_recognition_reward(self._make_hidden_state())
        assert reward == 0.0

    def test_total_reward_computation(self):
        """Test that total reward combines all components."""
        system = SelfModelSystem(snapshot_interval=1)

        # Build up history
        for i in range(10):
            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )

        total, breakdown = system.compute_total_reward(
            current_emotional_state=self._make_emotional_state(),
            current_drives=self._make_drive_levels(),
            current_hidden_state=self._make_hidden_state(),
        )

        # Check breakdown contains all components
        assert "consistency" in breakdown
        assert "prediction_accuracy" in breakdown
        assert "recognition" in breakdown
        assert "total" in breakdown
        assert "identity_consistency_score" in breakdown
        assert "temporal_continuity_score" in breakdown

        # Total should be sum of components
        expected_total = (
            breakdown["consistency"]
            + breakdown["prediction_accuracy"]
            + breakdown["recognition"]
        )
        assert abs(total - expected_total) < 1e-6

    def test_self_query_emotional_match(self):
        """Test self-query with emotional state matching."""
        system = SelfModelSystem(snapshot_interval=1)

        # Create snapshots with varying emotional states
        target_emotion = np.array([0.8, 0.7, 0.6, 0.5, 0.4])

        for i in range(10):
            if i == 5:
                # Insert target emotional state
                emotion = target_emotion.copy()
            else:
                emotion = self._make_emotional_state()

            system.update(
                emotional_state=emotion,
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )

        # Query for similar emotional state
        result = system.self_query(
            query_type="emotional_match",
            context={"emotional_state": target_emotion},
        )

        assert result is not None
        assert isinstance(result, SelfStateSnapshot)
        # Should find the snapshot with target emotion
        assert np.allclose(result.emotional_state, target_emotion, atol=0.1)

    def test_self_query_drive_match(self):
        """Test self-query with drive level matching."""
        system = SelfModelSystem(snapshot_interval=1)

        # Create snapshots with varying drive levels
        target_drives = {"hunger": 0.9, "curiosity": 0.2, "social": 0.5}

        for i in range(10):
            if i == 7:
                drives = target_drives.copy()
            else:
                drives = self._make_drive_levels()

            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=drives,
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )

        # Query for similar drive levels
        result = system.self_query(
            query_type="drive_match",
            context={"drive_levels": target_drives},
        )

        assert result is not None
        assert isinstance(result, SelfStateSnapshot)

    def test_self_query_recent(self):
        """Test self-query for most recent snapshot."""
        system = SelfModelSystem(snapshot_interval=1)

        for i in range(5):
            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )

        result = system.self_query(query_type="recent", context={})

        assert result is not None
        assert result.timestamp == 5  # Most recent

    def test_self_query_empty_history(self):
        """Test self-query returns None with empty history."""
        system = SelfModelSystem()

        result = system.self_query(
            query_type="emotional_match",
            context={"emotional_state": self._make_emotional_state()},
        )

        assert result is None

    def test_clear_history(self):
        """Test that clear_history resets all state."""
        system = SelfModelSystem(snapshot_interval=1)

        # Build up history
        for i in range(10):
            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )

        # Make a prediction
        system.predict_future_state(
            steps_ahead=5,
            current_emotional_state=self._make_emotional_state(),
            current_drives=self._make_drive_levels(),
        )

        assert system.get_history_size() > 0

        # Clear
        system.clear_history()

        assert system.get_history_size() == 0
        assert system.identity_consistency_score == 0.5
        assert system.temporal_continuity_score == 0.5
        assert system._step_count == 0
        assert len(system._pending_predictions) == 0

    def test_history_size_limit(self):
        """Test that history respects max size limit."""
        system = SelfModelSystem(history_size=10, snapshot_interval=1)

        # Create more snapshots than max size
        for i in range(20):
            system.update(
                emotional_state=self._make_emotional_state(),
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )

        # Should not exceed max size
        assert system.get_history_size() <= 10

    def test_identity_metrics_update(self):
        """Test that identity metrics are updated correctly."""
        system = SelfModelSystem(snapshot_interval=1)

        # Create snapshots with consistent emotional states
        base_emotion = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        for i in range(10):
            emotion = base_emotion + np.random.randn(5) * 0.05
            system.update(
                emotional_state=emotion,
                drive_levels=self._make_drive_levels(),
                behavioral_pattern=self._make_behavioral_pattern(),
                hidden_state=self._make_hidden_state(),
            )

        # Identity consistency should be high
        assert system.identity_consistency_score > 0.5

        # Temporal continuity should be reasonable
        assert 0.0 <= system.temporal_continuity_score <= 1.0
