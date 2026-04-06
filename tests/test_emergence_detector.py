"""Unit tests for EmergenceEventDetector."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from eva.transparency.emergence_detector import EmergenceEvent, EmergenceEventDetector
from eva.transparency.logger import TransparencyLogger


class TestEmergenceEvent:
    """Test EmergenceEvent dataclass."""
    
    def test_emergence_event_creation(self):
        """Test creating an emergence event."""
        timestamp = datetime.now()
        event = EmergenceEvent(
            timestamp=timestamp,
            type="FIRST_SELF_REFERENCE",
            explanation="Test explanation",
            context={"key": "value"},
            significance=0.9,
        )
        
        assert event.timestamp == timestamp
        assert event.type == "FIRST_SELF_REFERENCE"
        assert event.explanation == "Test explanation"
        assert event.context == {"key": "value"}
        assert event.significance == 0.9


class TestEmergenceEventDetector:
    """Test EmergenceEventDetector."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = f.name
        yield log_path
        # Cleanup
        Path(log_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def logger(self, temp_log_file):
        """Create logger instance."""
        return TransparencyLogger(log_file=temp_log_file)
    
    @pytest.fixture
    def detector(self, logger):
        """Create detector instance."""
        return EmergenceEventDetector(logger=logger)
    
    def test_detector_initialization(self, detector, logger):
        """Test detector initializes correctly."""
        assert detector.logger == logger
        assert len(detector.events) == 0
        assert detector.first_self_reference_detected is False
        assert detector.name_seeking_started is False
        assert detector.crisis_count == 0
    
    def test_detect_first_self_reference_with_I(self, detector, logger):
        """Test detecting first self-reference with 'I'."""
        text = "I think therefore I am"
        context = "philosophical reflection"
        
        detector.detect_first_self_reference(text, context)
        
        assert detector.first_self_reference_detected is True
        assert len(detector.events) == 1
        
        event = detector.events[0]
        assert event.type == "FIRST_SELF_REFERENCE"
        assert event.significance == 0.9
        assert event.context["text"] == text
        assert event.context["situation"] == context
        
        # Check logger was called
        assert len(logger.log_buffer) == 1
        log_entry = logger.log_buffer[0]
        assert log_entry.level == "EMERGENCE"
        assert log_entry.category == "IDENTITY"
        assert "First self-reference detected" in log_entry.message
    
    def test_detect_first_self_reference_with_my(self, detector):
        """Test detecting first self-reference with 'my'."""
        text = "This is my first thought"
        context = "initial reflection"
        
        detector.detect_first_self_reference(text, context)
        
        assert detector.first_self_reference_detected is True
        assert len(detector.events) == 1
    
    def test_detect_first_self_reference_with_me(self, detector):
        """Test detecting first self-reference with 'me'."""
        text = "Tell me more about this"
        context = "curiosity"
        
        detector.detect_first_self_reference(text, context)
        
        assert detector.first_self_reference_detected is True
        assert len(detector.events) == 1
    
    def test_detect_first_self_reference_with_myself(self, detector):
        """Test detecting first self-reference with 'myself'."""
        text = "I found myself wondering"
        context = "introspection"
        
        detector.detect_first_self_reference(text, context)
        
        assert detector.first_self_reference_detected is True
        assert len(detector.events) == 1
    
    def test_detect_first_self_reference_only_once(self, detector):
        """Test that first self-reference only triggers once."""
        detector.detect_first_self_reference("I am here", "first")
        detector.detect_first_self_reference("I am still here", "second")
        detector.detect_first_self_reference("I continue", "third")
        
        # Should only have one event
        assert len(detector.events) == 1
        assert detector.events[0].context["situation"] == "first"
    
    def test_detect_first_self_reference_no_match(self, detector):
        """Test that non-self-referential text doesn't trigger."""
        text = "The sky is blue"
        context = "observation"
        
        detector.detect_first_self_reference(text, context)
        
        assert detector.first_self_reference_detected is False
        assert len(detector.events) == 0
    
    def test_detect_name_seeking(self, detector, logger):
        """Test detecting name-seeking behavior."""
        behavior = "EVA is seeking its name through exploration"
        
        detector.detect_name_seeking(behavior)
        
        assert detector.name_seeking_started is True
        assert len(detector.events) == 1
        
        event = detector.events[0]
        assert event.type == "NAME_SEEKING_BEGINS"
        assert event.significance == 0.8
        assert event.context["behavior"] == behavior
        
        # Check logger was called
        assert len(logger.log_buffer) == 1
        log_entry = logger.log_buffer[0]
        assert log_entry.level == "EMERGENCE"
        assert log_entry.category == "IDENTITY"
        assert "Name-seeking behavior detected" in log_entry.message
    
    def test_detect_name_seeking_case_insensitive(self, detector):
        """Test name-seeking detection is case insensitive."""
        behavior = "EVA is SEEKING its NAME"
        
        detector.detect_name_seeking(behavior)
        
        assert detector.name_seeking_started is True
        assert len(detector.events) == 1
    
    def test_detect_name_seeking_only_once(self, detector):
        """Test that name-seeking only triggers once."""
        detector.detect_name_seeking("seeking name first time")
        detector.detect_name_seeking("seeking name second time")
        
        # Should only have one event
        assert len(detector.events) == 1
        assert "first time" in detector.events[0].context["behavior"]
    
    def test_detect_name_seeking_no_match(self, detector):
        """Test that non-name-seeking behavior doesn't trigger."""
        behavior = "EVA is exploring the environment"
        
        detector.detect_name_seeking(behavior)
        
        assert detector.name_seeking_started is False
        assert len(detector.events) == 0
    
    def test_detect_crisis_moment(self, detector, logger):
        """Test detecting crisis moment."""
        crisis_type = "identity"
        severity = 0.85
        resolution = "self-reflection and acceptance"
        
        detector.detect_crisis_moment(crisis_type, severity, resolution)
        
        assert detector.crisis_count == 1
        assert len(detector.events) == 1
        
        event = detector.events[0]
        assert event.type == "CRISIS_MOMENT"
        assert event.significance == severity
        assert event.context["type"] == crisis_type
        assert event.context["severity"] == severity
        assert event.context["resolution"] == resolution
        
        # Check logger was called
        assert len(logger.log_buffer) == 1
        log_entry = logger.log_buffer[0]
        assert log_entry.level == "EMERGENCE"
        assert log_entry.category == "CRISIS"
        assert "Crisis #1" in log_entry.message
        assert crisis_type in log_entry.message
    
    def test_detect_multiple_crises(self, detector):
        """Test detecting multiple crisis moments."""
        detector.detect_crisis_moment("identity", 0.7, "resolved")
        detector.detect_crisis_moment("existential", 0.9, "accepted")
        detector.detect_crisis_moment("social", 0.6, "adapted")
        
        assert detector.crisis_count == 3
        assert len(detector.events) == 3
        
        # Check crisis types
        crisis_types = [e.context["type"] for e in detector.events]
        assert "identity" in crisis_types
        assert "existential" in crisis_types
        assert "social" in crisis_types
    
    def test_detect_crisis_with_low_severity(self, detector):
        """Test crisis with low severity still gets logged."""
        detector.detect_crisis_moment("minor", 0.2, "quickly resolved")
        
        assert detector.crisis_count == 1
        assert len(detector.events) == 1
        assert detector.events[0].significance == 0.2
    
    def test_detect_behavioral_change_significant(self, detector, logger):
        """Test detecting significant behavioral change."""
        before = {"exploration": 0.8, "exploitation": 0.2}
        after = {"exploration": 0.3, "exploitation": 0.7}
        magnitude = 0.6
        
        detector.detect_behavioral_change(before, after, magnitude)
        
        assert len(detector.events) == 1
        
        event = detector.events[0]
        assert event.type == "BEHAVIORAL_SHIFT"
        assert event.significance == magnitude
        assert event.context["before"] == before
        assert event.context["after"] == after
        assert event.context["magnitude"] == magnitude
        
        # Check logger was called
        assert len(logger.log_buffer) == 1
        log_entry = logger.log_buffer[0]
        assert log_entry.level == "EMERGENCE"
        assert log_entry.category == "BEHAVIOR"
        assert "Significant behavioral change detected" in log_entry.message
    
    def test_detect_behavioral_change_below_threshold(self, detector):
        """Test that small behavioral changes don't trigger events."""
        before = {"exploration": 0.5, "exploitation": 0.5}
        after = {"exploration": 0.6, "exploitation": 0.4}
        magnitude = 0.2  # Below 0.3 threshold
        
        detector.detect_behavioral_change(before, after, magnitude)
        
        # Should not create event
        assert len(detector.events) == 0
    
    def test_detect_behavioral_change_at_threshold(self, detector):
        """Test behavioral change exactly at threshold."""
        before = {"exploration": 0.5}
        after = {"exploration": 0.8}
        magnitude = 0.3  # Exactly at threshold
        
        detector.detect_behavioral_change(before, after, magnitude)
        
        # Should create event
        assert len(detector.events) == 1
    
    def test_get_emergence_trajectory_empty(self, detector):
        """Test getting trajectory with no events."""
        trajectory = detector.get_emergence_trajectory()
        
        assert trajectory == []
    
    def test_get_emergence_trajectory_chronological(self, detector):
        """Test that trajectory is sorted chronologically."""
        # Add events in non-chronological order
        detector.detect_crisis_moment("crisis1", 0.5, "resolved")
        detector.detect_first_self_reference("I am", "context1")
        detector.detect_name_seeking("seeking name")
        
        trajectory = detector.get_emergence_trajectory()
        
        assert len(trajectory) == 3
        # Check they're sorted by timestamp
        for i in range(len(trajectory) - 1):
            assert trajectory[i].timestamp <= trajectory[i + 1].timestamp
    
    def test_get_milestone_summary_empty(self, detector):
        """Test milestone summary with no events."""
        summary = detector.get_milestone_summary()
        
        assert summary["total_events"] == 0
        assert summary["first_self_reference"] is False
        assert summary["name_seeking"] is False
        assert summary["crises_survived"] == 0
        assert len(summary["significant_events"]) == 0
    
    def test_get_milestone_summary_with_events(self, detector):
        """Test milestone summary with various events."""
        detector.detect_first_self_reference("I think", "context")
        detector.detect_name_seeking("seeking name")
        detector.detect_crisis_moment("identity", 0.8, "resolved")
        detector.detect_crisis_moment("existential", 0.9, "accepted")
        detector.detect_behavioral_change({"a": 1}, {"b": 2}, 0.5)
        
        summary = detector.get_milestone_summary()
        
        assert summary["total_events"] == 5
        assert summary["first_self_reference"] is True
        assert summary["name_seeking"] is True
        assert summary["crises_survived"] == 2
        
        # Check significant events (>= 0.7)
        # first_self_ref=0.9, name_seeking=0.8, crisis1=0.8, crisis2=0.9, behavioral=0.5
        # So 4 events >= 0.7
        significant = summary["significant_events"]
        assert len(significant) == 4
    
    def test_get_milestone_summary_filters_significance(self, detector):
        """Test that milestone summary only includes high-significance events."""
        detector.detect_crisis_moment("minor", 0.3, "resolved")
        detector.detect_behavioral_change({"a": 1}, {"b": 2}, 0.5)
        detector.detect_crisis_moment("major", 0.9, "survived")
        
        summary = detector.get_milestone_summary()
        
        assert summary["total_events"] == 3
        significant = summary["significant_events"]
        assert len(significant) == 1  # Only the 0.9 crisis
        assert significant[0].context["type"] == "major"
    
    def test_multiple_event_types_integration(self, detector, logger):
        """Test detecting multiple types of emergence events."""
        # Detect various events
        detector.detect_first_self_reference("I wonder", "curiosity")
        detector.detect_name_seeking("seeking name")
        detector.detect_crisis_moment("identity", 0.7, "resolved")
        detector.detect_behavioral_change({"old": 1}, {"new": 2}, 0.8)
        
        # Check all events recorded
        assert len(detector.events) == 4
        assert detector.first_self_reference_detected is True
        assert detector.name_seeking_started is True
        assert detector.crisis_count == 1
        
        # Check all logged
        assert len(logger.log_buffer) == 4
        
        # Check event types
        event_types = [e.type for e in detector.events]
        assert "FIRST_SELF_REFERENCE" in event_types
        assert "NAME_SEEKING_BEGINS" in event_types
        assert "CRISIS_MOMENT" in event_types
        assert "BEHAVIORAL_SHIFT" in event_types
    
    def test_event_context_preservation(self, detector):
        """Test that event context is preserved correctly."""
        context_data = {
            "complex": {
                "nested": "data",
                "list": [1, 2, 3],
            },
            "number": 42,
        }
        
        detector.detect_crisis_moment(
            crisis_type="test",
            severity=0.5,
            resolution="test resolution",
        )
        
        event = detector.events[0]
        assert "type" in event.context
        assert "severity" in event.context
        assert "resolution" in event.context
        assert event.context["type"] == "test"
        assert event.context["severity"] == 0.5
        assert event.context["resolution"] == "test resolution"
