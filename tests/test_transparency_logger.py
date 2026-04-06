"""Unit tests for TransparencyLogger."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from eva.transparency.logger import LogEntry, TransparencyLogger


class TestLogEntry:
    """Test LogEntry dataclass."""
    
    def test_log_entry_creation(self):
        """Test creating a log entry."""
        timestamp = datetime.now()
        entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            category="TEST",
            message="Test message",
            context={"key": "value"},
        )
        
        assert entry.timestamp == timestamp
        assert entry.level == "INFO"
        assert entry.category == "TEST"
        assert entry.message == "Test message"
        assert entry.context == {"key": "value"}


class TestTransparencyLogger:
    """Test TransparencyLogger."""
    
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
    
    def test_logger_initialization(self, temp_log_file):
        """Test logger initializes correctly."""
        logger = TransparencyLogger(log_file=temp_log_file)
        
        assert logger.log_file == Path(temp_log_file)
        assert len(logger.log_buffer) == 0
        assert logger.log_buffer.maxlen == 10000
        assert logger.file_handler is not None
    
    def test_log_basic(self, logger):
        """Test basic logging."""
        logger.log(
            level="INFO",
            category="TEST",
            message="Test message",
            context={"key": "value"},
        )
        
        assert len(logger.log_buffer) == 1
        entry = logger.log_buffer[0]
        assert entry.level == "INFO"
        assert entry.category == "TEST"
        assert entry.message == "Test message"
        assert entry.context == {"key": "value"}
    
    def test_log_without_context(self, logger):
        """Test logging without context."""
        logger.log(
            level="INFO",
            category="TEST",
            message="Test message",
        )
        
        assert len(logger.log_buffer) == 1
        entry = logger.log_buffer[0]
        assert entry.context == {}
    
    def test_log_tool_invocation(self, logger):
        """Test logging tool invocation."""
        logger.log_tool_invocation(
            tool_name="WebSearch",
            parameters="query=test",
            result_status="success",
        )
        
        assert len(logger.log_buffer) == 1
        entry = logger.log_buffer[0]
        assert entry.level == "INFO"
        assert entry.category == "TOOL"
        assert "WebSearch" in entry.message
        assert entry.context["parameters"] == "query=test"
        assert entry.context["status"] == "success"
    
    def test_log_environment_switch(self, logger):
        """Test logging environment switch."""
        logger.log_environment_switch(
            from_env="nursery",
            to_env="web",
            reasoning="curiosity_hunger exceeded threshold",
        )
        
        assert len(logger.log_buffer) == 1
        entry = logger.log_buffer[0]
        assert entry.level == "INFO"
        assert entry.category == "ENVIRONMENT"
        assert "nursery" in entry.message
        assert "web" in entry.message
        assert entry.context["reasoning"] == "curiosity_hunger exceeded threshold"
    
    def test_log_self_modification(self, logger):
        """Test logging self-modification."""
        logger.log_self_modification(
            mod_type="hyperparameter",
            parameters={"learning_rate": 0.001},
            approval_status="approved",
        )
        
        assert len(logger.log_buffer) == 1
        entry = logger.log_buffer[0]
        assert entry.level == "WARNING"
        assert entry.category == "SELF_MODIFICATION"
        assert "hyperparameter" in entry.message
        assert entry.context["parameters"] == {"learning_rate": 0.001}
        assert entry.context["approval"] == "approved"
    
    def test_log_drive_change(self, logger):
        """Test logging drive change."""
        logger.log_drive_change(
            drive_name="curiosity_hunger",
            old_value=0.5,
            new_value=0.8,
        )
        
        assert len(logger.log_buffer) == 1
        entry = logger.log_buffer[0]
        assert entry.level == "INFO"
        assert entry.category == "DRIVE"
        assert "curiosity_hunger" in entry.message
        assert entry.context["old"] == 0.5
        assert entry.context["new"] == 0.8
        assert entry.context["delta"] == pytest.approx(0.3)
    
    def test_log_emotional_transition(self, logger):
        """Test logging emotional transition."""
        old_affect = {"valence": 0.5, "arousal": 0.3}
        new_affect = {"valence": 0.7, "arousal": 0.6}
        
        logger.log_emotional_transition(
            old_affect=old_affect,
            new_affect=new_affect,
        )
        
        assert len(logger.log_buffer) == 1
        entry = logger.log_buffer[0]
        assert entry.level == "INFO"
        assert entry.category == "EMOTION"
        assert entry.context["old"] == old_affect
        assert entry.context["new"] == new_affect
    
    def test_log_curriculum_phase(self, logger):
        """Test logging curriculum phase transition."""
        logger.log_curriculum_phase(
            old_phase="sensorimotor",
            new_phase="preoperational",
        )
        
        assert len(logger.log_buffer) == 1
        entry = logger.log_buffer[0]
        assert entry.level == "INFO"
        assert entry.category == "CURRICULUM"
        assert "sensorimotor" in entry.message
        assert "preoperational" in entry.message
    
    def test_buffer_limit(self, logger):
        """Test that buffer respects maxlen limit."""
        # Add more than 10,000 entries
        for i in range(10500):
            logger.log(
                level="INFO",
                category="TEST",
                message=f"Message {i}",
            )
        
        # Buffer should contain only last 10,000
        assert len(logger.log_buffer) == 10000
        # First entry should be message 500 (0-499 were dropped)
        assert "Message 500" in logger.log_buffer[0].message
        # Last entry should be message 10499
        assert "Message 10499" in logger.log_buffer[-1].message
    
    def test_get_logs_no_filter(self, logger):
        """Test retrieving logs without filters."""
        for i in range(5):
            logger.log(
                level="INFO",
                category="TEST",
                message=f"Message {i}",
            )
        
        logs = logger.get_logs()
        assert len(logs) == 5
    
    def test_get_logs_level_filter(self, logger):
        """Test retrieving logs with level filter."""
        logger.log(level="INFO", category="TEST", message="Info message")
        logger.log(level="WARNING", category="TEST", message="Warning message")
        logger.log(level="ERROR", category="TEST", message="Error message")
        logger.log(level="INFO", category="TEST", message="Another info")
        
        logs = logger.get_logs(level="INFO")
        assert len(logs) == 2
        assert all(log.level == "INFO" for log in logs)
    
    def test_get_logs_category_filter(self, logger):
        """Test retrieving logs with category filter."""
        logger.log(level="INFO", category="TOOL", message="Tool message")
        logger.log(level="INFO", category="ENVIRONMENT", message="Env message")
        logger.log(level="INFO", category="TOOL", message="Another tool")
        logger.log(level="INFO", category="DRIVE", message="Drive message")
        
        logs = logger.get_logs(category="TOOL")
        assert len(logs) == 2
        assert all(log.category == "TOOL" for log in logs)
    
    def test_get_logs_since_filter(self, logger):
        """Test retrieving logs with timestamp filter."""
        now = datetime.now()
        
        # Add logs with different timestamps
        logger.log(level="INFO", category="TEST", message="Old message")
        
        # Manually set timestamp for testing
        logger.log_buffer[-1].timestamp = now - timedelta(hours=2)
        
        logger.log(level="INFO", category="TEST", message="Recent message 1")
        logger.log(level="INFO", category="TEST", message="Recent message 2")
        
        # Get logs since 1 hour ago
        since = now - timedelta(hours=1)
        logs = logger.get_logs(since=since)
        
        assert len(logs) == 2
        assert all(log.timestamp >= since for log in logs)
    
    def test_get_logs_limit(self, logger):
        """Test retrieving logs with limit."""
        for i in range(20):
            logger.log(level="INFO", category="TEST", message=f"Message {i}")
        
        logs = logger.get_logs(limit=10)
        assert len(logs) == 10
        # Should return most recent 10
        assert "Message 10" in logs[0].message
        assert "Message 19" in logs[-1].message
    
    def test_get_logs_combined_filters(self, logger):
        """Test retrieving logs with multiple filters."""
        now = datetime.now()
        
        logger.log(level="INFO", category="TOOL", message="Tool 1")
        logger.log(level="WARNING", category="TOOL", message="Tool warning")
        logger.log(level="INFO", category="ENVIRONMENT", message="Env 1")
        logger.log(level="INFO", category="TOOL", message="Tool 2")
        
        logs = logger.get_logs(level="INFO", category="TOOL", limit=5)
        assert len(logs) == 2
        assert all(log.level == "INFO" and log.category == "TOOL" for log in logs)
    
    def test_log_persists_to_file(self, logger, temp_log_file):
        """Test that logs are written to file."""
        logger.log(
            level="INFO",
            category="TEST",
            message="Test message",
            context={"key": "value"},
        )
        
        # Force flush
        logger.file_handler.flush()
        
        # Read file
        log_content = Path(temp_log_file).read_text()
        assert "INFO" in log_content
        assert "TEST" in log_content
        assert "Test message" in log_content
        assert "key" in log_content
    
    def test_multiple_log_types(self, logger):
        """Test logging multiple event types."""
        logger.log_tool_invocation("WebSearch", "query=test", "success")
        logger.log_environment_switch("nursery", "web", "curiosity")
        logger.log_drive_change("curiosity_hunger", 0.5, 0.8)
        logger.log_emotional_transition({"valence": 0.5}, {"valence": 0.7})
        logger.log_curriculum_phase("sensorimotor", "preoperational")
        
        assert len(logger.log_buffer) == 5
        
        categories = [entry.category for entry in logger.log_buffer]
        assert "TOOL" in categories
        assert "ENVIRONMENT" in categories
        assert "DRIVE" in categories
        assert "EMOTION" in categories
        assert "CURRICULUM" in categories
