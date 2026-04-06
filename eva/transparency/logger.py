"""Transparency logger for comprehensive event logging."""

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


@dataclass
class LogEntry:
    """Single log entry with timestamp and context."""
    
    timestamp: datetime
    level: str
    category: str
    message: str
    context: Dict[str, Any]


class TransparencyLogger:
    """Log all system events, decisions, and state changes.
    
    Provides comprehensive logging of EVA's internal operations including:
    - Tool invocations
    - Environment switches
    - Self-modifications
    - Drive changes
    - Emotional transitions
    - Curriculum phase changes
    
    Logs are written to a persistent file and buffered in memory for quick access.
    """
    
    def __init__(self, log_file: str = "logs/transparency.log"):
        """Initialize transparency logger.
        
        Args:
            log_file: Path to persistent log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_buffer: Deque[LogEntry] = deque(maxlen=10000)
        
        # Set up file handler
        self.logger = logging.getLogger("transparency")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        self.file_handler = logging.FileHandler(str(self.log_file))
        self.file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(category)s - %(message)s')
        )
        self.logger.addHandler(self.file_handler)
        
    def log(
        self,
        level: str,
        category: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Log an event with full context.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, EMERGENCE)
            category: Event category (TOOL, ENVIRONMENT, SELF_MODIFICATION, DRIVE, EMOTION, CURRICULUM, etc.)
            message: Human-readable message
            context: Additional contextual information
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            context=context or {},
        )
        
        # Add to memory buffer
        self.log_buffer.append(entry)
        
        # Write to file
        log_line = f"{entry.timestamp.isoformat()} - {entry.level} - {entry.category} - {entry.message}"
        if entry.context:
            log_line += f" | Context: {json.dumps(entry.context)}"
        
        # Map level to logging level
        log_level = getattr(logging, entry.level, logging.INFO)
        
        # Create log record with category in extra
        record = logging.LogRecord(
            name="transparency",
            level=log_level,
            pathname="",
            lineno=0,
            msg=log_line,
            args=(),
            exc_info=None,
        )
        record.category = entry.category
        
        self.file_handler.emit(record)
        
    def log_tool_invocation(self, tool_name: str, parameters: str, result_status: str):
        """Log tool invocation.
        
        Args:
            tool_name: Name of the tool invoked
            parameters: Tool parameters as string
            result_status: Result status (success, error, timeout, etc.)
        """
        self.log(
            level="INFO",
            category="TOOL",
            message=f"Invoked {tool_name}",
            context={"parameters": parameters, "status": result_status},
        )
        
    def log_environment_switch(self, from_env: str, to_env: str, reasoning: str):
        """Log environment switch.
        
        Args:
            from_env: Previous environment name
            to_env: New environment name
            reasoning: Reason for the switch
        """
        self.log(
            level="INFO",
            category="ENVIRONMENT",
            message=f"Switched from {from_env} to {to_env}",
            context={"reasoning": reasoning},
        )
        
    def log_self_modification(
        self,
        mod_type: str,
        parameters: Dict[str, Any],
        approval_status: str,
    ):
        """Log self-modification attempt.
        
        Args:
            mod_type: Type of modification (hyperparameter, architecture, curriculum)
            parameters: Modification parameters
            approval_status: Approval status (pending, approved, rejected, applied)
        """
        self.log(
            level="WARNING",
            category="SELF_MODIFICATION",
            message=f"Self-modification: {mod_type}",
            context={"parameters": parameters, "approval": approval_status},
        )
        
    def log_drive_change(self, drive_name: str, old_value: float, new_value: float):
        """Log drive state change.
        
        Args:
            drive_name: Name of the drive
            old_value: Previous drive value
            new_value: New drive value
        """
        self.log(
            level="INFO",
            category="DRIVE",
            message=f"{drive_name} changed",
            context={
                "old": old_value,
                "new": new_value,
                "delta": new_value - old_value,
            },
        )
        
    def log_emotional_transition(self, old_affect: Dict, new_affect: Dict):
        """Log emotional state transition.
        
        Args:
            old_affect: Previous affective state
            new_affect: New affective state
        """
        self.log(
            level="INFO",
            category="EMOTION",
            message="Emotional state changed",
            context={"old": old_affect, "new": new_affect},
        )
        
    def log_curriculum_phase(self, old_phase: str, new_phase: str):
        """Log curriculum phase transition.
        
        Args:
            old_phase: Previous curriculum phase
            new_phase: New curriculum phase
        """
        self.log(
            level="INFO",
            category="CURRICULUM",
            message=f"Phase transition: {old_phase} -> {new_phase}",
        )
        
    def get_logs(
        self,
        level: Optional[str] = None,
        category: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[LogEntry]:
        """Retrieve logs with filtering.
        
        Args:
            level: Filter by log level
            category: Filter by category
            since: Filter by timestamp (only logs after this time)
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries matching the filters
        """
        logs = list(self.log_buffer)
        
        # Apply filters
        if level:
            logs = [l for l in logs if l.level == level]
        if category:
            logs = [l for l in logs if l.category == category]
        if since:
            logs = [l for l in logs if l.timestamp >= since]
            
        # Return most recent entries up to limit
        return logs[-limit:]
