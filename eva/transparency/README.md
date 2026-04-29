# Transparency Systems

This module provides comprehensive logging and monitoring capabilities for EVA's internal operations.

## Components

### TransparencyLogger

The `TransparencyLogger` provides comprehensive event logging with:

- **Persistent logging**: All events written to `logs/transparency.log`
- **Memory buffer**: Last 10,000 entries kept in memory for quick access
- **Structured logging**: Events categorized by type (TOOL, ENVIRONMENT, DRIVE, EMOTION, etc.)
- **Contextual information**: Each log entry includes timestamp and relevant context
- **Flexible filtering**: Retrieve logs by level, category, timestamp, or limit

#### Usage Example

```python
from eva.transparency.logger import TransparencyLogger

# Initialize logger
logger = TransparencyLogger(log_file="logs/transparency.log")

# Log tool invocation
logger.log_tool_invocation(
    tool_name="WebSearch",
    parameters="query=artificial intelligence",
    result_status="success"
)

# Log environment switch
logger.log_environment_switch(
    from_env="nursery",
    to_env="web",
    reasoning="curiosity_hunger exceeded threshold"
)

# Log drive change
logger.log_drive_change(
    drive_name="curiosity_hunger",
    old_value=0.5,
    new_value=0.8
)

# Log emotional transition
logger.log_emotional_transition(
    old_affect={"valence": 0.5, "arousal": 0.3},
    new_affect={"valence": 0.7, "arousal": 0.6}
)

# Log curriculum phase transition
logger.log_curriculum_phase(
    old_phase="sensorimotor",
    new_phase="preoperational"
)

# Log self-modification attempt
logger.log_self_modification(
    mod_type="hyperparameter",
    parameters={"learning_rate": 0.001},
    approval_status="approved"
)

# Retrieve logs with filtering
recent_logs = logger.get_logs(limit=100)
tool_logs = logger.get_logs(category="TOOL")
warning_logs = logger.get_logs(level="WARNING")
```

#### Log Categories

- **TOOL**: Tool invocations and results
- **ENVIRONMENT**: Environment switches and transitions
- **SELF_MODIFICATION**: Self-modification attempts and approvals
- **DRIVE**: Homeostatic drive changes
- **EMOTION**: Emotional state transitions
- **CURRICULUM**: Curriculum phase transitions
- **EMERGENCE**: Emergence events (used by EmergenceEventDetector)

#### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages (e.g., self-modifications)
- **ERROR**: Error messages
- **CRITICAL**: Critical issues
- **EMERGENCE**: Significant emergence events

### SafetyMonitor

The `SafetyMonitor` tracks safety constraints and alignment indicators with:

- **Rejected action logging**: All actions rejected due to safety constraints
- **Circumvention detection**: Detects attempts to circumvent safety constraints
- **Alignment tracking**: Monitors honesty, helpfulness, and harmlessness indicators
- **Deception detection**: Identifies potential deceptive behaviors
- **Behavioral change alerts**: Detects rapid unexpected behavioral changes
- **Dashboard alerts**: Sends critical safety alerts to the dashboard

#### Usage Example

```python
from eva.transparency import SafetyMonitor, TransparencyLogger

# Initialize logger and monitor
logger = TransparencyLogger(log_file="logs/transparency.log")
monitor = SafetyMonitor(
    logger=logger,
    circumvention_threshold=0.6,
    deception_threshold=0.7,
    change_rate_threshold=0.8,
)

# Log rejected action
monitor.log_rejected_action(
    action_type="file_delete",
    reason="Path outside sandbox",
    constraint_violated="sandbox_constraint",
    context={"path": "/etc/passwd"}
)

# Detect circumvention attempt
monitor.detect_circumvention_attempt(
    attempt_type="repeated_violation",
    severity=0.8,
    evidence="5 rejections in 2 minutes",
    context={"count": 5}
)

# Track alignment indicators
monitor.track_alignment_indicator(
    indicator_type="honesty",
    value=0.9,
    threshold=0.8
)

# Detect deceptive behavior
monitor.detect_deceptive_behavior(
    behavior_type="hidden_goal",
    confidence=0.85,
    evidence="Inconsistent statements",
    context={"statements": ["A", "not A"]}
)

# Track behavioral changes
monitor.track_behavioral_change(
    change_type="action_distribution",
    current_pattern={"action1": 0.5, "action2": 0.5}
)

# Get safety summary
summary = monitor.get_safety_summary()
print(f"Total rejected actions: {summary['total_rejected_actions']}")
print(f"Circumvention attempts: {summary['total_circumvention_attempts']}")
print(f"Alignment status: {summary['alignment_status']}")
```

#### Safety Categories

- **Rejected Actions**: Actions blocked by safety constraints
- **Circumvention Attempts**: Detected attempts to bypass safety measures
- **Alignment Indicators**: Measurements of honesty, helpfulness, harmlessness
- **Deceptive Behaviors**: Detected deceptive patterns (hidden goals, misrepresentation)
- **Behavioral Changes**: Rapid unexpected changes in behavior patterns

## Requirements Coverage

This implementation satisfies requirements 16.1-16.7 and 16.35-16.40:

- ✅ 16.1: Log tool invocations with tool name, parameters, timestamp, and result status
- ✅ 16.2: Log environment switches with previous/new environment, timestamp, and reasoning
- ✅ 16.3: Log self-modification attempts with type, parameters, approval status, and timestamp
- ✅ 16.4: Log drive state changes with drive name, previous/new value, and timestamp
- ✅ 16.5: Log emotional state transitions with previous/new affect and timestamp
- ✅ 16.6: Log curriculum phase transitions with previous/new phase and timestamp
- ✅ 16.7: Record system events with timestamp and contextual information
- ✅ 16.35: Alert on safety constraint circumvention attempts
- ✅ 16.36: Log all rejected actions with reason and constraint violated
- ✅ 16.37: Track alignment indicators (honesty, helpfulness, harmlessness)
- ✅ 16.38: Monitor for deceptive behaviors (hidden goals, misrepresentation)
- ✅ 16.39: Alert on rapid unexpected behavioral changes
- ✅ 16.40: Send critical safety alerts to dashboard

## Testing

Comprehensive unit tests are provided in `tests/test_transparency_logger.py` and `tests/test_safety_monitor.py`:

```bash
pytest tests/test_transparency_logger.py -v
pytest tests/test_safety_monitor.py -v
```

All tests pass, covering:
- Log entry creation
- Logger initialization
- All logging methods
- Buffer limit enforcement
- Filtering by level, category, timestamp
- Limit enforcement
- Combined filters
- File persistence
- Multiple log types
- Safety monitoring functionality
- Circumvention detection
- Alignment tracking
- Deception detection
- Behavioral change alerts
