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

## Requirements Coverage

This implementation satisfies requirements 16.1-16.7:

- ✅ 16.1: Log tool invocations with tool name, parameters, timestamp, and result status
- ✅ 16.2: Log environment switches with previous/new environment, timestamp, and reasoning
- ✅ 16.3: Log self-modification attempts with type, parameters, approval status, and timestamp
- ✅ 16.4: Log drive state changes with drive name, previous/new value, and timestamp
- ✅ 16.5: Log emotional state transitions with previous/new affect and timestamp
- ✅ 16.6: Log curriculum phase transitions with previous/new phase and timestamp
- ✅ 16.7: Record system events with timestamp and contextual information

## Testing

Comprehensive unit tests are provided in `tests/test_transparency_logger.py`:

```bash
pytest tests/test_transparency_logger.py -v
```

All 19 tests pass, covering:
- Log entry creation
- Logger initialization
- All logging methods
- Buffer limit enforcement
- Filtering by level, category, timestamp
- Limit enforcement
- Combined filters
- File persistence
- Multiple log types
