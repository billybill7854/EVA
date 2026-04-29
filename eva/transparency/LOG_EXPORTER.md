# LogExporter

The `LogExporter` provides comprehensive export capabilities for all transparency system data in EVA.

## Features

### Export Formats

1. **JSON Logs** - Structured log data with filtering
2. **CSV Logs** - Tabular format for spreadsheet analysis
3. **Memory Snapshots** - Episodic memory exports with filters
4. **Thought Traces** - Internal reasoning and decision traces
5. **Summary Reports** - Comprehensive system summaries
6. **HTML Timeline** - Beautiful visual timeline of events

## Usage

```python
from eva.transparency import (
    TransparencyLogger,
    MemoryInspector,
    ThoughtProcessTracer,
    EmergenceEventDetector,
    BehavioralPatternAnalyzer,
    SafetyMonitor,
    LogExporter,
)

# Initialize systems
logger = TransparencyLogger()
memory_inspector = MemoryInspector(memory_system)
thought_tracer = ThoughtProcessTracer()
# ... other systems

# Create exporter
exporter = LogExporter(
    logger=logger,
    memory_inspector=memory_inspector,
    thought_tracer=thought_tracer,
    emergence_detector=emergence_detector,
    behavioral_analyzer=behavioral_analyzer,
    safety_monitor=safety_monitor,
)

# Export logs to JSON
exporter.export_logs_json("output/logs.json", level="WARNING")

# Export logs to CSV
exporter.export_logs_csv("output/logs.csv", category="SAFETY")

# Export memory snapshot
exporter.export_memory_snapshot(
    "output/memories.json",
    importance_min=0.7,
    limit=100,
)

# Export thought traces
exporter.export_thought_traces(
    "output/thoughts.json",
    trace_types=["decision", "curiosity_signal"],
)

# Generate summary report
exporter.generate_summary_report(
    "output/summary.json",
    include_sections=["logs", "emergence", "safety"],
)

# Generate HTML timeline
exporter.generate_html_timeline(
    "output/timeline.html",
    include_events=["emergence", "safety"],
)
```

## Export Methods

### `export_logs_json(output_path, level=None, category=None, since=None, limit=10000)`

Export logs to JSON format with optional filtering.

**Parameters:**
- `output_path`: Path to output JSON file
- `level`: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, EMERGENCE)
- `category`: Filter by category (TOOL, ENVIRONMENT, SAFETY, etc.)
- `since`: Filter by timestamp (datetime object)
- `limit`: Maximum number of logs to export

**Returns:** Path to exported file

### `export_logs_csv(output_path, level=None, category=None, since=None, limit=10000)`

Export logs to CSV format for spreadsheet analysis.

**Parameters:** Same as `export_logs_json`

**Returns:** Path to exported file

### `export_memory_snapshot(output_path, time_start=None, time_end=None, importance_min=0.0, limit=1000)`

Export episodic memory snapshot to JSON.

**Parameters:**
- `output_path`: Path to output JSON file
- `time_start`: Filter memories after this time
- `time_end`: Filter memories before this time
- `importance_min`: Minimum importance threshold (0.0 to 1.0)
- `limit`: Maximum number of memories to export

**Returns:** Path to exported file

### `export_thought_traces(output_path, trace_types=None, limit=100)`

Export thought process traces to JSON.

**Parameters:**
- `output_path`: Path to output JSON file
- `trace_types`: List of trace types to export:
  - `"prediction"` - Prediction probabilities
  - `"attention"` - Attention patterns
  - `"hidden_state"` - Hidden state evolution
  - `"decision"` - Decision traces
  - `"tool_selection"` - Tool selection reasoning
  - `"curiosity_signal"` - Curiosity signal activations
- `limit`: Maximum traces per type

**Returns:** Path to exported file

### `generate_summary_report(output_path, include_sections=None)`

Generate comprehensive summary report.

**Parameters:**
- `output_path`: Path to output JSON file
- `include_sections`: Sections to include:
  - `"logs"` - Log statistics
  - `"memories"` - Memory statistics
  - `"thoughts"` - Thought trace summary
  - `"emergence"` - Emergence milestones
  - `"behavior"` - Behavioral patterns
  - `"safety"` - Safety monitoring

**Returns:** Path to exported file

### `generate_html_timeline(output_path, include_events=None, time_start=None, time_end=None)`

Generate beautiful HTML timeline visualization.

**Parameters:**
- `output_path`: Path to output HTML file
- `include_events`: Event types to include:
  - `"logs"` - All log events
  - `"emergence"` - Emergence events
  - `"decisions"` - Decision traces
  - `"safety"` - Safety events
- `time_start`: Start time for timeline
- `time_end`: End time for timeline

**Returns:** Path to exported file

## HTML Timeline Features

The HTML timeline provides:

- **Beautiful visualization** with color-coded events
- **Responsive design** that works on mobile and desktop
- **Event statistics** showing counts by type
- **Interactive hover effects** for better readability
- **Significance indicators** for emergence events
- **Context expansion** showing detailed event information
- **Legend** explaining color codes

## Demo

Run the demo to see all export formats in action:

```bash
python examples/log_exporter_demo.py
```

This will create a temporary directory with all export formats and open the HTML timeline in your browser.

## Requirements

The LogExporter requires the following transparency systems:

- **Required:** `TransparencyLogger`
- **Optional:** `MemoryInspector`, `ThoughtProcessTracer`, `EmergenceEventDetector`, `BehavioralPatternAnalyzer`, `SafetyMonitor`

If optional components are not provided, their corresponding export methods will raise `ValueError`.

## File Format Examples

### JSON Log Entry
```json
{
  "timestamp": "2024-01-01T12:00:00.000000",
  "level": "INFO",
  "category": "TOOL",
  "message": "Web search tool invoked",
  "context": {
    "query": "artificial intelligence",
    "results": 10
  }
}
```

### Memory Snapshot Entry
```json
{
  "timestamp": "2024-01-01T12:00:00.000000",
  "content": "Action: 5, Outcome: 6, Surprise: 0.450",
  "importance": 0.75,
  "emotional_valence": 0.45,
  "source": "self",
  "tags": [],
  "retrieval_count": 3
}
```

### Thought Trace Entry
```json
{
  "timestamp": "2024-01-01T12:00:00.000000",
  "decision_type": "environment_switch",
  "options": ["conversation", "exploration", "rest"],
  "chosen": "conversation",
  "reasoning": "High social drive detected",
  "confidence": 0.85
}
```

## Integration

The LogExporter integrates seamlessly with:

- **Web Dashboard** - Export endpoints for download
- **TUI Dashboard** - Export commands
- **Life Loop** - Periodic automatic exports
- **Birth Ceremony** - Initial state export

## Best Practices

1. **Regular exports** - Export logs periodically to prevent data loss
2. **Filter appropriately** - Use filters to reduce export size
3. **HTML timelines** - Generate timelines for debugging and analysis
4. **Summary reports** - Use for high-level system health checks
5. **Memory snapshots** - Export important memories for analysis
6. **Thought traces** - Export for understanding decision-making

## Performance

- **Efficient filtering** - Filters are applied before serialization
- **Streaming writes** - Large exports don't load everything into memory
- **Batch processing** - HTML timeline processes events in batches
- **Directory creation** - Automatically creates output directories

## Error Handling

The LogExporter handles errors gracefully:

- Missing optional components raise `ValueError` with clear messages
- File I/O errors are propagated with context
- Invalid filters are silently ignored (no results)
- Empty data sets produce valid empty exports
