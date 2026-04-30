"""Unit tests for LogExporter."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from eva.memory.episodic import Episode, EpisodicMemory
from eva.transparency import (
    BehavioralPatternAnalyzer,
    EmergenceEventDetector,
    LogExporter,
    MemoryInspector,
    SafetyMonitor,
    ThoughtProcessTracer,
    TransparencyLogger,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def logger():
    """Create TransparencyLogger instance."""
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        log_file = f.name
    return TransparencyLogger(log_file=log_file)


@pytest.fixture
def memory_system():
    """Create EpisodicMemory instance."""
    return EpisodicMemory(max_size=100)


@pytest.fixture
def memory_inspector(memory_system):
    """Create MemoryInspector instance."""
    return MemoryInspector(memory_system)


@pytest.fixture
def thought_tracer():
    """Create ThoughtProcessTracer instance."""
    return ThoughtProcessTracer(buffer_size=100)


@pytest.fixture
def emergence_detector(logger):
    """Create EmergenceEventDetector instance."""
    return EmergenceEventDetector(logger)


@pytest.fixture
def behavioral_analyzer(logger):
    """Create BehavioralPatternAnalyzer instance."""
    return BehavioralPatternAnalyzer(logger)


@pytest.fixture
def safety_monitor(logger):
    """Create SafetyMonitor instance."""
    return SafetyMonitor(logger)


@pytest.fixture
def log_exporter(logger, memory_inspector, thought_tracer, emergence_detector,
                 behavioral_analyzer, safety_monitor):
    """Create LogExporter instance with all components."""
    return LogExporter(
        logger=logger,
        memory_inspector=memory_inspector,
        thought_tracer=thought_tracer,
        emergence_detector=emergence_detector,
        behavioral_analyzer=behavioral_analyzer,
        safety_monitor=safety_monitor,
    )


def test_export_logs_json(log_exporter, logger, temp_dir):
    """Test JSON log export."""
    # Add some logs
    logger.log("INFO", "TEST", "Test message 1", {"key": "value1"})
    logger.log("WARNING", "TEST", "Test message 2", {"key": "value2"})
    logger.log("ERROR", "TEST", "Test message 3", {"key": "value3"})

    # Export to JSON
    output_path = Path(temp_dir) / "logs.json"
    result_path = log_exporter.export_logs_json(str(output_path))

    # Verify file was created
    assert Path(result_path).exists()

    # Load and verify content
    with open(result_path) as f:
        data = json.load(f)

    assert len(data) == 3
    assert data[0]["level"] == "INFO"
    assert data[0]["category"] == "TEST"
    assert data[0]["message"] == "Test message 1"
    assert data[0]["context"]["key"] == "value1"
    assert "timestamp" in data[0]


def test_export_logs_json_with_filters(log_exporter, logger, temp_dir):
    """Test JSON log export with filters."""
    # Add logs with different levels and categories
    logger.log("INFO", "CAT1", "Message 1")
    logger.log("WARNING", "CAT2", "Message 2")
    logger.log("ERROR", "CAT1", "Message 3")
    logger.log("INFO", "CAT2", "Message 4")

    # Export only WARNING level
    output_path = Path(temp_dir) / "logs_warning.json"
    log_exporter.export_logs_json(str(output_path), level="WARNING")

    with open(output_path) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["level"] == "WARNING"

    # Export only CAT1 category
    output_path = Path(temp_dir) / "logs_cat1.json"
    log_exporter.export_logs_json(str(output_path), category="CAT1")

    with open(output_path) as f:
        data = json.load(f)
    assert len(data) == 2
    assert all(log["category"] == "CAT1" for log in data)


def test_export_logs_csv(log_exporter, logger, temp_dir):
    """Test CSV log export."""
    # Add some logs
    logger.log("INFO", "TEST", "Test message 1", {"key": "value1"})
    logger.log("WARNING", "TEST", "Test message 2", {"key": "value2"})

    # Export to CSV
    output_path = Path(temp_dir) / "logs.csv"
    result_path = log_exporter.export_logs_csv(str(output_path))

    # Verify file was created
    assert Path(result_path).exists()

    # Read and verify content
    with open(result_path) as f:
        lines = f.readlines()

    # Check header
    assert "timestamp" in lines[0]
    assert "level" in lines[0]
    assert "category" in lines[0]

    # Check data rows
    assert len(lines) == 3  # Header + 2 data rows
    assert "INFO" in lines[1]
    assert "WARNING" in lines[2]


def test_export_memory_snapshot(log_exporter, memory_system, memory_inspector, temp_dir):
    """Test memory snapshot export."""
    import torch

    # Add some memories
    for i in range(5):
        episode = Episode(
            state_embedding=torch.randn(10),
            action=i,
            outcome=i + 1,
            surprise=0.5,
            timestamp=i,
            emotional_importance=0.5 + i * 0.1,
            source_tag="test",
        )
        memory_system.store(episode)

    # Export memory snapshot
    output_path = Path(temp_dir) / "memories.json"
    result_path = log_exporter.export_memory_snapshot(str(output_path))

    # Verify file was created
    assert Path(result_path).exists()

    # Load and verify content
    with open(result_path) as f:
        data = json.load(f)

    assert "export_timestamp" in data
    assert "total_memories" in data
    assert "memories" in data
    assert data["total_memories"] == 5
    assert len(data["memories"]) == 5

    # Check memory structure
    mem = data["memories"][0]
    assert "timestamp" in mem
    assert "content" in mem
    assert "importance" in mem
    assert "source" in mem


def test_export_memory_snapshot_with_filters(log_exporter, memory_system, memory_inspector, temp_dir):
    """Test memory snapshot export with filters."""
    import torch

    # Add memories with different importance
    for i in range(10):
        episode = Episode(
            state_embedding=torch.randn(10),
            action=i,
            outcome=i + 1,
            surprise=0.5,
            timestamp=i,
            emotional_importance=i * 0.1,  # 0.0 to 0.9
            source_tag="test",
        )
        memory_system.store(episode)

    # Export with importance filter
    output_path = Path(temp_dir) / "important_memories.json"
    log_exporter.export_memory_snapshot(str(output_path), importance_min=0.5)

    with open(output_path) as f:
        data = json.load(f)

    # Should only include memories with importance >= 0.5
    assert data["total_memories"] <= 5
    assert all(mem["importance"] >= 0.5 for mem in data["memories"])


def test_export_thought_traces(log_exporter, thought_tracer, temp_dir):
    """Test thought trace export."""
    import torch

    # Add some traces
    thought_tracer.trace_prediction(
        "test context",
        torch.tensor([0.1, 0.2, 0.7]),
        top_k=2,
    )

    thought_tracer.trace_decision(
        "test_decision",
        ["option1", "option2"],
        "option1",
        "test reasoning",
        0.8,
    )

    thought_tracer.trace_curiosity_signals(
        "novelty",
        0.75,
        "test context",
    )

    # Export all traces
    output_path = Path(temp_dir) / "traces.json"
    result_path = log_exporter.export_thought_traces(str(output_path))

    # Verify file was created
    assert Path(result_path).exists()

    # Load and verify content
    with open(result_path) as f:
        data = json.load(f)

    assert "export_timestamp" in data
    assert "traces" in data
    assert "summary" in data

    # Check traces
    traces = data["traces"]
    assert "predictions" in traces
    assert "decisions" in traces
    assert "curiosity_signals" in traces

    assert len(traces["predictions"]) == 1
    assert len(traces["decisions"]) == 1
    assert len(traces["curiosity_signals"]) == 1


def test_export_thought_traces_selective(log_exporter, thought_tracer, temp_dir):
    """Test selective thought trace export."""
    import torch

    # Add various traces
    thought_tracer.trace_prediction("ctx", torch.tensor([0.5, 0.5]))
    thought_tracer.trace_decision("dec", ["a", "b"], "a", "reason", 0.9)
    thought_tracer.trace_curiosity_signals("novelty", 0.5, "ctx")

    # Export only decisions
    output_path = Path(temp_dir) / "decisions_only.json"
    log_exporter.export_thought_traces(str(output_path), trace_types=["decision"])

    with open(output_path) as f:
        data = json.load(f)

    traces = data["traces"]
    assert "decisions" in traces
    assert "predictions" not in traces
    assert "curiosity_signals" not in traces


def test_generate_summary_report(log_exporter, logger, temp_dir):
    """Test summary report generation."""
    # Add some data
    logger.log("INFO", "TEST", "Test message")
    logger.log("WARNING", "TEST", "Warning message")
    logger.log("ERROR", "SAFETY", "Error message")

    # Generate report
    output_path = Path(temp_dir) / "report.json"
    result_path = log_exporter.generate_summary_report(str(output_path))

    # Verify file was created
    assert Path(result_path).exists()

    # Load and verify content
    with open(result_path) as f:
        data = json.load(f)

    assert "report_timestamp" in data
    assert "sections" in data

    # Check logs section
    assert "logs" in data["sections"]
    logs_section = data["sections"]["logs"]
    assert "total_recent_logs" in logs_section
    assert "by_level" in logs_section
    assert "by_category" in logs_section

    # Verify counts
    assert logs_section["total_recent_logs"] == 3
    assert logs_section["by_level"]["INFO"] == 1
    assert logs_section["by_level"]["WARNING"] == 1
    assert logs_section["by_level"]["ERROR"] == 1


def test_generate_summary_report_selective(log_exporter, logger, temp_dir):
    """Test selective summary report generation."""
    logger.log("INFO", "TEST", "Test")

    # Generate report with only logs section
    output_path = Path(temp_dir) / "logs_only_report.json"
    log_exporter.generate_summary_report(str(output_path), include_sections=["logs"])

    with open(output_path) as f:
        data = json.load(f)

    sections = data["sections"]
    assert "logs" in sections
    # Other sections should not be present
    assert "memories" not in sections
    assert "thoughts" not in sections


def test_generate_html_timeline(log_exporter, logger, temp_dir):
    """Test HTML timeline generation."""
    # Add some events
    logger.log("INFO", "TEST", "Test event 1")
    logger.log("WARNING", "SAFETY", "Safety warning")
    logger.log("EMERGENCE", "IDENTITY", "First self-reference")

    # Generate timeline
    output_path = Path(temp_dir) / "timeline.html"
    result_path = log_exporter.generate_html_timeline(str(output_path))

    # Verify file was created
    assert Path(result_path).exists()

    # Read and verify HTML content
    with open(result_path) as f:
        html = f.read()

    # Check for key HTML elements
    assert "<!DOCTYPE html>" in html
    assert "EVA Transparency Timeline" in html
    assert "timeline-event" in html

    # Check for event content
    assert "Test event 1" in html
    assert "Safety warning" in html
    assert "First self-reference" in html

    # Check for styling
    assert "<style>" in html
    assert "timeline" in html


def test_generate_html_timeline_with_emergence(log_exporter, logger, emergence_detector, temp_dir):
    """Test HTML timeline with emergence events."""
    # Add emergence event
    emergence_detector.detect_first_self_reference("I am EVA", "test context")

    # Generate timeline
    output_path = Path(temp_dir) / "timeline_emergence.html"
    log_exporter.generate_html_timeline(str(output_path), include_events=["emergence"])

    with open(output_path) as f:
        html = f.read()

    # Check for emergence event
    assert "FIRST_SELF_REFERENCE" in html
    assert "🌟" in html  # Emergence emoji


def test_generate_html_timeline_with_time_range(log_exporter, logger, temp_dir):
    """Test HTML timeline with time range filtering."""
    now = datetime.now()

    # Add events at different times
    logger.log("INFO", "TEST", "Old event")

    # Generate timeline with time filter
    time_start = now + timedelta(seconds=1)
    output_path = Path(temp_dir) / "timeline_filtered.html"
    log_exporter.generate_html_timeline(
        str(output_path),
        time_start=time_start,
    )

    with open(output_path) as f:
        html = f.read()

    # Should show time range
    assert "Time Range:" in html


def test_log_exporter_without_optional_components(logger, temp_dir):
    """Test LogExporter works without optional components."""
    # Create exporter with only logger
    exporter = LogExporter(logger=logger)

    # Should be able to export logs
    logger.log("INFO", "TEST", "Test")
    output_path = Path(temp_dir) / "logs.json"
    result = exporter.export_logs_json(str(output_path))
    assert Path(result).exists()

    # Should raise error when trying to export memories without inspector
    with pytest.raises(ValueError, match="MemoryInspector not provided"):
        exporter.export_memory_snapshot(str(Path(temp_dir) / "mem.json"))

    # Should raise error when trying to export thoughts without tracer
    with pytest.raises(ValueError, match="ThoughtProcessTracer not provided"):
        exporter.export_thought_traces(str(Path(temp_dir) / "traces.json"))


def test_export_creates_directories(log_exporter, logger, temp_dir):
    """Test that export creates necessary directories."""
    # Export to nested path that doesn't exist
    nested_path = Path(temp_dir) / "nested" / "dir" / "logs.json"
    result = log_exporter.export_logs_json(str(nested_path))

    # Should create directories and file
    assert Path(result).exists()
    assert Path(result).parent.exists()


def test_html_timeline_event_colors(log_exporter, logger, temp_dir):
    """Test that HTML timeline uses correct colors for different log levels."""
    # Add events with different levels
    logger.log("DEBUG", "TEST", "Debug")
    logger.log("INFO", "TEST", "Info")
    logger.log("WARNING", "TEST", "Warning")
    logger.log("ERROR", "TEST", "Error")
    logger.log("CRITICAL", "TEST", "Critical")
    logger.log("EMERGENCE", "TEST", "Emergence")

    # Generate timeline
    output_path = Path(temp_dir) / "timeline_colors.html"
    log_exporter.generate_html_timeline(str(output_path))

    with open(output_path) as f:
        html = f.read()

    # Check for color codes (these are defined in the HTML generator)
    assert "#6c757d" in html  # DEBUG
    assert "#0dcaf0" in html  # INFO
    assert "#ffc107" in html  # WARNING
    assert "#dc3545" in html  # ERROR/CRITICAL
    assert "#d63384" in html  # EMERGENCE


def test_summary_report_with_all_components(log_exporter, logger, memory_system,
                                           thought_tracer, emergence_detector,
                                           behavioral_analyzer, safety_monitor, temp_dir):
    """Test summary report with all components providing data."""
    import torch

    # Add data to all components
    logger.log("INFO", "TEST", "Test")

    # Add memory
    episode = Episode(
        state_embedding=torch.randn(10),
        action=0,
        outcome=1,
        surprise=0.5,
        timestamp=0,
        emotional_importance=0.8,
        source_tag="test",
    )
    memory_system.store(episode)

    # Add thought trace
    thought_tracer.trace_prediction("ctx", torch.tensor([0.5, 0.5]))

    # Add emergence event
    emergence_detector.detect_first_self_reference("I am", "test")

    # Add behavioral data
    behavioral_analyzer.track_action(0, True, "test")

    # Add safety event
    safety_monitor.log_rejected_action("test_action", "test reason", "test_constraint")

    # Generate comprehensive report
    output_path = Path(temp_dir) / "full_report.json"
    log_exporter.generate_summary_report(str(output_path))

    with open(output_path) as f:
        data = json.load(f)

    sections = data["sections"]

    # All sections should be present
    assert "logs" in sections
    assert "memories" in sections
    assert "thoughts" in sections
    assert "emergence" in sections
    assert "behavior" in sections
    assert "safety" in sections

    # Verify each section has data
    assert sections["logs"]["total_recent_logs"] > 0
    assert sections["emergence"]["total_events"] > 0
    assert sections["safety"]["total_rejected_actions"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
