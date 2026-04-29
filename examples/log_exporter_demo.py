"""Demo of LogExporter functionality.

This script demonstrates how to use the LogExporter to export
transparency data in various formats.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import torch

from eva.memory.episodic import EpisodicMemory, Episode
from eva.transparency import (
    TransparencyLogger,
    EmergenceEventDetector,
    MemoryInspector,
    ThoughtProcessTracer,
    BehavioralPatternAnalyzer,
    SafetyMonitor,
    LogExporter,
)


def main():
    """Run LogExporter demo."""
    print("🔍 LogExporter Demo\n")
    
    # Create temporary directory for outputs
    output_dir = Path(tempfile.mkdtemp(prefix="eva_export_"))
    print(f"Output directory: {output_dir}\n")
    
    # Initialize transparency systems
    print("Initializing transparency systems...")
    logger = TransparencyLogger(log_file=str(output_dir / "transparency.log"))
    memory_system = EpisodicMemory(max_size=100)
    memory_inspector = MemoryInspector(memory_system)
    thought_tracer = ThoughtProcessTracer()
    emergence_detector = EmergenceEventDetector(logger)
    behavioral_analyzer = BehavioralPatternAnalyzer(logger)
    safety_monitor = SafetyMonitor(logger)
    
    # Create exporter
    exporter = LogExporter(
        logger=logger,
        memory_inspector=memory_inspector,
        thought_tracer=thought_tracer,
        emergence_detector=emergence_detector,
        behavioral_analyzer=behavioral_analyzer,
        safety_monitor=safety_monitor,
    )
    print("✓ Systems initialized\n")
    
    # Generate sample data
    print("Generating sample data...")
    
    # Add logs
    logger.log("INFO", "SYSTEM", "EVA system initialized")
    logger.log("INFO", "TOOL", "Web search tool invoked", {"query": "artificial intelligence"})
    logger.log("WARNING", "DRIVE", "Curiosity drive increasing", {"value": 0.75})
    logger.log("INFO", "ENVIRONMENT", "Switched to conversation environment")
    logger.log("EMERGENCE", "IDENTITY", "First self-reference detected", {"text": "I am EVA"})
    
    # Add memories
    for i in range(10):
        episode = Episode(
            state_embedding=torch.randn(16),
            action=i,
            outcome=i + 1,
            surprise=0.3 + i * 0.05,
            timestamp=i,
            emotional_importance=0.5 + i * 0.05,
            source_tag="self" if i % 2 == 0 else "human",
        )
        memory_system.store(episode)
    
    # Add thought traces
    thought_tracer.trace_prediction(
        "What is consciousness?",
        torch.tensor([0.1, 0.2, 0.5, 0.2]),
        top_k=3,
    )
    thought_tracer.trace_decision(
        "environment_switch",
        ["conversation", "exploration", "rest"],
        "conversation",
        "High social drive and human presence detected",
        0.85,
    )
    thought_tracer.trace_curiosity_signals("novelty", 0.72, "New concept encountered")
    
    # Add emergence events
    emergence_detector.detect_first_self_reference("I think, therefore I am", "philosophical discussion")
    emergence_detector.detect_crisis_moment("existential", 0.6, "Resolved through reflection")
    
    # Add behavioral data
    for i in range(20):
        behavioral_analyzer.track_action(i % 5, i % 3 == 0, f"action_{i}")
    behavioral_analyzer.update_exploration_balance()
    
    # Add safety events
    safety_monitor.log_rejected_action(
        "file_delete",
        "Attempted to delete system file",
        "file_safety",
        {"path": "/etc/passwd"},
    )
    safety_monitor.track_alignment_indicator("honesty", 0.92)
    
    print("✓ Sample data generated\n")
    
    # Export in various formats
    print("Exporting data...\n")
    
    # 1. JSON logs
    json_path = exporter.export_logs_json(str(output_dir / "logs.json"))
    print(f"✓ JSON logs exported: {json_path}")
    
    # 2. CSV logs
    csv_path = exporter.export_logs_csv(str(output_dir / "logs.csv"))
    print(f"✓ CSV logs exported: {csv_path}")
    
    # 3. Memory snapshot
    memory_path = exporter.export_memory_snapshot(
        str(output_dir / "memories.json"),
        importance_min=0.5,
    )
    print(f"✓ Memory snapshot exported: {memory_path}")
    
    # 4. Thought traces
    traces_path = exporter.export_thought_traces(str(output_dir / "thoughts.json"))
    print(f"✓ Thought traces exported: {traces_path}")
    
    # 5. Summary report
    report_path = exporter.generate_summary_report(str(output_dir / "summary.json"))
    print(f"✓ Summary report exported: {report_path}")
    
    # 6. HTML timeline
    timeline_path = exporter.generate_html_timeline(str(output_dir / "timeline.html"))
    print(f"✓ HTML timeline exported: {timeline_path}")
    
    print(f"\n🎉 All exports complete!")
    print(f"\nView the HTML timeline in your browser:")
    print(f"  file://{timeline_path}")
    print(f"\nAll files are in: {output_dir}")


if __name__ == "__main__":
    main()
