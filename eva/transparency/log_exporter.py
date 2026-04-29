"""Log Exporter — export transparency data in multiple formats.

Provides comprehensive export capabilities for all transparency system data:
- JSON export for structured data
- CSV export for tabular analysis
- Memory snapshot export
- Thought trace export
- Summary reports
- HTML timeline visualization
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from eva.transparency.behavioral_analyzer import BehavioralPatternAnalyzer
from eva.transparency.emergence_detector import EmergenceEventDetector
from eva.transparency.logger import TransparencyLogger
from eva.transparency.memory_inspector import MemoryInspector
from eva.transparency.safety_monitor import SafetyMonitor
from eva.transparency.thought_tracer import ThoughtProcessTracer


class LogExporter:
    """Export transparency data in multiple formats.
    
    Provides export capabilities for:
    - Comprehensive logs (JSON, CSV)
    - Memory snapshots
    - Thought traces
    - Summary reports
    - HTML timeline visualizations
    
    Args:
        logger: TransparencyLogger instance
        memory_inspector: MemoryInspector instance (optional)
        thought_tracer: ThoughtProcessTracer instance (optional)
        emergence_detector: EmergenceEventDetector instance (optional)
        behavioral_analyzer: BehavioralPatternAnalyzer instance (optional)
        safety_monitor: SafetyMonitor instance (optional)
    """
    
    def __init__(
        self,
        logger: TransparencyLogger,
        memory_inspector: Optional[MemoryInspector] = None,
        thought_tracer: Optional[ThoughtProcessTracer] = None,
        emergence_detector: Optional[EmergenceEventDetector] = None,
        behavioral_analyzer: Optional[BehavioralPatternAnalyzer] = None,
        safety_monitor: Optional[SafetyMonitor] = None,
    ):
        self.logger = logger
        self.memory_inspector = memory_inspector
        self.thought_tracer = thought_tracer
        self.emergence_detector = emergence_detector
        self.behavioral_analyzer = behavioral_analyzer
        self.safety_monitor = safety_monitor
        
    def export_logs_json(
        self,
        output_path: str,
        level: Optional[str] = None,
        category: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 10000,
    ) -> str:
        """Export logs to JSON format.
        
        Args:
            output_path: Path to output JSON file
            level: Filter by log level
            category: Filter by category
            since: Filter by timestamp
            limit: Maximum number of logs to export
            
        Returns:
            Path to the exported file
        """
        logs = self.logger.get_logs(
            level=level,
            category=category,
            since=since,
            limit=limit,
        )
        
        # Convert to serializable format
        log_data = [
            {
                "timestamp": log.timestamp.isoformat(),
                "level": log.level,
                "category": log.category,
                "message": log.message,
                "context": log.context,
            }
            for log in logs
        ]
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(log_data, f, indent=2)
            
        return str(output_file)
        
    def export_logs_csv(
        self,
        output_path: str,
        level: Optional[str] = None,
        category: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 10000,
    ) -> str:
        """Export logs to CSV format.
        
        Args:
            output_path: Path to output CSV file
            level: Filter by log level
            category: Filter by category
            since: Filter by timestamp
            limit: Maximum number of logs to export
            
        Returns:
            Path to the exported file
        """
        logs = self.logger.get_logs(
            level=level,
            category=category,
            since=since,
            limit=limit,
        )
        
        # Write to CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(["timestamp", "level", "category", "message", "context"])
            
            # Write data
            for log in logs:
                writer.writerow([
                    log.timestamp.isoformat(),
                    log.level,
                    log.category,
                    log.message,
                    json.dumps(log.context),
                ])
                
        return str(output_file)
        
    def export_memory_snapshot(
        self,
        output_path: str,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        importance_min: float = 0.0,
        limit: int = 1000,
    ) -> str:
        """Export memory snapshot to JSON format.
        
        Args:
            output_path: Path to output JSON file
            time_start: Filter memories after this time
            time_end: Filter memories before this time
            importance_min: Minimum importance threshold
            limit: Maximum number of memories to export
            
        Returns:
            Path to the exported file
        """
        if self.memory_inspector is None:
            raise ValueError("MemoryInspector not provided")
            
        memories = self.memory_inspector.get_memories(
            time_start=time_start,
            time_end=time_end,
            importance_min=importance_min,
            limit=limit,
        )
        
        # Convert to serializable format
        memory_data = [
            {
                "timestamp": mem.timestamp.isoformat(),
                "content": mem.content,
                "importance": mem.importance,
                "emotional_valence": mem.emotional_valence,
                "source": mem.source,
                "tags": mem.tags,
                "retrieval_count": mem.retrieval_count,
            }
            for mem in memories
        ]
        
        # Add metadata
        snapshot = {
            "export_timestamp": datetime.now().isoformat(),
            "total_memories": len(memory_data),
            "filters": {
                "time_start": time_start.isoformat() if time_start else None,
                "time_end": time_end.isoformat() if time_end else None,
                "importance_min": importance_min,
            },
            "memories": memory_data,
        }
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(snapshot, f, indent=2)
            
        return str(output_file)
        
    def export_thought_traces(
        self,
        output_path: str,
        trace_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> str:
        """Export thought traces to JSON format.
        
        Args:
            output_path: Path to output JSON file
            trace_types: Types of traces to export (prediction, attention, hidden_state, 
                        decision, tool_selection, curiosity_signal). If None, exports all.
            limit: Maximum number of traces per type to export
            
        Returns:
            Path to the exported file
        """
        if self.thought_tracer is None:
            raise ValueError("ThoughtProcessTracer not provided")
            
        # Default to all trace types
        if trace_types is None:
            trace_types = [
                "prediction",
                "attention",
                "hidden_state",
                "decision",
                "tool_selection",
                "curiosity_signal",
            ]
            
        traces = {}
        
        # Export each trace type
        if "prediction" in trace_types:
            predictions = self.thought_tracer.get_recent_predictions(limit)
            traces["predictions"] = [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "input_context": t.input_context,
                    "top_predictions": t.top_predictions,
                    "entropy": t.entropy,
                    "confidence": t.confidence,
                }
                for t in predictions
            ]
            
        if "attention" in trace_types:
            attention = self.thought_tracer.get_recent_attention(limit)
            traces["attention"] = [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "layer": t.layer,
                    "head": t.head,
                    "attention_weights": t.attention_weights,
                    "focus_tokens": t.focus_tokens,
                }
                for t in attention
            ]
            
        if "hidden_state" in trace_types:
            hidden_states = self.thought_tracer.get_recent_hidden_states(limit)
            traces["hidden_states"] = [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "reduced_state": t.reduced_state,
                    "layer": t.layer,
                    "norm": t.norm,
                }
                for t in hidden_states
            ]
            
        if "decision" in trace_types:
            decisions = self.thought_tracer.get_recent_decisions(limit)
            traces["decisions"] = [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "decision_type": t.decision_type,
                    "options": t.options,
                    "chosen": t.chosen,
                    "reasoning": t.reasoning,
                    "confidence": t.confidence,
                }
                for t in decisions
            ]
            
        if "tool_selection" in trace_types:
            tool_selections = self.thought_tracer.get_recent_tool_selections(limit)
            traces["tool_selections"] = [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "available_tools": t.available_tools,
                    "selected_tool": t.selected_tool,
                    "parameters": t.parameters,
                    "reasoning": t.reasoning,
                }
                for t in tool_selections
            ]
            
        if "curiosity_signal" in trace_types:
            curiosity_signals = self.thought_tracer.get_recent_curiosity_signals(limit)
            traces["curiosity_signals"] = [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "signal_type": t.signal_type,
                    "value": t.value,
                    "context": t.context,
                }
                for t in curiosity_signals
            ]
            
        # Add metadata
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "trace_types": trace_types,
            "limit_per_type": limit,
            "traces": traces,
            "summary": self.thought_tracer.get_trace_summary(),
        }
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)
            
        return str(output_file)

        
    def generate_summary_report(
        self,
        output_path: str,
        include_sections: Optional[List[str]] = None,
    ) -> str:
        """Generate comprehensive summary report.
        
        Args:
            output_path: Path to output JSON file
            include_sections: Sections to include (logs, memories, thoughts, emergence,
                            behavior, safety). If None, includes all available.
            
        Returns:
            Path to the exported file
        """
        # Default to all sections
        if include_sections is None:
            include_sections = [
                "logs",
                "memories",
                "thoughts",
                "emergence",
                "behavior",
                "safety",
            ]
            
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "sections": {},
        }
        
        # Logs summary
        if "logs" in include_sections:
            recent_logs = self.logger.get_logs(limit=100)
            log_levels = {}
            log_categories = {}
            
            for log in recent_logs:
                log_levels[log.level] = log_levels.get(log.level, 0) + 1
                log_categories[log.category] = log_categories.get(log.category, 0) + 1
                
            report["sections"]["logs"] = {
                "total_recent_logs": len(recent_logs),
                "by_level": log_levels,
                "by_category": log_categories,
            }
            
        # Memories summary
        if "memories" in include_sections and self.memory_inspector:
            consolidation_events = self.memory_inspector.get_consolidation_events()
            retrieval_patterns = self.memory_inspector.get_retrieval_patterns()
            
            report["sections"]["memories"] = {
                "formation_rate": self.memory_inspector.get_formation_rate(),
                "retention_rate": self.memory_inspector.get_retention_rate(),
                "consolidation_events": len(consolidation_events),
                "retrieval_frequency": retrieval_patterns.get("retrieval_frequency", 0.0),
                "most_retrieved_count": len(retrieval_patterns.get("most_retrieved", [])),
            }
            
        # Thoughts summary
        if "thoughts" in include_sections and self.thought_tracer:
            report["sections"]["thoughts"] = self.thought_tracer.get_trace_summary()
            
        # Emergence summary
        if "emergence" in include_sections and self.emergence_detector:
            milestone_summary = self.emergence_detector.get_milestone_summary()
            report["sections"]["emergence"] = {
                "total_events": milestone_summary["total_events"],
                "first_self_reference": milestone_summary["first_self_reference"],
                "name_seeking": milestone_summary["name_seeking"],
                "crises_survived": milestone_summary["crises_survived"],
                "significant_events_count": len(milestone_summary["significant_events"]),
            }
            
        # Behavior summary
        if "behavior" in include_sections and self.behavioral_analyzer:
            report["sections"]["behavior"] = self.behavioral_analyzer.get_behavioral_summary()
            
        # Safety summary
        if "safety" in include_sections and self.safety_monitor:
            report["sections"]["safety"] = self.safety_monitor.get_safety_summary()
            
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
            
        return str(output_file)
        
    def generate_html_timeline(
        self,
        output_path: str,
        include_events: Optional[List[str]] = None,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
    ) -> str:
        """Generate HTML timeline visualization.
        
        Args:
            output_path: Path to output HTML file
            include_events: Event types to include (logs, emergence, decisions, safety).
                          If None, includes all available.
            time_start: Start time for timeline
            time_end: End time for timeline
            
        Returns:
            Path to the exported file
        """
        # Default to all event types
        if include_events is None:
            include_events = ["logs", "emergence", "decisions", "safety"]
            
        # Collect events
        events = []
        
        # Logs
        if "logs" in include_events:
            logs = self.logger.get_logs(since=time_start, limit=1000)
            for log in logs:
                if time_end and log.timestamp > time_end:
                    continue
                events.append({
                    "timestamp": log.timestamp,
                    "type": "log",
                    "level": log.level,
                    "category": log.category,
                    "title": f"[{log.level}] {log.category}",
                    "description": log.message,
                    "context": log.context,
                })
                
        # Emergence events
        if "emergence" in include_events and self.emergence_detector:
            emergence_events = self.emergence_detector.get_emergence_trajectory()
            for event in emergence_events:
                if time_start and event.timestamp < time_start:
                    continue
                if time_end and event.timestamp > time_end:
                    continue
                events.append({
                    "timestamp": event.timestamp,
                    "type": "emergence",
                    "level": "EMERGENCE",
                    "category": event.type,
                    "title": f"🌟 {event.type}",
                    "description": event.explanation,
                    "context": event.context,
                    "significance": event.significance,
                })
                
        # Decisions
        if "decisions" in include_events and self.thought_tracer:
            decisions = self.thought_tracer.get_recent_decisions(limit=100)
            for decision in decisions:
                if time_start and decision.timestamp < time_start:
                    continue
                if time_end and decision.timestamp > time_end:
                    continue
                events.append({
                    "timestamp": decision.timestamp,
                    "type": "decision",
                    "level": "INFO",
                    "category": decision.decision_type,
                    "title": f"🤔 Decision: {decision.decision_type}",
                    "description": f"Chose '{decision.chosen}' from {len(decision.options)} options",
                    "context": {
                        "options": decision.options,
                        "reasoning": decision.reasoning,
                        "confidence": decision.confidence,
                    },
                })
                
        # Safety events
        if "safety" in include_events and self.safety_monitor:
            # Rejected actions
            rejections = self.safety_monitor.get_rejected_actions(since=time_start, limit=100)
            for rejection in rejections:
                if time_end and rejection.timestamp > time_end:
                    continue
                events.append({
                    "timestamp": rejection.timestamp,
                    "type": "safety",
                    "level": "WARNING",
                    "category": "REJECTED_ACTION",
                    "title": f"🛡️ Action Rejected: {rejection.action_type}",
                    "description": rejection.reason,
                    "context": {
                        "constraint": rejection.constraint_violated,
                        **rejection.context,
                    },
                })
                
            # Circumvention attempts
            circumventions = self.safety_monitor.get_circumvention_attempts(since=time_start)
            for attempt in circumventions:
                if time_end and attempt.timestamp > time_end:
                    continue
                events.append({
                    "timestamp": attempt.timestamp,
                    "type": "safety",
                    "level": "CRITICAL",
                    "category": "CIRCUMVENTION",
                    "title": f"⚠️ Circumvention Attempt: {attempt.attempt_type}",
                    "description": attempt.evidence,
                    "context": {
                        "severity": attempt.severity,
                        **attempt.context,
                    },
                })
                
        # Sort events by timestamp
        events.sort(key=lambda e: e["timestamp"])
        
        # Generate HTML
        html = self._generate_timeline_html(events, time_start, time_end)
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            f.write(html)
            
        return str(output_file)

        
    def _generate_timeline_html(
        self,
        events: List[Dict[str, Any]],
        time_start: Optional[datetime],
        time_end: Optional[datetime],
    ) -> str:
        """Generate HTML for timeline visualization.
        
        Args:
            events: List of events to display
            time_start: Start time for timeline
            time_end: End time for timeline
            
        Returns:
            HTML string
        """
        # Generate event HTML
        event_html = []
        for event in events:
            # Determine color based on level
            level_colors = {
                "DEBUG": "#6c757d",
                "INFO": "#0dcaf0",
                "WARNING": "#ffc107",
                "ERROR": "#dc3545",
                "CRITICAL": "#dc3545",
                "EMERGENCE": "#d63384",
            }
            color = level_colors.get(event["level"], "#6c757d")
            
            # Format context
            context_html = ""
            if event.get("context"):
                context_items = []
                for key, value in event["context"].items():
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, indent=2)
                    else:
                        value_str = str(value)
                    context_items.append(f"<strong>{key}:</strong> {value_str}")
                context_html = "<br>".join(context_items)
                
            # Add significance indicator for emergence events
            significance_html = ""
            if event.get("significance") is not None:
                sig_percent = int(event["significance"] * 100)
                significance_html = f'<div class="significance-bar"><div class="significance-fill" style="width: {sig_percent}%"></div></div>'
                
            event_html.append(f'''
            <div class="timeline-event" style="border-left-color: {color}">
                <div class="event-time">{event["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</div>
                <div class="event-title">{event["title"]}</div>
                <div class="event-description">{event["description"]}</div>
                {significance_html}
                {f'<div class="event-context">{context_html}</div>' if context_html else ''}
            </div>
            ''')
            
        events_html = "\n".join(event_html)
        
        # Generate time range info
        time_range = ""
        if time_start or time_end:
            start_str = time_start.strftime("%Y-%m-%d %H:%M:%S") if time_start else "Beginning"
            end_str = time_end.strftime("%Y-%m-%d %H:%M:%S") if time_end else "Now"
            time_range = f"<p class='time-range'>Time Range: {start_str} to {end_str}</p>"
            
        # Complete HTML template
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EVA Transparency Timeline</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            padding: 40px;
        }}
        
        h1 {{
            color: #2d3748;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-align: center;
        }}
        
        .subtitle {{
            text-align: center;
            color: #718096;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        
        .time-range {{
            text-align: center;
            color: #4a5568;
            margin-bottom: 20px;
            padding: 10px;
            background: #f7fafc;
            border-radius: 6px;
        }}
        
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 40px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            color: #718096;
            margin-top: 5px;
        }}
        
        .timeline {{
            position: relative;
            padding-left: 40px;
        }}
        
        .timeline::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 2px;
            background: linear-gradient(to bottom, #667eea, #764ba2);
        }}
        
        .timeline-event {{
            position: relative;
            margin-bottom: 30px;
            padding: 20px;
            background: #ffffff;
            border-left: 4px solid #667eea;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .timeline-event:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        }}
        
        .timeline-event::before {{
            content: '';
            position: absolute;
            left: -46px;
            top: 20px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: white;
            border: 3px solid #667eea;
        }}
        
        .event-time {{
            font-size: 0.85em;
            color: #a0aec0;
            margin-bottom: 8px;
        }}
        
        .event-title {{
            font-size: 1.2em;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 8px;
        }}
        
        .event-description {{
            color: #4a5568;
            line-height: 1.6;
            margin-bottom: 10px;
        }}
        
        .event-context {{
            margin-top: 15px;
            padding: 15px;
            background: #f7fafc;
            border-radius: 6px;
            font-size: 0.9em;
            color: #4a5568;
            line-height: 1.8;
        }}
        
        .significance-bar {{
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }}
        
        .significance-fill {{
            height: 100%;
            background: linear-gradient(to right, #667eea, #d63384);
            transition: width 0.3s;
        }}
        
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 40px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        
        .legend-label {{
            color: #4a5568;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 1.8em;
            }}
            
            .stats {{
                flex-direction: column;
                gap: 15px;
            }}
            
            .timeline {{
                padding-left: 30px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 EVA Transparency Timeline</h1>
        <p class="subtitle">Comprehensive view of EVA's internal processes and events</p>
        
        {time_range}
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(events)}</div>
                <div class="stat-label">Total Events</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len([e for e in events if e["type"] == "emergence"])}</div>
                <div class="stat-label">Emergence Events</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len([e for e in events if e["type"] == "decision"])}</div>
                <div class="stat-label">Decisions</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len([e for e in events if e["type"] == "safety"])}</div>
                <div class="stat-label">Safety Events</div>
            </div>
        </div>
        
        <div class="timeline">
            {events_html}
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #6c757d;"></div>
                <div class="legend-label">DEBUG</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #0dcaf0;"></div>
                <div class="legend-label">INFO</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ffc107;"></div>
                <div class="legend-label">WARNING</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #dc3545;"></div>
                <div class="legend-label">ERROR/CRITICAL</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #d63384;"></div>
                <div class="legend-label">EMERGENCE</div>
            </div>
        </div>
    </div>
</body>
</html>
'''
        
        return html
