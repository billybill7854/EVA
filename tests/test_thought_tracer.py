"""Unit tests for ThoughtProcessTracer."""

import torch
import pytest
from datetime import datetime, timedelta

from eva.transparency.thought_tracer import (
    ThoughtProcessTracer,
    PredictionTrace,
    AttentionTrace,
    HiddenStateTrace,
    DecisionTrace,
    ToolSelectionTrace,
    CuriositySignalTrace,
)


class TestThoughtProcessTracer:
    """Test suite for ThoughtProcessTracer."""
    
    def test_initialization(self):
        """Test tracer initialization."""
        tracer = ThoughtProcessTracer(buffer_size=500, pca_dimensions=4)
        
        assert tracer.buffer_size == 500
        assert tracer.pca_dimensions == 4
        assert len(tracer.prediction_traces) == 0
        assert len(tracer.attention_traces) == 0
        assert len(tracer.hidden_state_traces) == 0
        assert len(tracer.decision_traces) == 0
        assert len(tracer.tool_selection_traces) == 0
        assert len(tracer.curiosity_signal_traces) == 0
        
    def test_trace_prediction(self):
        """Test prediction tracing."""
        tracer = ThoughtProcessTracer()
        
        # Create sample logits
        logits = torch.randn(512)  # vocab_size = 512
        
        tracer.trace_prediction(
            input_context="Hello world",
            logits=logits,
            top_k=5,
        )
        
        assert len(tracer.prediction_traces) == 1
        trace = tracer.prediction_traces[0]
        
        assert isinstance(trace, PredictionTrace)
        assert trace.input_context == "Hello world"
        assert len(trace.top_predictions) == 5
        assert 0.0 <= trace.confidence <= 1.0
        assert trace.entropy > 0.0
        
        # Check that probabilities sum to approximately 1
        total_prob = sum(prob for _, prob in trace.top_predictions)
        assert total_prob <= 1.0
        
    def test_trace_prediction_with_batch(self):
        """Test prediction tracing with batch dimension."""
        tracer = ThoughtProcessTracer()
        
        # Create sample logits with batch dimension
        logits = torch.randn(2, 512)  # batch_size=2, vocab_size=512
        
        tracer.trace_prediction(
            input_context="Batch input",
            logits=logits,
            top_k=3,
        )
        
        assert len(tracer.prediction_traces) == 1
        trace = tracer.prediction_traces[0]
        assert len(trace.top_predictions) == 3
        
    def test_trace_attention(self):
        """Test attention tracing."""
        tracer = ThoughtProcessTracer()
        
        # Create sample attention weights
        seq_len = 20
        attention_weights = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
        
        tracer.trace_attention(
            attention_weights=attention_weights,
            layer=3,
            head=5,
            top_k=10,
        )
        
        assert len(tracer.attention_traces) == 1
        trace = tracer.attention_traces[0]
        
        assert isinstance(trace, AttentionTrace)
        assert trace.layer == 3
        assert trace.head == 5
        assert len(trace.focus_tokens) == 10
        assert len(trace.attention_weights) <= 50  # Truncated
        
    def test_trace_attention_with_batch(self):
        """Test attention tracing with batch dimension."""
        tracer = ThoughtProcessTracer()
        
        # Create sample attention weights with batch dimension
        attention_weights = torch.softmax(torch.randn(2, 20, 20), dim=-1)
        
        tracer.trace_attention(
            attention_weights=attention_weights,
            layer=1,
            head=2,
            top_k=5,
        )
        
        assert len(tracer.attention_traces) == 1
        trace = tracer.attention_traces[0]
        assert len(trace.focus_tokens) == 5
        
    def test_trace_hidden_state(self):
        """Test hidden state tracing with PCA reduction."""
        tracer = ThoughtProcessTracer(pca_dimensions=8)
        
        # Create sample hidden state
        d_model = 768
        hidden_state = torch.randn(d_model)
        
        tracer.trace_hidden_state(
            hidden_state=hidden_state,
            layer=5,
        )
        
        assert len(tracer.hidden_state_traces) == 1
        trace = tracer.hidden_state_traces[0]
        
        assert isinstance(trace, HiddenStateTrace)
        assert trace.layer == 5
        assert len(trace.reduced_state) == 8  # PCA dimensions
        assert trace.norm > 0.0
        assert tracer._pca_initialized
        
    def test_trace_hidden_state_with_batch(self):
        """Test hidden state tracing with batch and sequence dimensions."""
        tracer = ThoughtProcessTracer(pca_dimensions=4)
        
        # Create sample hidden state with batch and sequence dimensions
        hidden_state = torch.randn(2, 10, 768)  # batch=2, seq_len=10, d_model=768
        
        tracer.trace_hidden_state(
            hidden_state=hidden_state,
            layer=2,
        )
        
        assert len(tracer.hidden_state_traces) == 1
        trace = tracer.hidden_state_traces[0]
        assert len(trace.reduced_state) == 4
        
    def test_pca_consistency(self):
        """Test that PCA reduction is consistent across multiple traces."""
        tracer = ThoughtProcessTracer(pca_dimensions=4)
        
        # Trace multiple hidden states
        for i in range(5):
            hidden_state = torch.randn(768)
            tracer.trace_hidden_state(hidden_state, layer=i)
            
        assert len(tracer.hidden_state_traces) == 5
        
        # All traces should have the same reduced dimensionality
        for trace in tracer.hidden_state_traces:
            assert len(trace.reduced_state) == 4
            
    def test_trace_decision(self):
        """Test decision tracing."""
        tracer = ThoughtProcessTracer()
        
        tracer.trace_decision(
            decision_type="environment_switch",
            options=["nursery", "web", "conversation"],
            chosen="web",
            reasoning="Curiosity hunger is high, need to explore",
            confidence=0.85,
        )
        
        assert len(tracer.decision_traces) == 1
        trace = tracer.decision_traces[0]
        
        assert isinstance(trace, DecisionTrace)
        assert trace.decision_type == "environment_switch"
        assert trace.chosen == "web"
        assert len(trace.options) == 3
        assert trace.confidence == 0.85
        assert "Curiosity" in trace.reasoning
        
    def test_trace_tool_selection(self):
        """Test tool selection tracing."""
        tracer = ThoughtProcessTracer()
        
        tracer.trace_tool_selection(
            available_tools=["WebSearch", "FileHandler", "CodeExecution"],
            selected_tool="WebSearch",
            parameters={"query": "artificial intelligence"},
            reasoning="Need to search for information about AI",
        )
        
        assert len(tracer.tool_selection_traces) == 1
        trace = tracer.tool_selection_traces[0]
        
        assert isinstance(trace, ToolSelectionTrace)
        assert trace.selected_tool == "WebSearch"
        assert len(trace.available_tools) == 3
        assert trace.parameters["query"] == "artificial intelligence"
        assert "search" in trace.reasoning.lower()
        
    def test_trace_curiosity_signals(self):
        """Test curiosity signal tracing."""
        tracer = ThoughtProcessTracer()
        
        tracer.trace_curiosity_signals(
            signal_type="novelty",
            value=0.75,
            context="Encountered new environment",
        )
        
        assert len(tracer.curiosity_signal_traces) == 1
        trace = tracer.curiosity_signal_traces[0]
        
        assert isinstance(trace, CuriositySignalTrace)
        assert trace.signal_type == "novelty"
        assert trace.value == 0.75
        assert "new environment" in trace.context.lower()
        
    def test_buffer_limit(self):
        """Test that buffer respects maximum size."""
        tracer = ThoughtProcessTracer(buffer_size=10)
        
        # Add more traces than buffer size
        for i in range(20):
            logits = torch.randn(512)
            tracer.trace_prediction(f"Context {i}", logits)
            
        # Should only keep last 10
        assert len(tracer.prediction_traces) == 10
        
        # Check that we kept the most recent ones
        assert tracer.prediction_traces[-1].input_context == "Context 19"
        assert tracer.prediction_traces[0].input_context == "Context 10"
        
    def test_get_recent_predictions(self):
        """Test retrieving recent prediction traces."""
        tracer = ThoughtProcessTracer()
        
        # Add multiple predictions
        for i in range(15):
            logits = torch.randn(512)
            tracer.trace_prediction(f"Context {i}", logits)
            
        # Get recent predictions
        recent = tracer.get_recent_predictions(limit=5)
        
        assert len(recent) == 5
        assert recent[-1].input_context == "Context 14"
        assert recent[0].input_context == "Context 10"
        
    def test_get_recent_attention(self):
        """Test retrieving recent attention traces."""
        tracer = ThoughtProcessTracer()
        
        # Add multiple attention traces
        for i in range(8):
            attention = torch.softmax(torch.randn(10, 10), dim=-1)
            tracer.trace_attention(attention, layer=i, head=0)
            
        recent = tracer.get_recent_attention(limit=3)
        
        assert len(recent) == 3
        assert recent[-1].layer == 7
        assert recent[0].layer == 5
        
    def test_get_recent_hidden_states(self):
        """Test retrieving recent hidden state traces."""
        tracer = ThoughtProcessTracer()
        
        # Add multiple hidden state traces
        for i in range(12):
            hidden = torch.randn(768)
            tracer.trace_hidden_state(hidden, layer=i)
            
        recent = tracer.get_recent_hidden_states(limit=4)
        
        assert len(recent) == 4
        assert recent[-1].layer == 11
        assert recent[0].layer == 8
        
    def test_get_recent_decisions(self):
        """Test retrieving recent decision traces."""
        tracer = ThoughtProcessTracer()
        
        # Add multiple decisions
        for i in range(6):
            tracer.trace_decision(
                decision_type="action",
                options=["A", "B"],
                chosen="A",
                reasoning=f"Reason {i}",
                confidence=0.5,
            )
            
        recent = tracer.get_recent_decisions(limit=2)
        
        assert len(recent) == 2
        assert "Reason 5" in recent[-1].reasoning
        
    def test_get_recent_tool_selections(self):
        """Test retrieving recent tool selection traces."""
        tracer = ThoughtProcessTracer()
        
        # Add multiple tool selections
        for i in range(7):
            tracer.trace_tool_selection(
                available_tools=["Tool1", "Tool2"],
                selected_tool="Tool1",
                parameters={},
                reasoning=f"Selection {i}",
            )
            
        recent = tracer.get_recent_tool_selections(limit=3)
        
        assert len(recent) == 3
        assert "Selection 6" in recent[-1].reasoning
        
    def test_get_recent_curiosity_signals(self):
        """Test retrieving recent curiosity signal traces."""
        tracer = ThoughtProcessTracer()
        
        # Add multiple curiosity signals
        for i in range(9):
            tracer.trace_curiosity_signals(
                signal_type="novelty",
                value=float(i),
                context=f"Context {i}",
            )
            
        recent = tracer.get_recent_curiosity_signals(limit=4)
        
        assert len(recent) == 4
        assert recent[-1].value == 8.0
        
    def test_get_trace_summary(self):
        """Test getting trace summary statistics."""
        tracer = ThoughtProcessTracer()
        
        # Add various traces
        logits = torch.randn(512)
        tracer.trace_prediction("test", logits)
        tracer.trace_prediction("test2", logits)
        
        attention = torch.softmax(torch.randn(10, 10), dim=-1)
        tracer.trace_attention(attention, layer=0, head=0)
        
        hidden = torch.randn(768)
        tracer.trace_hidden_state(hidden, layer=0)
        
        tracer.trace_decision("test", ["A", "B"], "A", "reason", 0.9)
        tracer.trace_decision("test2", ["C", "D"], "C", "reason2", 0.7)
        
        tracer.trace_tool_selection(["T1"], "T1", {}, "reason")
        tracer.trace_curiosity_signals("novelty", 0.5, "context")
        
        summary = tracer.get_trace_summary()
        
        assert summary["prediction_count"] == 2
        assert summary["attention_count"] == 1
        assert summary["hidden_state_count"] == 1
        assert summary["decision_count"] == 2
        assert summary["tool_selection_count"] == 1
        assert summary["curiosity_signal_count"] == 1
        assert 0.0 <= summary["avg_prediction_confidence"] <= 1.0
        assert summary["avg_decision_confidence"] == 0.8  # (0.9 + 0.7) / 2
        
    def test_empty_summary(self):
        """Test summary with no traces."""
        tracer = ThoughtProcessTracer()
        
        summary = tracer.get_trace_summary()
        
        assert summary["prediction_count"] == 0
        assert summary["avg_prediction_confidence"] == 0.0
        assert summary["avg_decision_confidence"] == 0.0
        
    def test_truncation(self):
        """Test that long strings are truncated."""
        tracer = ThoughtProcessTracer()
        
        # Test prediction context truncation
        long_context = "x" * 200
        logits = torch.randn(512)
        tracer.trace_prediction(long_context, logits)
        
        assert len(tracer.prediction_traces[0].input_context) == 100
        
        # Test decision reasoning truncation
        long_reasoning = "y" * 300
        tracer.trace_decision("test", ["A"], "A", long_reasoning, 0.5)
        
        assert len(tracer.decision_traces[0].reasoning) == 200
        
        # Test tool selection reasoning truncation
        tracer.trace_tool_selection(["T1"], "T1", {}, long_reasoning)
        
        assert len(tracer.tool_selection_traces[0].reasoning) == 200
        
        # Test curiosity signal context truncation
        long_context2 = "z" * 250
        tracer.trace_curiosity_signals("novelty", 0.5, long_context2)
        
        assert len(tracer.curiosity_signal_traces[0].context) == 200
        
    def test_multiple_curiosity_signal_types(self):
        """Test tracing different types of curiosity signals."""
        tracer = ThoughtProcessTracer()
        
        signal_types = ["novelty", "prediction_error", "information_gain", "empowerment"]
        
        for signal_type in signal_types:
            tracer.trace_curiosity_signals(
                signal_type=signal_type,
                value=0.5,
                context=f"Testing {signal_type}",
            )
            
        assert len(tracer.curiosity_signal_traces) == 4
        
        # Check all signal types are recorded
        recorded_types = [t.signal_type for t in tracer.curiosity_signal_traces]
        assert set(recorded_types) == set(signal_types)
        
    def test_timestamp_ordering(self):
        """Test that traces maintain chronological order."""
        tracer = ThoughtProcessTracer()
        
        # Add traces with small delays
        import time
        
        for i in range(5):
            logits = torch.randn(512)
            tracer.trace_prediction(f"Context {i}", logits)
            time.sleep(0.01)  # Small delay to ensure different timestamps
            
        # Check timestamps are in order
        timestamps = [t.timestamp for t in tracer.prediction_traces]
        assert timestamps == sorted(timestamps)
