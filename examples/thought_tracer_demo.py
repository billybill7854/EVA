"""Demo of ThoughtProcessTracer usage.

This example demonstrates how to use the ThoughtProcessTracer to capture
and analyze EVA's internal reasoning processes.
"""

import torch
from eva.transparency import ThoughtProcessTracer


def main():
    """Demonstrate ThoughtProcessTracer functionality."""
    
    # Initialize tracer
    tracer = ThoughtProcessTracer(buffer_size=1000, pca_dimensions=8)
    print("ThoughtProcessTracer initialized")
    print(f"Buffer size: {tracer.buffer_size}")
    print(f"PCA dimensions: {tracer.pca_dimensions}\n")
    
    # 1. Trace predictions
    print("=== Tracing Predictions ===")
    logits = torch.randn(512)  # Simulated model output
    tracer.trace_prediction(
        input_context="What is artificial intelligence?",
        logits=logits,
        top_k=5,
    )
    
    recent_predictions = tracer.get_recent_predictions(limit=1)
    pred = recent_predictions[0]
    print(f"Context: {pred.input_context}")
    print(f"Confidence: {pred.confidence:.3f}")
    print(f"Entropy: {pred.entropy:.3f}")
    print(f"Top 3 predictions: {pred.top_predictions[:3]}\n")
    
    # 2. Trace attention patterns
    print("=== Tracing Attention ===")
    seq_len = 20
    attention_weights = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
    tracer.trace_attention(
        attention_weights=attention_weights,
        layer=3,
        head=5,
        top_k=5,
    )
    
    recent_attention = tracer.get_recent_attention(limit=1)
    attn = recent_attention[0]
    print(f"Layer: {attn.layer}, Head: {attn.head}")
    print(f"Top focus tokens: {attn.focus_tokens}\n")
    
    # 3. Trace hidden states
    print("=== Tracing Hidden States ===")
    hidden_state = torch.randn(768)  # d_model = 768
    tracer.trace_hidden_state(
        hidden_state=hidden_state,
        layer=5,
    )
    
    recent_hidden = tracer.get_recent_hidden_states(limit=1)
    hidden = recent_hidden[0]
    print(f"Layer: {hidden.layer}")
    print(f"Norm: {hidden.norm:.3f}")
    print(f"Reduced state (first 4 dims): {hidden.reduced_state[:4]}\n")
    
    # 4. Trace decisions
    print("=== Tracing Decisions ===")
    tracer.trace_decision(
        decision_type="environment_switch",
        options=["nursery", "web", "conversation"],
        chosen="web",
        reasoning="Curiosity hunger exceeded threshold, need to explore new information",
        confidence=0.85,
    )
    
    recent_decisions = tracer.get_recent_decisions(limit=1)
    decision = recent_decisions[0]
    print(f"Decision type: {decision.decision_type}")
    print(f"Options: {decision.options}")
    print(f"Chosen: {decision.chosen}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Reasoning: {decision.reasoning}\n")
    
    # 5. Trace tool selection
    print("=== Tracing Tool Selection ===")
    tracer.trace_tool_selection(
        available_tools=["WebSearch", "FileHandler", "CodeExecution"],
        selected_tool="WebSearch",
        parameters={"query": "machine learning basics"},
        reasoning="Need to search for information about machine learning",
    )
    
    recent_tools = tracer.get_recent_tool_selections(limit=1)
    tool = recent_tools[0]
    print(f"Selected tool: {tool.selected_tool}")
    print(f"Available tools: {tool.available_tools}")
    print(f"Parameters: {tool.parameters}")
    print(f"Reasoning: {tool.reasoning}\n")
    
    # 6. Trace curiosity signals
    print("=== Tracing Curiosity Signals ===")
    curiosity_types = ["novelty", "prediction_error", "information_gain", "empowerment"]
    
    for signal_type in curiosity_types:
        tracer.trace_curiosity_signals(
            signal_type=signal_type,
            value=0.5 + (hash(signal_type) % 100) / 200,  # Simulated values
            context=f"Detected {signal_type} signal in current environment",
        )
    
    recent_curiosity = tracer.get_recent_curiosity_signals(limit=4)
    for signal in recent_curiosity:
        print(f"{signal.signal_type}: {signal.value:.3f} - {signal.context}")
    print()
    
    # 7. Get summary
    print("=== Trace Summary ===")
    summary = tracer.get_trace_summary()
    print(f"Total predictions: {summary['prediction_count']}")
    print(f"Total attention traces: {summary['attention_count']}")
    print(f"Total hidden state traces: {summary['hidden_state_count']}")
    print(f"Total decisions: {summary['decision_count']}")
    print(f"Total tool selections: {summary['tool_selection_count']}")
    print(f"Total curiosity signals: {summary['curiosity_signal_count']}")
    print(f"Avg prediction confidence: {summary['avg_prediction_confidence']:.3f}")
    print(f"Avg decision confidence: {summary['avg_decision_confidence']:.3f}")


if __name__ == "__main__":
    main()
