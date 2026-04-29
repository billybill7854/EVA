# ThoughtProcessTracer

The `ThoughtProcessTracer` captures EVA's internal reasoning and decision-making processes, providing visibility into predictions, attention patterns, hidden states, and curiosity signals.

## Features

- **Prediction Tracing**: Captures model predictions with top-k probabilities, entropy, and confidence
- **Attention Tracing**: Records attention patterns from transformer layers (layer and head specific)
- **Hidden State Tracing**: Tracks hidden state evolution with PCA dimensionality reduction
- **Decision Tracing**: Logs decision points with options, chosen action, reasoning, and confidence
- **Tool Selection Tracing**: Records tool selection reasoning with available tools and parameters
- **Curiosity Signal Tracing**: Captures curiosity signal activations (novelty, prediction error, information gain, empowerment)

## Usage

```python
from eva.transparency import ThoughtProcessTracer
import torch

# Initialize tracer
tracer = ThoughtProcessTracer(buffer_size=1000, pca_dimensions=8)

# Trace predictions
logits = model(input_ids)
tracer.trace_prediction(
    input_context="What is AI?",
    logits=logits,
    top_k=5,
)

# Trace attention patterns (for transformer architecture)
attention_weights = model.get_attention_weights()
tracer.trace_attention(
    attention_weights=attention_weights,
    layer=3,
    head=5,
    top_k=10,
)

# Trace hidden states
hidden_state = model.get_hidden_state()
tracer.trace_hidden_state(
    hidden_state=hidden_state,
    layer=5,
)

# Trace decisions
tracer.trace_decision(
    decision_type="environment_switch",
    options=["nursery", "web", "conversation"],
    chosen="web",
    reasoning="Curiosity hunger exceeded threshold",
    confidence=0.85,
)

# Trace tool selection
tracer.trace_tool_selection(
    available_tools=["WebSearch", "FileHandler", "CodeExecution"],
    selected_tool="WebSearch",
    parameters={"query": "machine learning"},
    reasoning="Need to search for information",
)

# Trace curiosity signals
tracer.trace_curiosity_signals(
    signal_type="novelty",
    value=0.75,
    context="Encountered new environment",
)

# Retrieve recent traces
recent_predictions = tracer.get_recent_predictions(limit=10)
recent_decisions = tracer.get_recent_decisions(limit=10)
recent_curiosity = tracer.get_recent_curiosity_signals(limit=10)

# Get summary statistics
summary = tracer.get_trace_summary()
print(f"Total predictions: {summary['prediction_count']}")
print(f"Avg confidence: {summary['avg_prediction_confidence']:.3f}")
```

## Buffer Management

The tracer maintains circular buffers with a configurable maximum size (default: 1,000 entries). When the buffer is full, the oldest traces are automatically removed to make room for new ones.

## PCA Dimensionality Reduction

Hidden states are high-dimensional (typically 768 or larger). The tracer uses PCA-like dimensionality reduction to compress these states into a smaller representation (default: 8 dimensions) for efficient storage and visualization.

The PCA implementation uses random projection with running mean updates, providing a stable dimensionality reduction without requiring a training set.

## Requirements Coverage

This implementation satisfies requirements 16.23-16.28:

- ✅ 16.23: Trace prediction probabilities with top-k predictions and confidence
- ✅ 16.24: Trace attention patterns with layer and head information
- ✅ 16.25: Trace hidden states with PCA dimensionality reduction
- ✅ 16.26: Trace decision points with options, chosen action, and reasoning
- ✅ 16.27: Trace tool selection with available tools and parameters
- ✅ 16.28: Trace curiosity signals with signal type and value

## Testing

Comprehensive unit tests are provided in `tests/test_thought_tracer.py`:

```bash
pytest tests/test_thought_tracer.py -v
```

All 23 tests pass, covering:
- Initialization
- All tracing methods (prediction, attention, hidden state, decision, tool selection, curiosity)
- Batch dimension handling
- PCA consistency
- Buffer limit enforcement
- Retrieval methods
- Summary statistics
- Truncation
- Timestamp ordering

## Example

See `examples/thought_tracer_demo.py` for a complete demonstration of all features.
