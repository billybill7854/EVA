"""Thought Process Tracer — captures EVA's internal reasoning and decision-making.

Provides visibility into predictions, attention patterns, hidden states,
and curiosity signals to understand EVA's thought processes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class PredictionTrace:
    """Record of a prediction with probabilities."""
    
    timestamp: datetime
    input_context: str
    top_predictions: List[tuple[int, float]]  # (token_id, probability)
    entropy: float
    confidence: float


@dataclass
class AttentionTrace:
    """Record of attention patterns."""
    
    timestamp: datetime
    layer: int
    head: int
    attention_weights: List[float]  # Simplified attention pattern
    focus_tokens: List[int]  # Token indices with highest attention


@dataclass
class HiddenStateTrace:
    """Record of hidden state (dimensionality reduced)."""
    
    timestamp: datetime
    reduced_state: List[float]  # PCA-reduced representation
    layer: int
    norm: float


@dataclass
class DecisionTrace:
    """Record of a decision point."""
    
    timestamp: datetime
    decision_type: str
    options: List[str]
    chosen: str
    reasoning: str
    confidence: float


@dataclass
class ToolSelectionTrace:
    """Record of tool selection reasoning."""
    
    timestamp: datetime
    available_tools: List[str]
    selected_tool: str
    parameters: Dict[str, Any]
    reasoning: str


@dataclass
class CuriositySignalTrace:
    """Record of curiosity signals."""
    
    timestamp: datetime
    signal_type: str  # novelty, prediction_error, information_gain, empowerment
    value: float
    context: str


class ThoughtProcessTracer:
    """Capture and trace EVA's internal reasoning processes.
    
    Provides visibility into:
    - Prediction probabilities and confidence
    - Attention patterns (for transformer architecture)
    - Hidden state evolution (PCA-reduced)
    - Decision-making processes
    - Tool selection reasoning
    - Curiosity signal activations
    
    All traces are buffered with a maximum size of 1,000 entries.
    """
    
    def __init__(self, buffer_size: int = 1000, pca_dimensions: int = 8):
        """Initialize thought process tracer.
        
        Args:
            buffer_size: Maximum number of traces to keep in memory
            pca_dimensions: Number of dimensions for PCA reduction of hidden states
        """
        self.buffer_size = buffer_size
        self.pca_dimensions = pca_dimensions
        
        # Trace buffers
        self.prediction_traces: Deque[PredictionTrace] = deque(maxlen=buffer_size)
        self.attention_traces: Deque[AttentionTrace] = deque(maxlen=buffer_size)
        self.hidden_state_traces: Deque[HiddenStateTrace] = deque(maxlen=buffer_size)
        self.decision_traces: Deque[DecisionTrace] = deque(maxlen=buffer_size)
        self.tool_selection_traces: Deque[ToolSelectionTrace] = deque(maxlen=buffer_size)
        self.curiosity_signal_traces: Deque[CuriositySignalTrace] = deque(maxlen=buffer_size)
        
        # PCA state for hidden state reduction
        self._pca_mean: Optional[torch.Tensor] = None
        self._pca_components: Optional[torch.Tensor] = None
        self._pca_initialized = False
        
    def trace_prediction(
        self,
        input_context: str,
        logits: torch.Tensor,
        top_k: int = 5,
    ) -> None:
        """Trace prediction probabilities.
        
        Args:
            input_context: The input context for this prediction
            logits: Model output logits (shape: [vocab_size] or [batch, vocab_size])
            top_k: Number of top predictions to record
        """
        # Handle batch dimension
        if logits.dim() > 1:
            logits = logits[0]  # Take first batch element
            
        # Convert to probabilities
        probs = F.softmax(logits.float(), dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
        top_predictions = [
            (idx.item(), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        # Compute entropy (measure of uncertainty)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        # Confidence is the probability of the top prediction
        confidence = top_probs[0].item()
        
        trace = PredictionTrace(
            timestamp=datetime.now(),
            input_context=input_context[:100],  # Truncate for storage
            top_predictions=top_predictions,
            entropy=entropy,
            confidence=confidence,
        )
        
        self.prediction_traces.append(trace)
        
    def trace_attention(
        self,
        attention_weights: torch.Tensor,
        layer: int,
        head: int,
        top_k: int = 10,
    ) -> None:
        """Trace attention patterns.
        
        Args:
            attention_weights: Attention weights (shape: [seq_len, seq_len] or [batch, seq_len, seq_len])
            layer: Layer index
            head: Attention head index
            top_k: Number of top attention positions to record
        """
        # Handle batch dimension
        if attention_weights.dim() > 2:
            attention_weights = attention_weights[0]  # Take first batch element
            
        # Average attention across query positions to get overall pattern
        avg_attention = attention_weights.mean(dim=0)
        
        # Get top-k attended positions
        top_attn, top_indices = torch.topk(avg_attention, k=min(top_k, len(avg_attention)))
        
        trace = AttentionTrace(
            timestamp=datetime.now(),
            layer=layer,
            head=head,
            attention_weights=avg_attention.tolist()[:50],  # Store first 50 for visualization
            focus_tokens=top_indices.tolist(),
        )
        
        self.attention_traces.append(trace)
        
    def trace_hidden_state(
        self,
        hidden_state: torch.Tensor,
        layer: int,
    ) -> None:
        """Trace hidden state with PCA dimensionality reduction.
        
        Args:
            hidden_state: Hidden state tensor (shape: [d_model] or [batch, seq_len, d_model])
            layer: Layer index
        """
        # Handle batch and sequence dimensions
        if hidden_state.dim() > 1:
            # Take last position of first batch element
            hidden_state = hidden_state[0, -1, :]
            
        # Initialize PCA if needed
        if not self._pca_initialized:
            self._initialize_pca(hidden_state)
            
        # Apply PCA reduction
        reduced = self._apply_pca(hidden_state)
        
        # Compute norm
        norm = torch.norm(hidden_state).item()
        
        trace = HiddenStateTrace(
            timestamp=datetime.now(),
            reduced_state=reduced.tolist(),
            layer=layer,
            norm=norm,
        )
        
        self.hidden_state_traces.append(trace)
        
    def _initialize_pca(self, hidden_state: torch.Tensor) -> None:
        """Initialize PCA with random projection.
        
        For simplicity, we use random projection instead of computing
        actual PCA components. This provides a stable dimensionality
        reduction without requiring a training set.
        
        Args:
            hidden_state: Sample hidden state to determine dimensions
        """
        d_model = hidden_state.shape[-1]
        
        # Random projection matrix (normalized)
        self._pca_components = torch.randn(
            self.pca_dimensions, d_model,
            device=hidden_state.device,
            dtype=hidden_state.dtype
        )
        # Normalize each component
        self._pca_components = F.normalize(self._pca_components, dim=1)
        
        # Zero mean (will be updated with running average)
        self._pca_mean = torch.zeros(d_model, device=hidden_state.device, dtype=hidden_state.dtype)
        
        self._pca_initialized = True
        
    def _apply_pca(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Apply PCA reduction to hidden state.
        
        Args:
            hidden_state: Hidden state vector
            
        Returns:
            Reduced representation
        """
        if not self._pca_initialized or self._pca_components is None or self._pca_mean is None:
            return torch.zeros(self.pca_dimensions)
            
        # Center the data
        centered = hidden_state - self._pca_mean
        
        # Project onto principal components
        reduced = torch.matmul(self._pca_components, centered)
        
        # Update running mean (exponential moving average)
        alpha = 0.01
        self._pca_mean = (1 - alpha) * self._pca_mean + alpha * hidden_state
        
        return reduced
        
    def trace_decision(
        self,
        decision_type: str,
        options: List[str],
        chosen: str,
        reasoning: str,
        confidence: float,
    ) -> None:
        """Trace a decision point.
        
        Args:
            decision_type: Type of decision (e.g., "environment_switch", "action_selection")
            options: Available options
            chosen: Chosen option
            reasoning: Reasoning for the decision
            confidence: Confidence in the decision (0.0 to 1.0)
        """
        trace = DecisionTrace(
            timestamp=datetime.now(),
            decision_type=decision_type,
            options=options,
            chosen=chosen,
            reasoning=reasoning[:200],  # Truncate for storage
            confidence=confidence,
        )
        
        self.decision_traces.append(trace)
        
    def trace_tool_selection(
        self,
        available_tools: List[str],
        selected_tool: str,
        parameters: Dict[str, Any],
        reasoning: str,
    ) -> None:
        """Trace tool selection reasoning.
        
        Args:
            available_tools: List of available tools
            selected_tool: Selected tool name
            parameters: Tool parameters
            reasoning: Reasoning for tool selection
        """
        trace = ToolSelectionTrace(
            timestamp=datetime.now(),
            available_tools=available_tools,
            selected_tool=selected_tool,
            parameters=parameters,
            reasoning=reasoning[:200],  # Truncate for storage
        )
        
        self.tool_selection_traces.append(trace)
        
    def trace_curiosity_signals(
        self,
        signal_type: str,
        value: float,
        context: str,
    ) -> None:
        """Trace curiosity signal activation.
        
        Args:
            signal_type: Type of curiosity signal (novelty, prediction_error, information_gain, empowerment)
            value: Signal value
            context: Context for the signal
        """
        trace = CuriositySignalTrace(
            timestamp=datetime.now(),
            signal_type=signal_type,
            value=value,
            context=context[:200],  # Truncate for storage
        )
        
        self.curiosity_signal_traces.append(trace)
        
    def get_recent_predictions(self, limit: int = 10) -> List[PredictionTrace]:
        """Get recent prediction traces.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of recent prediction traces
        """
        traces = list(self.prediction_traces)
        return traces[-limit:]
        
    def get_recent_attention(self, limit: int = 10) -> List[AttentionTrace]:
        """Get recent attention traces.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of recent attention traces
        """
        traces = list(self.attention_traces)
        return traces[-limit:]
        
    def get_recent_hidden_states(self, limit: int = 10) -> List[HiddenStateTrace]:
        """Get recent hidden state traces.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of recent hidden state traces
        """
        traces = list(self.hidden_state_traces)
        return traces[-limit:]
        
    def get_recent_decisions(self, limit: int = 10) -> List[DecisionTrace]:
        """Get recent decision traces.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of recent decision traces
        """
        traces = list(self.decision_traces)
        return traces[-limit:]
        
    def get_recent_tool_selections(self, limit: int = 10) -> List[ToolSelectionTrace]:
        """Get recent tool selection traces.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of recent tool selection traces
        """
        traces = list(self.tool_selection_traces)
        return traces[-limit:]
        
    def get_recent_curiosity_signals(self, limit: int = 10) -> List[CuriositySignalTrace]:
        """Get recent curiosity signal traces.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of recent curiosity signal traces
        """
        traces = list(self.curiosity_signal_traces)
        return traces[-limit:]
        
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of all traces.
        
        Returns:
            Dictionary containing trace counts and statistics
        """
        return {
            "prediction_count": len(self.prediction_traces),
            "attention_count": len(self.attention_traces),
            "hidden_state_count": len(self.hidden_state_traces),
            "decision_count": len(self.decision_traces),
            "tool_selection_count": len(self.tool_selection_traces),
            "curiosity_signal_count": len(self.curiosity_signal_traces),
            "avg_prediction_confidence": (
                sum(t.confidence for t in self.prediction_traces) / len(self.prediction_traces)
                if self.prediction_traces else 0.0
            ),
            "avg_decision_confidence": (
                sum(t.confidence for t in self.decision_traces) / len(self.decision_traces)
                if self.decision_traces else 0.0
            ),
        }
