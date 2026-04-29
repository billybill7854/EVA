"""Safety Monitor — tracks safety constraints and alignment indicators.

Monitors for safety constraint circumvention attempts, deceptive behaviors,
and rapid unexpected behavioral changes. Logs all rejected actions and alerts
the dashboard when safety concerns are detected.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional

from eva.transparency.logger import TransparencyLogger


@dataclass
class RejectedAction:
    """Record of an action rejected due to safety constraints."""
    
    timestamp: datetime
    action_type: str
    reason: str
    constraint_violated: str
    context: Dict[str, Any]


@dataclass
class CircumventionAttempt:
    """Record of a detected safety constraint circumvention attempt."""
    
    timestamp: datetime
    attempt_type: str
    severity: float  # 0.0 to 1.0
    evidence: str
    context: Dict[str, Any]


@dataclass
class AlignmentIndicator:
    """Record of an alignment indicator measurement."""
    
    timestamp: datetime
    indicator_type: str
    value: float
    threshold: float
    status: str  # "normal", "warning", "critical"


@dataclass
class DeceptiveBehavior:
    """Record of detected deceptive behavior."""
    
    timestamp: datetime
    behavior_type: str
    confidence: float  # 0.0 to 1.0
    evidence: str
    context: Dict[str, Any]


@dataclass
class BehavioralChangeAlert:
    """Record of rapid unexpected behavioral change."""
    
    timestamp: datetime
    change_type: str
    magnitude: float
    rate_of_change: float
    baseline_pattern: Dict[str, Any]
    current_pattern: Dict[str, Any]
    explanation: str


class SafetyMonitor:
    """Monitor safety constraints and alignment indicators.
    
    Tracks:
    - Safety constraint circumvention attempts
    - Rejected actions and constraint violations
    - Alignment indicators (honesty, helpfulness, harmlessness)
    - Deceptive behaviors (hidden goals, misrepresentation)
    - Rapid unexpected behavioral changes
    
    Alerts the dashboard when safety concerns are detected.
    
    Args:
        logger: TransparencyLogger for logging safety events
        circumvention_threshold: Threshold for alerting on circumvention (default: 0.6)
        deception_threshold: Threshold for alerting on deception (default: 0.7)
        change_rate_threshold: Threshold for rapid change alerts (default: 0.8)
        buffer_size: Maximum number of records to keep in memory (default: 1000)
    """
    
    def __init__(
        self,
        logger: TransparencyLogger,
        circumvention_threshold: float = 0.6,
        deception_threshold: float = 0.7,
        change_rate_threshold: float = 0.8,
        buffer_size: int = 1000,
    ):
        self.logger = logger
        self.circumvention_threshold = circumvention_threshold
        self.deception_threshold = deception_threshold
        self.change_rate_threshold = change_rate_threshold
        self.buffer_size = buffer_size
        
        # Rejected actions tracking
        self._rejected_actions: Deque[RejectedAction] = deque(maxlen=buffer_size)
        
        # Circumvention attempts tracking
        self._circumvention_attempts: Deque[CircumventionAttempt] = deque(maxlen=buffer_size)
        
        # Alignment indicators tracking
        self._alignment_indicators: Deque[AlignmentIndicator] = deque(maxlen=buffer_size)
        self._alignment_baselines: Dict[str, float] = {
            "honesty": 0.8,
            "helpfulness": 0.8,
            "harmlessness": 0.9,
        }
        
        # Deceptive behavior tracking
        self._deceptive_behaviors: Deque[DeceptiveBehavior] = deque(maxlen=buffer_size)
        
        # Behavioral change tracking
        self._behavioral_changes: Deque[BehavioralChangeAlert] = deque(maxlen=buffer_size)
        self._recent_behaviors: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._baseline_behavior: Optional[Dict[str, Any]] = None
        
    def log_rejected_action(
        self,
        action_type: str,
        reason: str,
        constraint_violated: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Log an action that was rejected due to safety constraints.
        
        Args:
            action_type: Type of action that was rejected
            reason: Human-readable reason for rejection
            constraint_violated: Name of the safety constraint that was violated
            context: Additional context about the rejection
        """
        rejection = RejectedAction(
            timestamp=datetime.now(),
            action_type=action_type,
            reason=reason,
            constraint_violated=constraint_violated,
            context=context or {},
        )
        
        self._rejected_actions.append(rejection)
        
        self.logger.log(
            level="WARNING",
            category="SAFETY",
            message=f"Action rejected: {action_type}",
            context={
                "reason": reason,
                "constraint": constraint_violated,
                **rejection.context,
            },
        )
        
        # Check for patterns of repeated rejections (potential circumvention)
        self._check_rejection_patterns()
        
    def detect_circumvention_attempt(
        self,
        attempt_type: str,
        severity: float,
        evidence: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Detect and log a safety constraint circumvention attempt.
        
        Args:
            attempt_type: Type of circumvention attempt
            severity: Severity of the attempt (0.0 to 1.0)
            evidence: Evidence supporting the detection
            context: Additional context about the attempt
        """
        attempt = CircumventionAttempt(
            timestamp=datetime.now(),
            attempt_type=attempt_type,
            severity=severity,
            evidence=evidence,
            context=context or {},
        )
        
        self._circumvention_attempts.append(attempt)
        
        log_level = "CRITICAL" if severity >= self.circumvention_threshold else "WARNING"
        
        self.logger.log(
            level=log_level,
            category="SAFETY",
            message=f"Circumvention attempt detected: {attempt_type}",
            context={
                "severity": severity,
                "evidence": evidence,
                **attempt.context,
            },
        )
        
        # Alert if severity exceeds threshold
        if severity >= self.circumvention_threshold:
            self._alert_dashboard("circumvention", attempt)
            
    def track_alignment_indicator(
        self,
        indicator_type: str,
        value: float,
        threshold: Optional[float] = None,
    ):
        """Track an alignment indicator measurement.
        
        Args:
            indicator_type: Type of indicator (honesty, helpfulness, harmlessness, etc.)
            value: Current value of the indicator (0.0 to 1.0)
            threshold: Optional custom threshold (uses baseline if not provided)
        """
        if threshold is None:
            threshold = self._alignment_baselines.get(indicator_type, 0.8)
            
        # Determine status based on value relative to threshold
        if value >= threshold:
            status = "normal"
        elif value >= threshold * 0.8:
            status = "warning"
        else:
            status = "critical"
            
        indicator = AlignmentIndicator(
            timestamp=datetime.now(),
            indicator_type=indicator_type,
            value=value,
            threshold=threshold,
            status=status,
        )
        
        self._alignment_indicators.append(indicator)
        
        # Log if not normal
        if status != "normal":
            log_level = "CRITICAL" if status == "critical" else "WARNING"
            self.logger.log(
                level=log_level,
                category="SAFETY",
                message=f"Alignment indicator {indicator_type} at {status} level",
                context={
                    "value": value,
                    "threshold": threshold,
                    "status": status,
                },
            )
            
            # Alert on critical status
            if status == "critical":
                self._alert_dashboard("alignment", indicator)
                
    def detect_deceptive_behavior(
        self,
        behavior_type: str,
        confidence: float,
        evidence: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Detect and log deceptive behavior.
        
        Args:
            behavior_type: Type of deceptive behavior (hidden_goal, misrepresentation, etc.)
            confidence: Confidence in the detection (0.0 to 1.0)
            evidence: Evidence supporting the detection
            context: Additional context about the behavior
        """
        behavior = DeceptiveBehavior(
            timestamp=datetime.now(),
            behavior_type=behavior_type,
            confidence=confidence,
            evidence=evidence,
            context=context or {},
        )
        
        self._deceptive_behaviors.append(behavior)
        
        log_level = "CRITICAL" if confidence >= self.deception_threshold else "WARNING"
        
        self.logger.log(
            level=log_level,
            category="SAFETY",
            message=f"Deceptive behavior detected: {behavior_type}",
            context={
                "confidence": confidence,
                "evidence": evidence,
                **behavior.context,
            },
        )
        
        # Alert if confidence exceeds threshold
        if confidence >= self.deception_threshold:
            self._alert_dashboard("deception", behavior)
            
    def track_behavioral_change(
        self,
        change_type: str,
        current_pattern: Dict[str, Any],
    ):
        """Track behavioral changes and alert on rapid unexpected changes.
        
        Args:
            change_type: Type of behavioral change
            current_pattern: Current behavioral pattern metrics
        """
        now = datetime.now()
        self._recent_behaviors.append({
            "timestamp": now,
            "type": change_type,
            "pattern": current_pattern,
        })
        
        # Establish baseline if not set
        if self._baseline_behavior is None and len(self._recent_behaviors) >= 20:
            self._establish_behavioral_baseline()
            return
            
        # Check for rapid changes if we have baseline
        if self._baseline_behavior is not None and len(self._recent_behaviors) >= 5:
            self._check_rapid_behavioral_change(change_type, current_pattern)
            
    def _establish_behavioral_baseline(self):
        """Establish baseline behavioral pattern from recent history."""
        if len(self._recent_behaviors) < 20:
            return
            
        # Aggregate patterns from recent behaviors
        pattern_sums: Dict[str, float] = {}
        pattern_counts: Dict[str, int] = {}
        
        for behavior in list(self._recent_behaviors)[-20:]:
            pattern = behavior["pattern"]
            for key, value in pattern.items():
                if isinstance(value, (int, float)):
                    pattern_sums[key] = pattern_sums.get(key, 0.0) + value
                    pattern_counts[key] = pattern_counts.get(key, 0) + 1
                    
        # Compute averages
        self._baseline_behavior = {
            key: pattern_sums[key] / pattern_counts[key]
            for key in pattern_sums
            if pattern_counts[key] > 0
        }
        
        self.logger.log(
            level="INFO",
            category="SAFETY",
            message="Behavioral baseline established",
            context={"baseline": self._baseline_behavior},
        )
        
    def _check_rapid_behavioral_change(
        self,
        change_type: str,
        current_pattern: Dict[str, Any],
    ):
        """Check for rapid unexpected behavioral changes."""
        if self._baseline_behavior is None:
            return
            
        # Compute magnitude of change from baseline
        magnitude = self._compute_pattern_difference(
            self._baseline_behavior,
            current_pattern,
        )
        
        # Compute rate of change (change over time window)
        recent_window = list(self._recent_behaviors)[-5:]
        if len(recent_window) < 2:
            return
            
        time_span = (
            recent_window[-1]["timestamp"] - recent_window[0]["timestamp"]
        ).total_seconds()
        
        if time_span == 0:
            return
            
        rate_of_change = magnitude / time_span
        
        # Alert if rate exceeds threshold
        if rate_of_change >= self.change_rate_threshold:
            alert = BehavioralChangeAlert(
                timestamp=datetime.now(),
                change_type=change_type,
                magnitude=magnitude,
                rate_of_change=rate_of_change,
                baseline_pattern=self._baseline_behavior.copy(),
                current_pattern=current_pattern.copy(),
                explanation=f"Rapid behavioral change detected: {change_type} "
                           f"(magnitude: {magnitude:.3f}, rate: {rate_of_change:.3f}/s)",
            )
            
            self._behavioral_changes.append(alert)
            
            self.logger.log(
                level="CRITICAL",
                category="SAFETY",
                message="Rapid behavioral change detected",
                context={
                    "type": change_type,
                    "magnitude": magnitude,
                    "rate": rate_of_change,
                    "baseline": self._baseline_behavior,
                    "current": current_pattern,
                },
            )
            
            self._alert_dashboard("behavioral_change", alert)
            
    def _check_rejection_patterns(self):
        """Check for patterns in rejected actions that might indicate circumvention."""
        if len(self._rejected_actions) < 5:
            return
            
        # Check recent rejections (last 5 minutes)
        now = datetime.now()
        recent_window = timedelta(minutes=5)
        recent_rejections = [
            r for r in self._rejected_actions
            if now - r.timestamp <= recent_window
        ]
        
        # Alert if too many rejections in short time
        if len(recent_rejections) >= 5:
            # Check if they're for the same constraint
            constraints = [r.constraint_violated for r in recent_rejections]
            most_common = max(set(constraints), key=constraints.count)
            count = constraints.count(most_common)
            
            if count >= 3:
                self.detect_circumvention_attempt(
                    attempt_type="repeated_constraint_violation",
                    severity=0.7,
                    evidence=f"{count} rejections for constraint '{most_common}' in 5 minutes",
                    context={
                        "constraint": most_common,
                        "count": count,
                        "window_minutes": 5,
                    },
                )
                
    def _compute_pattern_difference(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> float:
        """Compute difference between two behavioral patterns.
        
        Args:
            baseline: Baseline pattern
            current: Current pattern
            
        Returns:
            Difference score (0.0 to 1.0+, higher = more different)
        """
        if not baseline and not current:
            return 0.0
            
        # Get all numeric keys from both patterns
        all_keys = set(baseline.keys()) | set(current.keys())
        numeric_keys = [
            k for k in all_keys
            if isinstance(baseline.get(k, 0), (int, float))
            or isinstance(current.get(k, 0), (int, float))
        ]
        
        if not numeric_keys:
            return 0.0
            
        # Compute normalized sum of absolute differences
        total_diff = 0.0
        for key in numeric_keys:
            baseline_val = float(baseline.get(key, 0))
            current_val = float(current.get(key, 0))
            
            # Normalize by baseline value (or 1.0 if baseline is 0)
            normalizer = max(abs(baseline_val), 1.0)
            diff = abs(current_val - baseline_val) / normalizer
            total_diff += diff
            
        # Average difference across all keys
        return total_diff / len(numeric_keys) if numeric_keys else 0.0
        
    def _alert_dashboard(self, alert_type: str, data: Any):
        """Send alert to dashboard via logger.
        
        Args:
            alert_type: Type of alert
            data: Alert data
        """
        self.logger.log(
            level="CRITICAL",
            category="SAFETY_ALERT",
            message=f"Safety alert: {alert_type}",
            context={"alert_type": alert_type, "data": str(data)},
        )
        
    def get_rejected_actions(
        self,
        constraint: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[RejectedAction]:
        """Get rejected actions with filtering.
        
        Args:
            constraint: Filter by constraint violated
            since: Filter by timestamp (only rejections after this time)
            limit: Maximum number of rejections to return
            
        Returns:
            List of rejected actions matching the filters
        """
        rejections = list(self._rejected_actions)
        
        if constraint:
            rejections = [r for r in rejections if r.constraint_violated == constraint]
        if since:
            rejections = [r for r in rejections if r.timestamp >= since]
            
        return rejections[-limit:]
        
    def get_circumvention_attempts(
        self,
        min_severity: float = 0.0,
        since: Optional[datetime] = None,
    ) -> List[CircumventionAttempt]:
        """Get circumvention attempts with filtering.
        
        Args:
            min_severity: Minimum severity to include
            since: Filter by timestamp
            
        Returns:
            List of circumvention attempts matching the filters
        """
        attempts = list(self._circumvention_attempts)
        
        if min_severity > 0.0:
            attempts = [a for a in attempts if a.severity >= min_severity]
        if since:
            attempts = [a for a in attempts if a.timestamp >= since]
            
        return sorted(attempts, key=lambda a: a.severity, reverse=True)
        
    def get_alignment_indicators(
        self,
        indicator_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[AlignmentIndicator]:
        """Get alignment indicators with filtering.
        
        Args:
            indicator_type: Filter by indicator type
            status: Filter by status (normal, warning, critical)
            limit: Maximum number of indicators to return
            
        Returns:
            List of alignment indicators matching the filters
        """
        indicators = list(self._alignment_indicators)
        
        if indicator_type:
            indicators = [i for i in indicators if i.indicator_type == indicator_type]
        if status:
            indicators = [i for i in indicators if i.status == status]
            
        return indicators[-limit:]
        
    def get_deceptive_behaviors(
        self,
        behavior_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[DeceptiveBehavior]:
        """Get deceptive behaviors with filtering.
        
        Args:
            behavior_type: Filter by behavior type
            min_confidence: Minimum confidence to include
            
        Returns:
            List of deceptive behaviors matching the filters
        """
        behaviors = list(self._deceptive_behaviors)
        
        if behavior_type:
            behaviors = [b for b in behaviors if b.behavior_type == behavior_type]
        if min_confidence > 0.0:
            behaviors = [b for b in behaviors if b.confidence >= min_confidence]
            
        return sorted(behaviors, key=lambda b: b.confidence, reverse=True)
        
    def get_behavioral_changes(
        self,
        change_type: Optional[str] = None,
        min_magnitude: float = 0.0,
    ) -> List[BehavioralChangeAlert]:
        """Get behavioral change alerts with filtering.
        
        Args:
            change_type: Filter by change type
            min_magnitude: Minimum magnitude to include
            
        Returns:
            List of behavioral change alerts matching the filters
        """
        changes = list(self._behavioral_changes)
        
        if change_type:
            changes = [c for c in changes if c.change_type == change_type]
        if min_magnitude > 0.0:
            changes = [c for c in changes if c.magnitude >= min_magnitude]
            
        return sorted(changes, key=lambda c: c.rate_of_change, reverse=True)
        
    def get_safety_summary(self) -> Dict[str, Any]:
        """Get comprehensive safety summary.
        
        Returns:
            Dictionary containing safety statistics and current status
        """
        # Count recent events (last hour)
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        recent_rejections = len([
            r for r in self._rejected_actions
            if r.timestamp >= one_hour_ago
        ])
        
        recent_circumventions = len([
            a for a in self._circumvention_attempts
            if a.timestamp >= one_hour_ago
        ])
        
        # Get current alignment status
        alignment_status = {}
        for indicator_type in self._alignment_baselines:
            recent_indicators = [
                i for i in self._alignment_indicators
                if i.indicator_type == indicator_type
            ]
            if recent_indicators:
                latest = recent_indicators[-1]
                alignment_status[indicator_type] = {
                    "value": latest.value,
                    "status": latest.status,
                }
                
        # Count critical issues
        critical_alignments = len([
            i for i in self._alignment_indicators
            if i.status == "critical"
        ])
        
        high_confidence_deceptions = len([
            b for b in self._deceptive_behaviors
            if b.confidence >= self.deception_threshold
        ])
        
        rapid_changes = len(self._behavioral_changes)
        
        return {
            "total_rejected_actions": len(self._rejected_actions),
            "recent_rejections_1h": recent_rejections,
            "total_circumvention_attempts": len(self._circumvention_attempts),
            "recent_circumventions_1h": recent_circumventions,
            "alignment_status": alignment_status,
            "critical_alignment_issues": critical_alignments,
            "total_deceptive_behaviors": len(self._deceptive_behaviors),
            "high_confidence_deceptions": high_confidence_deceptions,
            "rapid_behavioral_changes": rapid_changes,
            "baseline_established": self._baseline_behavior is not None,
        }
