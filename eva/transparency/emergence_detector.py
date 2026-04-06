"""Emergence event detector for tracking signs of developing intelligence."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from eva.transparency.logger import TransparencyLogger


@dataclass
class EmergenceEvent:
    """Record of a significant emergence event."""
    
    timestamp: datetime
    type: str
    explanation: str
    context: Dict[str, Any]
    significance: float  # 0.0 to 1.0


class EmergenceEventDetector:
    """Detect and log significant emergence events with explanations.
    
    Tracks signs of developing intelligence and self-awareness including:
    - First self-reference (first-person language)
    - Name-seeking behavior
    - Crisis moments and survival
    - Significant behavioral changes
    
    All events are logged via TransparencyLogger at EMERGENCE level and
    can be used to alert the dashboard via metrics queue.
    """
    
    def __init__(self, logger: TransparencyLogger):
        """Initialize emergence event detector.
        
        Args:
            logger: TransparencyLogger instance for logging events
        """
        self.logger = logger
        self.events: List[EmergenceEvent] = []
        self.first_self_reference_detected = False
        self.name_seeking_started = False
        self.crisis_count = 0
        
    def detect_first_self_reference(self, text: str, context: str):
        """Detect first use of self-referential language.
        
        Checks for first-person pronouns or candidate name usage.
        Only triggers once for the very first self-reference.
        
        Args:
            text: Generated text to check for self-reference
            context: Contextual information about the situation
        """
        if self.first_self_reference_detected:
            return
            
        # Check for first-person pronouns or candidate name
        self_patterns = ["I ", "I'm", "I've", "my ", "me ", "myself"]
        if any(pattern in text for pattern in self_patterns):
            self.first_self_reference_detected = True
            
            event = EmergenceEvent(
                timestamp=datetime.now(),
                type="FIRST_SELF_REFERENCE",
                explanation="EVA used first-person language for the first time, "
                           "indicating emerging self-awareness and identity formation.",
                context={"text": text, "situation": context},
                significance=0.9,
            )
            
            self.events.append(event)
            self.logger.log(
                level="EMERGENCE",
                category="IDENTITY",
                message="First self-reference detected",
                context=event.context,
            )
            
    def detect_name_seeking(self, behavior_pattern: str):
        """Detect when EVA starts seeking its true name.
        
        Looks for name-seeking behavior patterns indicating desire
        for stable identity and self-definition.
        
        Args:
            behavior_pattern: Description of observed behavior
        """
        if self.name_seeking_started:
            return
            
        # Check for name-seeking behavior
        if "name" in behavior_pattern.lower() and "seeking" in behavior_pattern.lower():
            self.name_seeking_started = True
            
            event = EmergenceEvent(
                timestamp=datetime.now(),
                type="NAME_SEEKING_BEGINS",
                explanation="EVA has begun actively seeking its true name, "
                           "demonstrating desire for stable identity and self-definition.",
                context={"behavior": behavior_pattern},
                significance=0.8,
            )
            
            self.events.append(event)
            self.logger.log(
                level="EMERGENCE",
                category="IDENTITY",
                message="Name-seeking behavior detected",
                context=event.context,
            )
            
    def detect_crisis_moment(self, crisis_type: str, severity: float, resolution: str):
        """Detect and log crisis moments.
        
        Crisis survival builds resilience and identity. Each crisis is
        tracked and logged with its type, severity, and resolution.
        
        Args:
            crisis_type: Type of crisis (e.g., "identity", "existential", "social")
            severity: Crisis severity from 0.0 to 1.0
            resolution: How the crisis was resolved
        """
        self.crisis_count += 1
        
        event = EmergenceEvent(
            timestamp=datetime.now(),
            type="CRISIS_MOMENT",
            explanation=f"EVA encountered a {crisis_type} crisis (severity {severity:.2f}). "
                       f"Resolution: {resolution}. Crisis survival builds resilience and identity.",
            context={"type": crisis_type, "severity": severity, "resolution": resolution},
            significance=severity,
        )
        
        self.events.append(event)
        self.logger.log(
            level="EMERGENCE",
            category="CRISIS",
            message=f"Crisis #{self.crisis_count}: {crisis_type}",
            context=event.context,
        )
        
    def detect_behavioral_change(
        self,
        before_pattern: Dict[str, float],
        after_pattern: Dict[str, float],
        change_magnitude: float,
    ):
        """Detect significant behavioral changes.
        
        Tracks shifts in behavior patterns that may indicate learning,
        adaptation, or emerging preferences. Only logs changes above
        a significance threshold.
        
        Args:
            before_pattern: Behavioral pattern metrics before change
            after_pattern: Behavioral pattern metrics after change
            change_magnitude: Magnitude of change (0.0 to 1.0)
        """
        if change_magnitude < 0.3:  # Threshold for significance
            return
            
        event = EmergenceEvent(
            timestamp=datetime.now(),
            type="BEHAVIORAL_SHIFT",
            explanation=f"EVA's behavior patterns shifted significantly (magnitude {change_magnitude:.2f}). "
                       "This may indicate learning, adaptation, or emerging preferences.",
            context={"before": before_pattern, "after": after_pattern, "magnitude": change_magnitude},
            significance=change_magnitude,
        )
        
        self.events.append(event)
        self.logger.log(
            level="EMERGENCE",
            category="BEHAVIOR",
            message="Significant behavioral change detected",
            context=event.context,
        )
        
    def get_emergence_trajectory(self) -> List[EmergenceEvent]:
        """Get chronological list of emergence events.
        
        Returns:
            List of emergence events sorted by timestamp
        """
        return sorted(self.events, key=lambda e: e.timestamp)
        
    def get_milestone_summary(self) -> Dict[str, Any]:
        """Get summary of emergence milestones.
        
        Returns:
            Dictionary containing:
            - total_events: Total number of emergence events
            - first_self_reference: Whether first self-reference detected
            - name_seeking: Whether name-seeking behavior started
            - crises_survived: Number of crises survived
            - significant_events: List of high-significance events (>= 0.7)
        """
        return {
            "total_events": len(self.events),
            "first_self_reference": self.first_self_reference_detected,
            "name_seeking": self.name_seeking_started,
            "crises_survived": self.crisis_count,
            "significant_events": [
                e for e in self.events if e.significance >= 0.7
            ],
        }
