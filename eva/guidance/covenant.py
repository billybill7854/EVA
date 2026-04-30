"""Covenant — runtime enforcement of system invariants.

The Covenant is a set of mutual commitments between system layers
and EVA. It enforces honesty, participation, and safety. Violations
are always logged — never silent.
"""

from __future__ import annotations

import logging

from eva.core.tokenizer import (
    SOURCE_TOKEN_MAP,
    EVATokenizer,
)

logger = logging.getLogger(__name__)


class Covenant:
    """Runtime enforcement of system invariants.

    The Covenant ensures:
    1. No layer pretends to be another (source honesty)
    2. No override without EVA's participation
    3. All layers grow (checked externally)
    4. Graceful graduation (competence-based)
    5. Archive immutability
    6. No duplication (portage safety)
    """

    def __init__(self) -> None:
        self._violations: list[dict[str, str]] = []

    def verify_source_honesty(
        self,
        message_tokens: list[int],
        claimed_source: str,
        tokenizer: EVATokenizer,
    ) -> bool:
        """Verify that the appropriate source token is present.

        Args:
            message_tokens: The tokenized message.
            claimed_source: Who claims to have sent this ("human",
                           "scaffold", "ancestor", "self").
            tokenizer: The tokenizer instance.

        Returns:
            True if source token matches claimed source.
        """
        expected_id = SOURCE_TOKEN_MAP.get(claimed_source.lower())
        if expected_id is None:
            self.log_violation(
                "unknown_source",
                f"Unknown source claimed: {claimed_source}",
            )
            return False

        if expected_id not in message_tokens:
            self.log_violation(
                "source_dishonesty",
                f"Message claims source '{claimed_source}' but "
                f"source token {expected_id} not found in tokens.",
            )
            return False

        return True

    def verify_no_override(
        self, action_source: str, eva_consulted: bool
    ) -> bool:
        """Verify that no action overrides EVA without participation.

        Args:
            action_source: Who initiated the action.
            eva_consulted: Whether EVA was consulted/participated.

        Returns:
            True only if EVA participated in the decision.
        """
        if not eva_consulted and action_source != "self":
            self.log_violation(
                "override_without_participation",
                f"Action by '{action_source}' without EVA consultation.",
            )
            return False
        return True

    def check_graduation(
        self, competence_scores: dict[str, float], threshold: float
    ) -> dict[str, bool]:
        """Check graduation readiness for each competence domain.

        Args:
            competence_scores: Dict mapping domain name to score [0, 1].
            threshold: Minimum score for graduation.

        Returns:
            Dict mapping domain to graduation readiness (bool).
        """
        return {
            domain: score >= threshold
            for domain, score in competence_scores.items()
        }

    def verify_archive_immutable(self, archive_modified: bool) -> bool:
        """Verify that the ancestor archive was NOT modified.

        Args:
            archive_modified: Whether any archive file was changed.

        Returns:
            True only if archive was NOT modified.
        """
        if archive_modified:
            self.log_violation(
                "archive_mutation",
                "Ancestor archive was modified. This is forbidden.",
            )
            return False
        return True

    def verify_no_duplicate(
        self, source_active: bool, destination_active: bool
    ) -> bool:
        """Verify no duplication during portage.

        Args:
            source_active: Whether the source EVA is active.
            destination_active: Whether the destination EVA is active.

        Returns:
            True only if exactly one is active.
        """
        if source_active and destination_active:
            self.log_violation(
                "duplication",
                "CRITICAL: Both source and destination EVA are active. "
                "Portage safety violated.",
            )
            return False
        return True

    def log_violation(self, violation_type: str, details: str) -> None:
        """Log a Covenant violation. Never silent.

        Args:
            violation_type: Category of violation.
            details: Human-readable description.
        """
        violation = {
            "type": violation_type,
            "details": details,
        }
        self._violations.append(violation)
        logger.warning(
            "COVENANT VIOLATION [%s]: %s", violation_type, details
        )

    @property
    def violations(self) -> list[dict[str, str]]:
        """All recorded violations."""
        return list(self._violations)

    @property
    def violation_count(self) -> int:
        """Total number of violations recorded."""
        return len(self._violations)
