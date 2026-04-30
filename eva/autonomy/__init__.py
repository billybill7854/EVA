"""Autonomy — self-model and self-modification subsystems."""

from eva.autonomy.self_model import SelfModelSystem, SelfStateSnapshot
from eva.autonomy.self_modifier import (
    ModificationRecord,
    Proposal,
    SelfModifier,
)

__all__ = [
    "SelfModelSystem",
    "SelfStateSnapshot",
    "SelfModifier",
    "Proposal",
    "ModificationRecord",
]
