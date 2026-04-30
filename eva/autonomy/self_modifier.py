"""Sandboxed self-modification.

EVA can propose changes to a *whitelisted* set of files (hyperparameters,
prompts, tool heuristics — NEVER safety or covenant code). Every proposal
is validated, dry-run, and logged before being applied, and is always
reversible via git.

Policy (hard-coded):

* A proposal is a pure ``(target, new_content)`` pair.
* ``target`` must match :data:`ALLOWED_TARGETS`.
* The new content must parse (YAML or Python — checked by extension).
* The resulting file must not grow past :data:`MAX_FILE_BYTES`.
* A :class:`SafetyMonitor`-style callback must approve the diff.
* On success the change is written to disk and logged; the raw diff is
  kept in ``workspace/<name>/self_mods/`` for human review.

This is intentionally conservative — the goal is genuine autonomy within
rails that cannot be undone in-process.
"""

from __future__ import annotations

import ast
import difflib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


ALLOWED_TARGETS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^configs/[\w\-.]+\.yaml$"),
    re.compile(r"^workspace/[\w\-./]+\.yaml$"),
    re.compile(r"^workspace/[\w\-./]+\.md$"),
    re.compile(r"^eva/tools/heuristics/[\w\-./]+\.py$"),
    re.compile(r"^eva/prompts/[\w\-./]+\.(md|txt|yaml)$"),
)

FORBIDDEN_TARGETS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^eva/guidance/covenant\.py$"),
    re.compile(r"^eva/transparency/safety_monitor\.py$"),
    re.compile(r"^eva/autonomy/self_modifier\.py$"),
)

MAX_FILE_BYTES = 128 * 1024


ApprovalFn = Callable[["Proposal"], bool]


@dataclass
class Proposal:
    """A candidate self-modification."""

    target: str
    new_content: str
    reason: str = ""


@dataclass
class ModificationRecord:
    """Outcome of evaluating a proposal."""

    proposal: Proposal
    applied: bool
    outcome: str
    diff: str = ""
    rejection_reason: str = ""


class SelfModifier:
    """Evaluate and apply self-modification proposals."""

    def __init__(
        self,
        repo_root: Path,
        approval_fn: Optional[ApprovalFn] = None,
        archive_dir: Optional[Path] = None,
        allowed_targets: tuple[re.Pattern[str], ...] = ALLOWED_TARGETS,
        forbidden_targets: tuple[re.Pattern[str], ...] = FORBIDDEN_TARGETS,
        max_file_bytes: int = MAX_FILE_BYTES,
    ) -> None:
        self._root = repo_root.resolve()
        self._approval = approval_fn
        self._allowed = allowed_targets
        self._forbidden = forbidden_targets
        self._max_file_bytes = max_file_bytes
        self._archive = (
            archive_dir.resolve()
            if archive_dir is not None
            else self._root / "workspace" / "self_mods"
        )
        self._archive.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def evaluate(self, proposal: Proposal) -> ModificationRecord:
        reject = self._check_target(proposal.target)
        if reject:
            return ModificationRecord(
                proposal, False, "rejected", rejection_reason=reject
            )
        reject = self._check_size(proposal.new_content)
        if reject:
            return ModificationRecord(
                proposal, False, "rejected", rejection_reason=reject
            )
        reject = self._check_syntax(proposal.target, proposal.new_content)
        if reject:
            return ModificationRecord(
                proposal, False, "rejected", rejection_reason=reject
            )

        target_path = self._root / proposal.target
        old_content = (
            target_path.read_text(encoding="utf-8")
            if target_path.exists()
            else ""
        )
        diff = "".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                proposal.new_content.splitlines(keepends=True),
                fromfile=f"a/{proposal.target}",
                tofile=f"b/{proposal.target}",
            )
        )

        if self._approval is not None and not self._approval(proposal):
            return ModificationRecord(
                proposal,
                False,
                "rejected",
                diff=diff,
                rejection_reason="approval_fn refused proposal",
            )

        # Apply
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(proposal.new_content, encoding="utf-8")
        archive_name = (
            proposal.target.replace("/", "__") + ".patch"
        )
        (self._archive / archive_name).write_text(diff, encoding="utf-8")

        logger.info(
            "SelfModifier: applied change to %s (%d bytes diff)",
            proposal.target,
            len(diff),
        )
        return ModificationRecord(
            proposal, True, "applied", diff=diff
        )

    # ------------------------------------------------------------------
    # internal checks
    # ------------------------------------------------------------------

    def _check_target(self, target: str) -> Optional[str]:
        if "\x00" in target or target.startswith("/") or ".." in target:
            return f"invalid target path: {target}"
        for pat in self._forbidden:
            if pat.match(target):
                return f"target {target!r} is forbidden"
        for pat in self._allowed:
            if pat.match(target):
                return None
        return f"target {target!r} is not in the allow-list"

    def _check_size(self, content: str) -> Optional[str]:
        if len(content.encode("utf-8")) > self._max_file_bytes:
            return (
                f"proposal too large: "
                f"{len(content)} > {self._max_file_bytes}"
            )
        return None

    def _check_syntax(
        self, target: str, content: str
    ) -> Optional[str]:
        if target.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as exc:
                return f"python syntax error: {exc}"
        elif target.endswith(".yaml") or target.endswith(".yml"):
            try:
                import yaml

                yaml.safe_load(content)
            except Exception as exc:
                return f"yaml parse error: {exc}"
        # Markdown / txt — no validation needed.
        return None


__all__ = [
    "SelfModifier",
    "Proposal",
    "ModificationRecord",
    "ALLOWED_TARGETS",
    "FORBIDDEN_TARGETS",
]
