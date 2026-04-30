"""Tool usage statistics.

Feeds the curiosity tool-reward bonus (:class:`eva.curiosity.reward.CuriosityEngine`).

Reward contract (used by tests):

* +0.5 on the very first *successful* use of a tool.
* +0.2 when a successful use introduces a new usage pattern (pattern is
  a coarse string of the arguments).
* -0.1 when a successful use is repetitive and overall diversity for
  the tool has dropped below 0.5.
* 0.0 for failures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ToolUsage:
    """One recorded invocation of a tool."""

    tool_name: str
    pattern: str
    success: bool
    result: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ToolUsageTracker:
    """In-memory usage log with success / diversity / first-success queries."""

    def __init__(self) -> None:
        self._usages: list[ToolUsage] = []
        self._per_tool: dict[str, list[ToolUsage]] = defaultdict(list)

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------

    def record_usage(
        self,
        tool_name: str,
        pattern: str,
        success: bool,
        result: str = "",
    ) -> ToolUsage:
        usage = ToolUsage(
            tool_name=tool_name,
            pattern=pattern,
            success=success,
            result=result,
        )
        self._usages.append(usage)
        self._per_tool[tool_name].append(usage)
        return usage

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def get_success_rate(self, tool_name: str) -> float:
        usages = self._per_tool.get(tool_name, [])
        if not usages:
            return 0.0
        successes = sum(1 for u in usages if u.success)
        return successes / len(usages)

    def get_usage_diversity(self, tool_name: str) -> float:
        usages = self._per_tool.get(tool_name, [])
        if not usages:
            return 0.0
        unique = len({u.pattern for u in usages})
        return unique / len(usages)

    def is_first_success(self, tool_name: str) -> bool:
        """True iff there has been exactly one successful use."""

        usages = self._per_tool.get(tool_name, [])
        successes = [u for u in usages if u.success]
        return len(successes) == 1

    def has_new_pattern(self, tool_name: str) -> bool:
        """True if the most recent successful use introduced a new pattern."""

        usages = [u for u in self._per_tool.get(tool_name, []) if u.success]
        if len(usages) < 2:
            return False
        last = usages[-1].pattern
        return all(u.pattern != last for u in usages[:-1])

    def get_statistics(self) -> dict:
        tools = list(self._per_tool.keys())
        return {
            "total_invocations": len(self._usages),
            "unique_tools_used": len(tools),
            "success_rates": {t: self.get_success_rate(t) for t in tools},
            "diversity_scores": {t: self.get_usage_diversity(t) for t in tools},
        }

    def recent(self, limit: int = 20) -> list[ToolUsage]:
        return list(self._usages[-limit:])
