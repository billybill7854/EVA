"""Tool-learning subsystem.

EVA autonomously discovers, uses, and (with approval) acquires new tools.
Public API:

* :class:`eva.tools.base.Tool` — abstract tool interface.
* :class:`eva.tools.registry.ToolRegistry` / :class:`ToolInfo` /
  :class:`ParameterInfo` — capability catalogue.
* :class:`eva.tools.usage_tracker.ToolUsageTracker` / :class:`ToolUsage`
  — usage statistics feeding the curiosity tool-reward bonus.
* :class:`eva.tools.acquirer.ToolAcquirer` — safe runtime acquisition.
* :func:`eva.tools.builtins.register_default_tools` — populate a registry
  with the stock tools (web search, file handler, python exec, shell).
"""

from eva.tools.base import Tool
from eva.tools.registry import ParameterInfo, ToolInfo, ToolRegistry
from eva.tools.usage_tracker import ToolUsage, ToolUsageTracker

__all__ = [
    "Tool",
    "ParameterInfo",
    "ToolInfo",
    "ToolRegistry",
    "ToolUsage",
    "ToolUsageTracker",
]
