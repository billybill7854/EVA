"""Tool registry — catalogue of every tool EVA knows about.

Separates *capability metadata* (``ToolInfo``) from the *runtime
instance* (``Tool``) so the same registry can describe tools whose
backing packages aren't installed yet (for the acquirer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from eva.tools.base import Tool


@dataclass
class ParameterInfo:
    """One parameter of a tool."""

    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class ToolInfo:
    """Documentation for a tool."""

    name: str
    description: str
    parameters: list[ParameterInfo] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


class ToolRegistry:
    """Maps tool name -> (Tool instance, ToolInfo).

    Discovery is explicit (``discover`` returns names of *available*
    tools); documentation is always available even for unavailable
    tools so EVA knows what it's missing.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._info: dict[str, ToolInfo] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool, info: ToolInfo) -> None:
        name = tool.get_name()
        if info.name != name:
            raise ValueError(
                f"ToolInfo.name ({info.name}) must match tool.get_name() ({name})"
            )
        self._tools[name] = tool
        self._info[name] = info

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._info.pop(name, None)

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------

    def discover(self) -> list[str]:
        """Names of registered tools that are currently available."""

        return sorted(
            name for name, tool in self._tools.items() if tool.is_available()
        )

    def list_all(self) -> list[str]:
        """Names of all registered tools, including unavailable ones."""

        return sorted(self._tools.keys())

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_documentation(self, name: str) -> Optional[ToolInfo]:
        return self._info.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
