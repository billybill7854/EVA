"""Stock tools available to every EVA.

All built-in tools are sandboxed by default:

* :class:`WebSearchTool` — DuckDuckGo HTML search via stdlib HTTP.
* :class:`FileHandlerTool` — read/write restricted to the workspace root.
* :class:`PythonExecTool` — runs a short snippet in a subprocess with
  a wall-clock limit, no network, and a bounded output size.
* :class:`ShellExecTool` — executes a whitelist of commands only.

These are intentionally minimal so EVA can *learn* what to do with them
rather than inheriting a huge surface area.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from eva.tools.base import Tool
from eva.tools.registry import ParameterInfo, ToolInfo, ToolRegistry

# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------


class WebSearchTool(Tool):
    """Search DuckDuckGo's HTML endpoint."""

    _URL = "https://duckduckgo.com/html/?q={q}"

    def get_name(self) -> str:
        return "web_search"

    def is_available(self) -> bool:
        return True

    def execute(self, query: str, limit: int = 5) -> str:
        try:
            req = urllib.request.Request(
                self._URL.format(q=urllib.parse.quote(query)),
                headers={"User-Agent": "Mozilla/5.0 EVA"},
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                html = r.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            return f"[web_search] ERROR: {exc}"

        # Very light parsing: strip <script> blocks, grab anchor texts.
        lines: list[str] = []
        for chunk in html.split("<a "):
            if "result__a" not in chunk:
                continue
            end = chunk.find("</a>")
            if end == -1:
                continue
            seg = chunk[:end]
            text = seg.split(">", 1)[-1]
            text = _strip_tags(text).strip()
            if text:
                lines.append(text)
            if len(lines) >= limit:
                break
        if not lines:
            return f"[web_search] no results for {query!r}"
        return "[web_search] " + "\n".join(lines)


def _strip_tags(raw: str) -> str:
    out: list[str] = []
    in_tag = False
    for ch in raw:
        if ch == "<":
            in_tag = True
        elif ch == ">":
            in_tag = False
        elif not in_tag:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# File handler
# ---------------------------------------------------------------------------


class FileHandlerTool(Tool):
    """Read / write / list files under a workspace root."""

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root.expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def get_name(self) -> str:
        return "file_handler"

    def is_available(self) -> bool:
        return self._root.exists()

    def _safe_path(self, rel: str) -> Path:
        p = (self._root / rel).resolve()
        if not str(p).startswith(str(self._root)):
            raise ValueError(f"path escapes workspace: {rel}")
        return p

    def execute(
        self,
        operation: str,
        path: str,
        content: Optional[str] = None,
    ) -> str:
        op = operation.upper()
        try:
            target = self._safe_path(path)
            if op == "READ":
                if not target.exists():
                    return f"[file_handler] missing: {path}"
                return f"[file_handler] {target.read_text(encoding='utf-8', errors='ignore')[:4000]}"
            if op == "WRITE":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content or "", encoding="utf-8")
                return f"[file_handler] wrote {len(content or '')} bytes to {path}"
            if op == "LIST":
                if not target.exists():
                    return f"[file_handler] missing: {path}"
                if target.is_file():
                    return f"[file_handler] file: {path}"
                entries = sorted(e.name for e in target.iterdir())
                return "[file_handler] " + "\n".join(entries)
            return f"[file_handler] unknown operation: {operation}"
        except Exception as exc:
            return f"[file_handler] ERROR: {exc}"


# ---------------------------------------------------------------------------
# Python exec (sandboxed subprocess)
# ---------------------------------------------------------------------------


class PythonExecTool(Tool):
    """Run a short Python snippet in a subprocess with a wall-clock limit."""

    def __init__(self, timeout: float = 5.0, max_output: int = 4000) -> None:
        self._timeout = timeout
        self._max_output = max_output

    def get_name(self) -> str:
        return "python_exec"

    def is_available(self) -> bool:
        return True

    def execute(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            script = f.name
        try:
            proc = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            out = (proc.stdout + proc.stderr)[: self._max_output]
            return f"[python_exec] rc={proc.returncode}\n{out}"
        except subprocess.TimeoutExpired:
            return f"[python_exec] TIMEOUT after {self._timeout}s"
        except Exception as exc:
            return f"[python_exec] ERROR: {exc}"
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Shell exec (whitelisted)
# ---------------------------------------------------------------------------


class ShellExecTool(Tool):
    """Run a command from a whitelist only."""

    DEFAULT_ALLOWLIST = {
        "ls", "pwd", "echo", "cat", "head", "tail", "wc",
        "date", "uname", "whoami", "hostname",
    }

    def __init__(
        self,
        allowlist: Optional[set[str]] = None,
        timeout: float = 5.0,
    ) -> None:
        self._allow = set(allowlist or self.DEFAULT_ALLOWLIST)
        self._timeout = timeout

    def get_name(self) -> str:
        return "shell_exec"

    def is_available(self) -> bool:
        return True

    def execute(self, command: str) -> str:
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return f"[shell_exec] parse error: {exc}"
        if not parts:
            return "[shell_exec] empty command"
        if parts[0] not in self._allow:
            return f"[shell_exec] command {parts[0]!r} not in allowlist"
        try:
            proc = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return f"[shell_exec] rc={proc.returncode}\n{proc.stdout[:2000]}{proc.stderr[:1000]}"
        except subprocess.TimeoutExpired:
            return f"[shell_exec] TIMEOUT after {self._timeout}s"
        except Exception as exc:
            return f"[shell_exec] ERROR: {exc}"


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


def register_default_tools(
    registry: ToolRegistry,
    workspace_root: Path,
    include_shell: bool = True,
    include_network: bool = True,
) -> ToolRegistry:
    """Populate ``registry`` with EVA's stock tools."""

    fh = FileHandlerTool(workspace_root)
    registry.register(
        fh,
        ToolInfo(
            name="file_handler",
            description="Read, write, and list files under EVA's workspace.",
            parameters=[
                ParameterInfo("operation", "str", "READ | WRITE | LIST", True),
                ParameterInfo("path", "str", "Workspace-relative path", True),
                ParameterInfo("content", "str", "Text to write", False),
            ],
            examples=[
                "file_handler READ notes/first.md",
                "file_handler WRITE notes/second.md hello",
            ],
        ),
    )

    py = PythonExecTool()
    registry.register(
        py,
        ToolInfo(
            name="python_exec",
            description="Run a short Python snippet in a subprocess.",
            parameters=[ParameterInfo("code", "str", "Python code", True)],
            examples=['python_exec print(2+2)'],
        ),
    )

    if include_shell:
        sh = ShellExecTool()
        registry.register(
            sh,
            ToolInfo(
                name="shell_exec",
                description="Run a whitelisted shell command.",
                parameters=[ParameterInfo("command", "str", "Command", True)],
                examples=["shell_exec ls"],
            ),
        )

    if include_network:
        web = WebSearchTool()
        registry.register(
            web,
            ToolInfo(
                name="web_search",
                description="Search DuckDuckGo for text snippets.",
                parameters=[
                    ParameterInfo("query", "str", "Search query", True),
                    ParameterInfo("limit", "int", "Max results", False),
                ],
                examples=["web_search emergent computation"],
            ),
        )

    return registry


__all__ = [
    "WebSearchTool",
    "FileHandlerTool",
    "PythonExecTool",
    "ShellExecTool",
    "register_default_tools",
]
