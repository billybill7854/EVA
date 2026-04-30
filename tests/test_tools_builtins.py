"""Unit tests for the built-in tools and acquirer allow-list."""

from __future__ import annotations

from pathlib import Path

import pytest

from eva.tools.acquirer import ToolAcquirer
from eva.tools.builtins import (
    FileHandlerTool,
    PythonExecTool,
    ShellExecTool,
    register_default_tools,
)
from eva.tools.registry import ToolRegistry


def test_file_handler_roundtrip(tmp_path: Path) -> None:
    tool = FileHandlerTool(tmp_path)
    assert "wrote" in tool.execute("WRITE", "a/b.txt", "hello")
    assert "hello" in tool.execute("READ", "a/b.txt")


def test_file_handler_sandbox(tmp_path: Path) -> None:
    tool = FileHandlerTool(tmp_path)
    result = tool.execute("READ", "../escape.txt")
    assert "ERROR" in result


def test_python_exec_runs_simple_code() -> None:
    tool = PythonExecTool(timeout=5.0)
    out = tool.execute("print(21*2)")
    assert "42" in out
    assert "rc=0" in out


def test_python_exec_times_out() -> None:
    tool = PythonExecTool(timeout=0.5)
    out = tool.execute("import time; time.sleep(5)")
    assert "TIMEOUT" in out


def test_shell_exec_allowlist() -> None:
    tool = ShellExecTool(allowlist={"echo"})
    assert "hello" in tool.execute("echo hello")
    assert "not in allowlist" in tool.execute("rm -rf /")


def test_register_default_tools(tmp_path: Path) -> None:
    registry = register_default_tools(
        ToolRegistry(), workspace_root=tmp_path, include_network=False
    )
    names = registry.list_all()
    assert "file_handler" in names
    assert "python_exec" in names
    assert "shell_exec" in names


def test_acquirer_rejects_unknown_package() -> None:
    acq = ToolAcquirer()
    result = acq.acquire("definitely-not-a-real-package-123")
    assert result.installed is False
    assert "allowlist" in result.reason


def test_acquirer_rejects_if_approval_refuses() -> None:
    acq = ToolAcquirer(approval_fn=lambda pkg, reason: False)
    # numpy is on the allowlist, but approval refuses.
    result = acq.acquire("numpy")
    assert result.installed is False
    assert "approval" in result.reason
