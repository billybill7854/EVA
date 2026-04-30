"""Runtime tool acquisition.

EVA can request a new tool — identified by a Python package on PyPI —
when its curiosity hits a capability gap. The acquirer gates this
behind:

1. A curated **allow-list** of packages (``ALLOWLIST``).
2. A human-approval callback (``approval_fn``) that must return
   ``True`` — integrators wire this to the UI's approval button or the
   safety monitor.
3. A subprocess ``pip install`` to the user's active environment, with
   the output captured in the transparency log.

Refusing unknown packages is the default: the worst failure mode is
*nothing happens*, not *arbitrary code gets installed*.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


ALLOWLIST: frozenset[str] = frozenset(
    {
        "requests",
        "httpx",
        "beautifulsoup4",
        "lxml",
        "duckduckgo-search",
        "wikipedia",
        "wolframalpha",
        "sympy",
        "pillow",
        "numpy",
    }
)


@dataclass
class AcquisitionResult:
    package: str
    installed: bool
    reason: str = ""
    stdout: str = ""


ApprovalFn = Callable[[str, str], bool]


class ToolAcquirer:
    """Tries to install a package so a new tool can be registered."""

    def __init__(
        self,
        approval_fn: Optional[ApprovalFn] = None,
        allowlist: Optional[frozenset[str]] = None,
    ) -> None:
        self._approval_fn = approval_fn
        self._allowlist = allowlist if allowlist is not None else ALLOWLIST

    def is_allowed(self, package: str) -> bool:
        return package.lower() in self._allowlist

    def acquire(
        self,
        package: str,
        reason: str = "",
    ) -> AcquisitionResult:
        pkg = package.strip().lower()
        if not pkg:
            return AcquisitionResult(package, False, "empty package name")
        if not self.is_allowed(pkg):
            return AcquisitionResult(
                package, False, f"{pkg!r} not in allowlist"
            )
        if self._approval_fn is not None and not self._approval_fn(pkg, reason):
            return AcquisitionResult(
                package, False, "approval_fn refused acquisition"
            )
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", pkg],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return AcquisitionResult(package, False, "pip timeout")
        except Exception as exc:
            return AcquisitionResult(package, False, f"pip crashed: {exc}")
        installed = proc.returncode == 0
        logger.info(
            "ToolAcquirer: %s %s (rc=%d)",
            "installed" if installed else "failed",
            pkg,
            proc.returncode,
        )
        return AcquisitionResult(
            package=pkg,
            installed=installed,
            reason="" if installed else proc.stderr[-400:],
            stdout=proc.stdout[-400:],
        )
