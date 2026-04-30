# Tools

EVA autonomously discovers and uses tools, then earns curiosity reward
for first-successful and diverse usage. New tools can be acquired at
runtime from an allow-listed set of PyPI packages, but only with
approval.

## Built-in tools

Registered by `eva.tools.builtins.register_default_tools`:

| Name | Description |
|---|---|
| `file_handler` | Read / write / list files inside EVA's workspace (sandboxed — paths that escape the workspace are rejected). |
| `python_exec` | Runs a short Python snippet in a subprocess with a wall-clock timeout and bounded output. |
| `shell_exec` | Executes a *whitelist* of shell commands (`ls`, `pwd`, `echo`, `cat`, `head`, `tail`, `wc`, `date`, `uname`, `whoami`, `hostname`). |
| `web_search` | Simple DuckDuckGo HTML search (no API key required). |

## Reward shaping

`CuriosityEngine._compute_tool_reward` gives:

* **+0.5** on the first successful use of a tool.
* **+0.2** on successful uses that introduce a new *pattern* (argument string).
* **-0.1** on successful uses once overall diversity drops below 0.5.
* **0.0** on failures or when no tool tracker is attached.

These shape EVA towards *exploring* what tools do rather than
mechanically repeating them.

## Acquisition

`eva.tools.acquirer.ToolAcquirer` can install a package at runtime, but
only if:

1. The package name is in `ALLOWLIST` (a curated list of generally-safe
   utilities — `numpy`, `requests`, `duckduckgo-search`, `sympy`, …).
2. The approval callback returns `True` (wire this to the UI's approval
   dialog or the safety monitor).
3. `pip install` succeeds.

Anything outside the allow-list is refused. The worst failure mode is
*nothing happens*, not *arbitrary code gets installed*.

## Registering a new tool

```python
from eva.tools.base import Tool
from eva.tools.registry import ParameterInfo, ToolInfo, ToolRegistry


class MyCalculator(Tool):
    def get_name(self): return "calculator"
    def is_available(self): return True
    def execute(self, expression: str) -> str:
        import ast, operator
        # minimal safe eval omitted — implement your own
        return f"calc: {expression} = ?"


registry = ToolRegistry()
registry.register(
    MyCalculator(),
    ToolInfo(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        parameters=[ParameterInfo("expression", "str", "Expression", True)],
        examples=["calculator 2*(3+4)"],
    ),
)
```

List everything EVA knows about: `python run.py tools`.
