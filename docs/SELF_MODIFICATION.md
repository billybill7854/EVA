# Self-modification

> **Q:** "Does EVA have permanent memory and can it modify its own code?"
>
> **A (before this PR):** *No on both counts.* Memory was a RAM-only
> circular buffer, and there was no code path for EVA to change its
> own source.
>
> **A (after this PR):** *Yes, within rails.* Permanent memory lives in
> SQLite under `workspace/<name>/memory.db`. Self-modification is
> possible but gated.

## Rails

`eva.autonomy.self_modifier.SelfModifier` enforces the following
invariants on every proposed change:

1. **Allow-list of writable paths** (see `ALLOWED_TARGETS`):
   * `configs/*.yaml`
   * `workspace/**/*.yaml`
   * `workspace/**/*.md`
   * `eva/tools/heuristics/**.py`
   * `eva/prompts/**.(md|txt|yaml)`
2. **Forbid-list of paths EVA may never touch**:
   * `eva/guidance/covenant.py`
   * `eva/transparency/safety_monitor.py`
   * `eva/autonomy/self_modifier.py` (can't rewrite its own rules)
3. **Size cap** — `MAX_FILE_BYTES = 128 KiB`.
4. **Syntax check** — `.py` via `ast.parse`, `.yaml` via `yaml.safe_load`.
5. **Approval callback** — if configured, the callback (hooked up to the
   UI's approval button and the `SafetyMonitor`) must return `True`.
6. **Audit trail** — every applied diff is written to
   `workspace/<name>/self_mods/<path>.patch` and logged to the persistent
   memory store's `self_modifications` table.

## Example

```python
from pathlib import Path
from eva.autonomy.self_modifier import Proposal, SelfModifier

sm = SelfModifier(
    repo_root=Path.cwd(),
    approval_fn=lambda proposal: True,   # wire this to your UI
)

record = sm.evaluate(
    Proposal(
        target="configs/custom.yaml",
        new_content="model:\n  d_model: 256\n  n_layers: 6\n",
        reason="curiosity plateau — try a wider brain next restart",
    )
)
print(record.applied, record.rejection_reason)
```

## Why not `exec`?

Because we want every change to be **reviewable, reversible, and
constrained**. `exec` is none of those. Diffs are the minimal thing that
gives EVA real agency without giving it root on its own codebase.

## Rolling back

Every applied diff is archived. Git gives you the final word:

```bash
git diff               # see EVA's pending self-mods
git checkout -- .      # roll everything back
```
