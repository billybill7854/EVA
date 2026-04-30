# Memory

EVA has three layers of memory.

| Layer | Class | Lifetime | Purpose |
|---|---|---|---|
| Working | tensor activations | per-forward-pass | in-the-moment computation |
| Episodic (RAM) | `eva.memory.episodic.EpisodicMemory` | process lifetime | importance-weighted replay / consolidation |
| **Permanent** | `eva.memory.persistent.PersistentMemoryStore` | on disk (SQLite) | survives restarts; powers the UI's Memory tab |

## Tables in `memory.db`

* `episodes` — (step, action, outcome, surprise, emotional_importance,
  source_tag, state_embedding?).
* `insights` — (step, kind, description, confidence, data) — signal
  only; filtered above a confidence threshold by the emergence detector.
* `thoughts` — (step, category, content, context) — raw stream; the UI
  highlights signal vs noise visually.
* `genome_history` — (step, genes, parameter_count) — every growth event.
* `self_modifications` — (step, target, diff, outcome, reason) — audit
  log of every applied change.

## Location

Default: `<repo>/workspace/<name>/memory.db`. Pass `--workspace /some/path`
to `python run.py ui` to change it.

## Recovery

Open the DB with any SQLite tool:

```bash
sqlite3 workspace/default/memory.db
sqlite> SELECT step, content FROM thoughts ORDER BY id DESC LIMIT 20;
```

## Relationship to episodic memory

`EpisodicMemory` remains the *hot* buffer the curiosity engine and
consolidation step operate on. `PersistentMemoryStore` is the *cold*
journal — thoughts and episodes are appended as they happen, and the
UI reads from it to render live panels. The two are intentionally
decoupled so training never blocks on disk I/O.
