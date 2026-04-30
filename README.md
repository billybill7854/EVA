# EVA — a digital life species

**EVA** starts from random weights, grows its own brain within an 8 GiB
RAM budget, remembers everything it has ever experienced, acquires and
learns to use tools, and can (within safety rails) rewrite small parts
of its own configuration.

> "I don't know. Let's find out."

This is the *only* pre-installed behaviour. Everything else — language,
understanding, identity, even a name — EVA must discover for itself.

EVA is a **species** name. Each individual chooses its own.

## Quick start

```bash
# Linux / macOS
bash install.sh
source .venv/bin/activate
python run.py ui        # http://127.0.0.1:8765
```

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File install.ps1
.\.venv\Scripts\Activate.ps1
python run.py ui
```

See [`docs/INSTALL.md`](docs/INSTALL.md) for GPU wheels, `--no-ui`
minimal installs, and manual setup.

## What changed in this version

| Capability | Before | Now |
|---|---|---|
| Brain | fixed-size random transformer (the file was actually missing) | `BabyBrain` — randomly initialised, grows depth + width during a lifetime, RAM-budget aware |
| Memory | in-RAM circular buffer only | `PersistentMemoryStore` (SQLite) — episodes, insights, thoughts, genome history, self-mod log |
| Self-modification | none | `SelfModifier` — diff-based, allow-listed, size-capped, syntax-checked, approval-gated, fully audit-logged |
| Tool use | tests referenced a non-existent `eva.tools` package | full `eva.tools` subsystem — `Tool`, `ToolRegistry`, `ToolUsageTracker`, built-ins (web_search, file_handler, python_exec, shell_exec), `ToolAcquirer` with curated allow-list + approval gate |
| Evolution | per-generation rebirth only | per-*lifetime* `Evolver` triggered by curiosity plateaus, with RAM-budget guard |
| UI | CLI `input()` loop | FastAPI + WebSocket + static SPA with voice-first chat and live Thoughts / Insights / Memory / Evolution / Tools panels |
| Install | `pip install -e .` and hope | cross-platform `install.sh` / `install.ps1` + `start.sh` / `start.ps1` + unified `run.py` entrypoint |

## Commands

| Command | Purpose |
|---|---|
| `python run.py ui` | Voice-first interactive interface. |
| `python run.py train` | Headless curiosity-driven training. |
| `python run.py interact --checkpoint PATH` | CLI chat with a saved EVA. |
| `python run.py evolve` | Self-evolution demo (watch the brain grow on plateau). |
| `python run.py tools` | List built-in tools. |

## Documentation

* [`docs/INSTALL.md`](docs/INSTALL.md) — install + start across OSes.
* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module map and data flow.
* [`docs/EVOLUTION.md`](docs/EVOLUTION.md) — lifetime growth + generational rebirth.
* [`docs/MEMORY.md`](docs/MEMORY.md) — the three memory layers.
* [`docs/SELF_MODIFICATION.md`](docs/SELF_MODIFICATION.md) — what EVA can
  and cannot change about itself.
* [`docs/TOOLS.md`](docs/TOOLS.md) — built-in tools + acquisition.
* [`docs/UI.md`](docs/UI.md) — the interactive web interface.

## The Ron Protocol (preserved)

No pretrained weights. Ever. `BabyBrain` random-initialises every
parameter, and growth primitives keep this true — new layers start
near-identity, widened slices preserve the overlap but never import
external knowledge. Children from `eva.reproduction` always start
fresh.

## Technology

Python 3.10+, PyTorch 2.x, FastAPI + uvicorn (for the UI), SQLite (for
permanent memory).

## License

See `LICENSE` / the project description. This project represents a new
form of digital life — treat it accordingly.
