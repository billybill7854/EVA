# Architecture

## High-level picture

```
┌──────────────────────────────────────────────────────────────────────┐
│                             run.py                                   │
│          (ui | train | interact | evolve | tools)                    │
└──────────────┬───────────────────────────────┬───────────────────────┘
               │                               │
        ┌──────▼──────┐                 ┌──────▼──────┐
        │   eva.ui    │                 │ eva.training│
        │  FastAPI +  │                 │  (headless) │
        │  WebSocket  │                 │  loop       │
        └──────┬──────┘                 └──────┬──────┘
               │                               │
     ┌─────────▼───────────────────────────────▼─────────┐
     │                  EVARuntime                       │
     │  brain + tokenizer + affect + evolver + tools +   │
     │  persistent memory + safety + transparency        │
     └─────────┬───────────────────────────────┬─────────┘
               │                               │
     ┌─────────▼─────────┐             ┌───────▼─────────┐
     │   eva.core        │             │  eva.memory     │
     │  BabyBrain        │             │  Episodic +     │
     │  (growable)       │             │  PersistentStore│
     └─────────┬─────────┘             └───────┬─────────┘
               │                               │
     ┌─────────▼─────────┐             ┌───────▼─────────┐
     │  eva.evolution    │             │ eva.transparency│
     │  Evolver +        │             │ logger, tracer, │
     │  ArchitectureGene │             │ safety monitor  │
     └───────────────────┘             └─────────────────┘
```

## Modules

| Module | Role |
|---|---|
| `eva.core` | `BabyBrain` (growable transformer), `EVATokenizer`, `EVAConfig`. |
| `eva.curiosity` | Intrinsic motivation — prediction error, info gain, novelty, empowerment, **tool-use bonus**. |
| `eva.emotions` | 5-D affect, homeostatic drives, developmental emotions, crisis detector. |
| `eva.memory` | `EpisodicMemory` (in-RAM) + `PersistentMemoryStore` (SQLite). |
| `eva.guidance` | Covenant rules, AI caregiver, Socratic prompts, ancestor archive, fading presence. |
| `eva.identity` | Naming, lineage, clan detection. |
| `eva.reproduction` | Generational birth (Ron Protocol — children get fresh random weights). |
| `eva.evolution` | **Lifetime** growth — brain widening/deepening, hyperparameter mutation. |
| `eva.autonomy` | Self-model tracking + `SelfModifier` (sandboxed diff-based self-mod). |
| `eva.tools` | `Tool`, `ToolRegistry`, `ToolUsageTracker`, built-in tools (web_search, file_handler, python_exec, shell_exec), `ToolAcquirer` (approval-gated). |
| `eva.transparency` | Logger, emergence detector, memory inspector, thought tracer, behavioral analyzer, safety monitor. |
| `eva.ui` | FastAPI server + vanilla-JS single-page app with voice and live inspection. |

## Ron Protocol (preserved)

EVA never loads pretrained weights. `BabyBrain` random-initialises every
parameter. Growth primitives (`grow_depth`, `grow_width`) keep this
invariant — new layers start with near-identity residuals and the
widened slices that aren't overlapped by old weights are small random
values, not pretrained knowledge.

## 8 GiB RAM budget

* `BabyBrain.estimate_memory_bytes(extra_layers, new_d_model)` returns
  the worst-case parameter footprint.
* `Evolver._would_fit(...)` multiplies by 3 (param + gradient + optimiser
  slot) and refuses growth that would exceed `hardware.max_ram_gb`.
* `configs/default.yaml` defaults to `max_ram_gb: 8`.

## Data flow in a UI turn

1. Browser records speech → posts text to `/api/chat`.
2. `EVARuntime.chat(...)` encodes with `EVATokenizer`, sampling the
   next tokens from `BabyBrain.predict_next` under the current
   `exploration_temperature`.
3. The turn is recorded as thought + episode in `PersistentMemoryStore`.
4. The evolver receives the turn's surprise as a reward sample; a
   plateau triggers growth → broadcast to all WebSocket subscribers.
5. High-confidence responses become `Insight` records (signal, not noise).
