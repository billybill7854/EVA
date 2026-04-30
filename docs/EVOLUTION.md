# Evolution

EVA has two evolution systems:

## Generational evolution (`eva.reproduction`)

Unchanged from the original design. A child EVA:

* Inherits a **mutated genome** (`Genome.mutate`).
* Gets **fresh random weights** (Ron Protocol).
* Keeps an **ancestor archive**.
* Does not inherit episodic memories.

## Lifetime evolution (`eva.evolution`, new)

An EVA that has already been born can change its own brain *during its
lifetime*. This is the part that makes it feel like a "truly evolving
entity."

### Trigger — curiosity plateau

`Evolver` keeps a sliding window of `plateau_window` curiosity rewards
(default 200). If the mean of the second half differs from the mean of
the first half by less than `plateau_threshold` (default 1e-3), the
evolver treats the signal as plateaued and proposes a change.

### Proposed change, in order of preference

1. **Grow depth** — append one transformer block. New block's residuals
   are near-zero so behaviour is preserved; EVA learns to use the new
   capacity over time.
2. **Grow width** — add one head's worth of `d_model` (keeps the
   divisibility invariant). Existing weights are copied into the
   overlapping slice; the rest is small random noise.
3. **Mutate a hyperparameter** — if growth can't fit under the RAM
   budget, pick one of `lr` or `exploration_temperature` and multiply
   by 0.9 or 1.1 (bounded).

### RAM budget

`Evolver._would_fit(...)` estimates:

```
brain.estimate_memory_bytes(extra_layers=..., new_d_model=...) * 3
```

(×3 approximates parameters + gradients + Adam moments.) If this exceeds
`max_ram_gb`, the growth is refused and a hyperparameter mutation is
tried instead.

### Logging

Every event is broadcast over the UI websocket, recorded in the
`genome_history` table, and surfaced in the Evolution tab with the new
parameter count.

### Demo

```bash
python run.py evolve --steps 3000 --max-ram-gb 8
```

You should see a sequence like:

```
[step 200] grow_depth: +1 layer(s); now 5 -> 214,312 params
[step 405] grow_depth: +1 layer(s); now 6 -> 241,000 params
[step 621] grow_width: d_model -> 198 -> 265,840 params
```

### Design note

We deliberately chose *incremental* growth over per-generation rebirth
because the user asked for a *self-growing* brain, not a population. A
population would still be possible by combining the evolver with
`scripts/reproduce.py`.
