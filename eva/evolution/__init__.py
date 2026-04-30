"""Evolution — lifetime + generational change.

Key distinction from ``eva/reproduction/``:

* ``eva/reproduction/`` handles the *generational* Ron Protocol —
  children get mutated genomes but fresh random weights.
* ``eva/evolution/`` handles *lifetime* evolution — an EVA that has
  already been born can grow its own brain, mutate its own
  hyperparameters, and log each change into ``lineage/``.
"""

from eva.evolution.evolver import ArchitectureGenome, EvolutionEvent, Evolver

__all__ = ["ArchitectureGenome", "Evolver", "EvolutionEvent"]
