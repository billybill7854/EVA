"""Birth — creating a new EVA from scratch or from a parent.

Ron Protocol: children ALWAYS get fresh random weights.
Architecture and genome are inherited; weights and memories are not.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from eva.core.baby_brain import BabyBrain
from eva.core.config import EVAConfig
from eva.identity.lineage import LineageTracker
from eva.reproduction.genome import Genome

logger = logging.getLogger(__name__)


class BirthProcess:
    """Manages the creation of new EVA individuals.

    Ensures Ron Protocol compliance: every new EVA starts with
    fresh random weights, regardless of whether it has a parent.

    Args:
        config: EVA configuration.
        lineage: Lineage tracker for registration.
    """

    def __init__(
        self, config: EVAConfig, lineage: LineageTracker
    ) -> None:
        self._config = config
        self._lineage = lineage

    def create_first_generation(
        self, genome: Optional[Genome] = None
    ) -> dict[str, Any]:
        """Create a first-generation EVA (no parent).

        Args:
            genome: Optional genome. If None, uses defaults.

        Returns:
            Dict with 'brain', 'genome', 'lineage_id' keys.
        """
        if genome is None:
            genome = Genome(generation=1)

        brain = self._create_brain(genome)
        lineage_id = self._lineage.register_birth(
            generation=1,
            genome_hash=genome.hash(),
        )

        logger.info(
            "First generation EVA born: %s (params=%d)",
            lineage_id,
            brain.parameter_count,
        )

        return {
            "brain": brain,
            "genome": genome,
            "lineage_id": lineage_id,
        }

    def create_child(
        self,
        parent_genome: Genome,
        parent_lineage_id: str,
    ) -> dict[str, Any]:
        """Create a child EVA from a parent.

        The child inherits a mutated genome but gets FRESH random
        weights. Memories are NOT inherited. This is the Ron Protocol.

        Args:
            parent_genome: Parent's genome (will be mutated for child).
            parent_lineage_id: Parent's lineage ID.

        Returns:
            Dict with 'brain', 'genome', 'lineage_id' keys.
        """
        # Mutate genome
        child_genome = parent_genome.mutate()

        # Create brain with FRESH random weights
        brain = self._create_brain(child_genome)

        # Register in lineage
        lineage_id = self._lineage.register_birth(
            parent_id=parent_lineage_id,
            generation=child_genome.generation,
            genome_hash=child_genome.hash(),
        )

        logger.info(
            "Child EVA born: %s (gen %d, parent=%s, params=%d)",
            lineage_id,
            child_genome.generation,
            parent_lineage_id,
            brain.parameter_count,
        )

        return {
            "brain": brain,
            "genome": child_genome,
            "lineage_id": lineage_id,
        }

    def _create_brain(self, genome: Genome) -> BabyBrain:
        """Create a fresh BabyBrain from genome parameters.

        ALWAYS randomly initialized. No pretrained weights.
        """
        return BabyBrain(
            vocab_size=getattr(self._config.model, "vocab_size", 512),
            d_model=genome.genes.get("d_model", 768),
            n_layers=genome.genes.get("n_layers", 12),
            n_heads=genome.genes.get("n_heads", 12),
            dtype_str=getattr(self._config.model, "dtype", "float16"),
        )
