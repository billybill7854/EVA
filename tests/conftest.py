"""Pytest configuration shared across the test suite."""

from __future__ import annotations

import random

import pytest


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    random.seed(0)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except ImportError:  # pragma: no cover
        pass
