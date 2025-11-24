"""Utilities for building and simulating doped MoS₂ systems.

The package reorganises the loose scripts from the accompanying
preprint into reusable modules for structure generation, DFT/MLIP
energy evaluation, MD workflows, and post-processing.
"""

__all__ = [
    "DEFAULT_DOPANTS",
    "STRUCTURE_TRANSFORM",
]

__version__ = "0.1.0"

from .structures import DEFAULT_DOPANTS, STRUCTURE_TRANSFORM  # noqa: E402
