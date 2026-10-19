"""dsh4SCI: rules for building and judging atomistic structures.

A library of materials-science domain logic — surfaces, slabs, adsorbates — as
pure functions over ``pymatgen`` structures. See ``CONTEXT.md`` for vocabulary
and ``docs/adr/`` for the founding decisions.
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- public API re-exports: one block per module, added by later tickets ---

# ticket #2 -- structure loading (path in, Structure out)
from dsh4sci.loading import SUPPORTED_SUFFIXES, load_structure
