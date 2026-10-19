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

# ticket #3 -- shared geometry primitives (Vacuum measurement and the Slab/
# Adsorbate role contract), consumed by tickets #4-#7
from dsh4sci.geometry import (
    ADSORBATE_ROLE,
    SLAB_ROLE,
    count_z_levels,
    group_z_levels,
    set_vacuum,
    split_roles,
    vacuum_gap,
)

# ticket #3 -- slab generation and the Termination fingerprint
from dsh4sci.slabs import (
    TerminationFingerprint,
    generate_slabs,
    termination_fingerprint,
)
