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

# ticket #4 -- selective dynamics: mark the bottom of a Slab as fixed
from dsh4sci.constraints import fix_bottom_atoms

# ticket #5 -- adsorption site enumeration: the site vocabulary, the typed site
# record ticket #6 places an Adsorbate on, and the calibrated "is this a Slab?"
# predicate that rejects bulk crystals
from dsh4sci.adsorption import (
    MIN_VACUUM,
    MIN_Z_LEVELS,
    SITE_TYPES,
    AdsorptionSite,
    SiteType,
    find_adsorption_sites,
    is_slab_like,
)

# ticket #6 -- adsorbate placement: put a molecule on one of those sites, with the
# combined Structure, its Slab atom count (the source of truth across file I/O)
# and the anchor atom's index
from dsh4sci.adsorption import (
    DEFAULT_BINDING_HEIGHT,
    PlacedAdsorbate,
    place_adsorbate,
)
