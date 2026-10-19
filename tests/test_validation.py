"""Tests for the structure self-check (#7): the physics rules as executable checks.

The checks exist because a structure can be *geometrically* well formed and still be
*physically* wrong: an Adsorbate buried inside the Slab satisfies every schema rule
and is useless. Each check therefore reports itself as a :class:`CheckResult` --
a name, a pass flag and a detail that states the measured quantity -- and the
caller gets a **list**, never a bare boolean.

Every expectation here is a re-measured number (CONVENTIONS.md §G, §J), restated as a
literal so these tests can disagree with the implementation:

- the canonical Pt(111) Slab's top layer is z = 16.560011 A; it is *read* from the
  structure, never hard-coded into the check (§J.6);
- a correct CO/Pt(111) has a closest pair of 1.15034 A -- the intramolecular C-O
  bond -- which is why the threshold is 0.5 A and must not be raised (§G.5);
- ``Structure.distance_matrix`` is minimum-image aware, so two atoms at fractional
  z 0.001 and 0.999 in a c = 3.9231 cell are 0.0078 A apart, not 3.9153 A (§G.1);
- ``Structure.is_valid()`` is a 0.5 A distance check that accepts a zero-volume
  cell, so the lattice check is implemented explicitly (§G.3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab, SlabGenerator

from dsh4sci.adsorption import find_adsorption_sites, place_adsorbate
from dsh4sci.geometry import ADSORBATE_ROLE, SLAB_ROLE, split_roles
from dsh4sci.slabs import generate_slabs
from dsh4sci.validation import CheckResult, check_min_distance

# --- pinned numbers, CONVENTIONS.md §G / §J --------------------------------

A_PT = 3.9231  # conventional cubic Pt lattice parameter, A (§J.4)

CO_CLOSEST_PAIR = 1.15034  # intramolecular C-O bond in ase CO, A (§G.5)
BURIED_ANCHOR_DEPTH = 5.7330  # CO pushed into the Pt(111) Slab, A (§G.2)

THRESHOLD = 0.5  # "closer than 0.5 A fails", A (§G.1)

CUBIC_VOLUME = 60.3793  # a = 3.9231 A cubic cell, A^3 (§G.3)
HEX_VOLUME = 38.9711  # (3, 3, 5, 90, 90, 120) hexagonal cell, A^3
SLAB_VOLUME = 160.3449  # the Pt(111) Slab cell (90, 90, 60), A^3

# The smallest accepted cell volume; below this pymatgen's cell is degenerate.
MIN_CELL_VOLUME = 1e-6


# --- helpers ---------------------------------------------------------------


def cartesian_z(structure: Structure, index: int) -> float:
    """Cartesian z of one site, A."""
    return float(np.asarray(structure.cart_coords, dtype=float)[index, 2])


def slab_top_layer_z(structure: Structure) -> float:
    """Top z-level of the **Slab** part of a role-tagged structure (§J.6)."""
    slab_indices, _ = split_roles(structure)
    return float(np.max(np.asarray(structure.cart_coords)[slab_indices, 2]))


def naive_cartesian_span(structure: Structure, first: int, second: int) -> float:
    """Plain |z1 - z2| with no periodic folding -- what the check must *not* do."""
    return abs(cartesian_z(structure, first) - cartesian_z(structure, second))


# --- fixtures (ticket-local: conftest.py belongs to ticket #2) --------------


@pytest.fixture
def pt_slab(pt_bulk: Structure) -> Slab:
    """The canonical Pt(111) Slab: ``Pt5``, 5 z-levels, exact 15 A Vacuum."""
    return generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=10.0, vacuum=15.0)[0]


# --- minimum interatomic distance over periodic images (§G.1) --------------


def test_two_atoms_closer_than_the_threshold_fail_and_the_report_names_the_pair() -> (
    None
):
    """A deliberate 0.4 A overlap fails, and the detail names both atoms.

    ``[FAIL] minimum_interatomic_distance: sites 0 (Pt) and 1 (C) are 0.4000 A
    apart (<= 0.5 A)`` -- an index pair alone would not tell a chemist which atoms
    collided, so the element symbols are part of the message contract.
    """
    structure = Structure(
        Lattice.cubic(A_PT), ["Pt", "C"], [[0, 0, 0], [0, 0, 0.4 / A_PT]]
    )

    result = check_min_distance(structure)

    assert isinstance(result, CheckResult)
    assert result.name == "minimum_interatomic_distance"
    assert result.passed is False
    assert "0.4000" in result.detail
    assert "sites 0 (Pt)" in result.detail
    assert "sites 1 (C)" in result.detail or "1 (C)" in result.detail
    assert "(<= 0.5 A)" in result.detail


def test_a_collision_through_the_periodic_image_is_detected() -> None:
    """Atoms 0.0078 A apart *across the cell boundary* fail -- the §G.1 trap.

    Fractional z 0.001 and 0.999 are 0.002 of the cell apart through the periodic
    image and 0.998 of it apart in plain Cartesians. The measured pair is
    ``0.0078462`` A, so this test also fails for any implementation that forgets the
    minimum-image convention, while the naive span asserted below stays safely
    above the 0.5 A threshold.
    """
    structure = Structure(
        Lattice.cubic(A_PT), ["Pt", "C"], [[0, 0, 0.001], [0, 0, 0.999]]
    )

    result = check_min_distance(structure)

    assert result.passed is False
    assert "0.0078" in result.detail
    assert "sites 0 (Pt)" in result.detail
    assert "1 (C)" in result.detail
    # The evidence that this is not a Cartesian check: unfolded, the pair is
    # 500x the threshold and would pass.
    assert naive_cartesian_span(structure, 0, 1) > THRESHOLD
    assert naive_cartesian_span(structure, 0, 1) == pytest.approx(
        0.998 * A_PT, abs=1e-9
    )


def test_a_well_separated_pair_passes_and_reports_the_closest_pair() -> None:
    """0.6 A is above the threshold: pass, and say which pair was closest.

    A check that never passes is not a check, so the accepting branch is pinned
    too -- and the detail still names the pair, which is how a caller learns that
    the closest contact in an otherwise fine structure is a short bond.
    """
    structure = Structure(
        Lattice.cubic(2.0), ["Pt", "C"], [[0, 0, 0], [0, 0, 0.3]]
    )

    result = check_min_distance(structure)

    assert result.passed is True
    assert "0.6000" in result.detail
    assert "0 (Pt)" in result.detail
    assert "1 (C)" in result.detail
    assert "(> 0.5 A)" in result.detail


def test_the_threshold_is_inclusive_because_closer_or_equal_fails() -> None:
    """Exactly 0.5 A **fails**: the rule is ``<=``, the safe side of a collision.

    ``Lattice.cubic(2.0)`` with fractional z 0.25 gives a distance that is exactly
    0.5 in binary floating point -- no tolerance, no near-miss.
    """
    structure = Structure(
        Lattice.cubic(2.0), ["Pt", "C"], [[0, 0, 0], [0, 0, 0.25]]
    )

    result = check_min_distance(structure)

    assert result.passed is False
    assert "0.5000" in result.detail
    assert "(<= 0.5 A)" in result.detail


def test_a_structure_with_a_single_site_has_no_pair_and_passes() -> None:
    """One atom cannot collide with itself -- the masked diagonal must not leak.

    An unmasked distance matrix has a zero diagonal, so ``argmin`` would find
    ``0.0`` at ``(0, 0)`` and report a 0 A "collision" for every structure.
    """
    structure = Structure(Lattice.cubic(A_PT), ["Pt"], [[0, 0, 0]])

    result = check_min_distance(structure)

    assert result.passed is True
    assert "only 1 site" in result.detail
