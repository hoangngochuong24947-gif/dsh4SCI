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
import warnings

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab, SlabGenerator

from dsh4sci.adsorption import find_adsorption_sites, place_adsorbate
from dsh4sci.geometry import ADSORBATE_ROLE, SLAB_ROLE, split_roles
from dsh4sci.slabs import generate_slabs
from dsh4sci.errors import InvalidParameterError
from dsh4sci.validation import (
    CheckResult,
    check_adsorbate_anchor_above_surface,
    check_lattice_angles,
    check_min_distance,
)

# --- pinned numbers, CONVENTIONS.md §G / §J --------------------------------

A_PT = 3.9231  # conventional cubic Pt lattice parameter, A (§J.4)

CO_CLOSEST_PAIR = 1.15034  # intramolecular C-O bond in ase CO, A (§G.5)
BURIED_ANCHOR_Z = 10.8270  # anchor z after CO is pushed into Pt(111), A (§G.2)
BURIED_ANCHOR_DEPTH = 5.7330  # ... which is this far below the top layer, A

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


def cautionary_lattice(*parameters: float) -> Lattice:
    """A lattice pymatgen warns about while measuring it.

    Building a collapsed cell divides by zero inside pymatgen; that warning is not
    the subject of these tests, so it is absorbed at construction. The check's own
    suppression is proven separately, with warnings promoted to errors.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return Lattice.from_parameters(*parameters)


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


# --- lattice-angle validity (§G.3) -----------------------------------------
#
# ``Structure.is_valid()`` is deliberately *not* used: it is a 0.5 A distance check
# that returns True for a zero-volume cell, and ``Lattice`` has no ``is_valid`` at
# all. These tests pin the explicit rule and the evidence for it.


@pytest.mark.parametrize(
    ("lattice", "volume"),
    [
        pytest.param(Lattice.cubic(A_PT), CUBIC_VOLUME, id="cubic-90-90-90"),
        pytest.param(
            Lattice.from_parameters(3, 3, 5, 90, 90, 120),
            HEX_VOLUME,
            id="hexagonal-90-90-120",
        ),
        pytest.param(
            Lattice.from_parameters(3, 3, 5, 90, 90, 60),
            38.9711,
            id="slab-like-90-90-60",
        ),
    ],
)
def test_a_real_cell_passes_the_lattice_angle_check(
    lattice: Lattice, volume: float
) -> None:
    """A cubic cell passes -- and so do 120 and 60 degree cells.

    Hexagonal (90, 90, 120) and the Pt(111) Slab's own (90, 90, 60) are normal
    surface cells; tightening the rule to "near 90 degrees" would reject every
    hexagonal surface in the library.
    """
    structure = Structure(lattice, ["Pt"], [[0, 0, 0]])

    result = check_lattice_angles(structure)

    assert result.name == "lattice_angles"
    assert result.passed is True
    assert f"volume={volume:.4f}" in result.detail
    assert result.detail.startswith("angles=(")


def test_the_canonical_pt111_slab_cell_passes(pt_slab: Slab) -> None:
    """The real Slab's cell -- angles (90, 90, 60), 160.3449 A^3 -- must pass.

    A structure the library itself generates must never fail its own self-check.
    """
    result = check_lattice_angles(pt_slab)

    assert result.passed is True
    assert "volume=160.3449" in result.detail
    assert "60.0" in result.detail


def test_a_badly_skewed_cell_fails_even_though_its_angles_look_plausible() -> None:
    """gamma = 177 degrees narrows the cell to ~0 volume while staying under 175.

    This cell is inside any sane angle range, so only the volume test catches it --
    which is exactly why the rule has two conditions.
    """
    structure = Structure(cautionary_lattice(3, 3, 3, 89, 88, 177), ["Pt"], [[0, 0, 0]])

    result = check_lattice_angles(structure)

    assert result.passed is False
    assert "invalid cell" in result.detail
    assert "[5.0, 175.0]" in result.detail
    assert float(structure.lattice.volume) <= MIN_CELL_VOLUME


def test_structure_is_valid_is_the_wrong_tool_for_a_degenerate_cell() -> None:
    """``is_valid()`` says True and ``Lattice`` has no ``is_valid`` at all.

    This is the D9 evidence, pinned so nobody "simplifies" the check back to
    ``structure.is_valid()``: a zero-volume cell would pass.
    """
    structure = Structure(cautionary_lattice(3, 3, 3, 89, 88, 177), ["Pt"], [[0, 0, 0]])

    assert structure.is_valid() is True
    assert structure.volume <= MIN_CELL_VOLUME
    assert not hasattr(Lattice, "is_valid")
    assert check_lattice_angles(structure).passed is False


def test_a_collapsed_cell_with_non_finite_angles_fails() -> None:
    """alpha = 0 degrees is not a lattice: the angles come back ``nan``.

    ``nan`` fails every range comparison, so the angle condition alone must reject
    it -- a bare ``min(angles) < 5`` would silently accept it.
    """
    structure = Structure(cautionary_lattice(3, 3, 3, 0, 90, 90), ["Pt"], [[0, 0, 0]])

    assert any(math.isnan(angle) for angle in structure.lattice.angles)

    result = check_lattice_angles(structure)

    assert result.passed is False
    assert "invalid cell" in result.detail
    assert "nan" in result.detail


def test_the_check_silences_pymatgen_warnings_for_a_degenerate_cell() -> None:
    """pymatgen divides by zero to measure a collapsed cell; the check absorbs it.

    Run with warnings promoted to errors, so an unsuppressed ``RuntimeWarning``
    from inside the check fails this test instead of polluting the test output.
    """
    skewed = Structure(cautionary_lattice(3, 3, 3, 89, 88, 177), ["Pt"], [[0, 0, 0]])
    degenerate = Structure(cautionary_lattice(3, 3, 3, 0, 90, 90), ["Pt"], [[0, 0, 0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_lattice_angles(skewed).passed is False
        assert check_lattice_angles(degenerate).passed is False


# --- Adsorbate anchor above the Slab surface (§G.2) -------------------------
#
# The motivating failure: an Adsorbate buried *inside* the Slab satisfies every
# schema rule and is useless. Detecting it needs the shared Slab/Adsorbate split
# (§G.4) -- a species test cannot separate O-adsorbate from O-slab, and a z test
# would classify the buried atom as part of the Slab.


def flat_cell(anchor_frac_z: float) -> Structure:
    """A one-atom Slab whose top layer is exactly ``c / 2`` = 1.96155 A.

    The Adsorbate atom is offset in x so the two sites never coincide, which keeps
    the distance check valid and lets a test use this cell for either check.
    ``3.9231 / 2`` renders as ``1.9615`` at 4 decimals -- the float is just under
    the decimal value (§G.2 writes 1.96155).
    """
    structure = Structure(
        Lattice.cubic(A_PT), ["Pt", "C"], [[0, 0, 0.5], [0.3, 0, anchor_frac_z]]
    )
    structure.add_site_property("role", [SLAB_ROLE, ADSORBATE_ROLE])
    return structure


def bury_adsorbate(structure: Structure, anchor_index: int, target_z: float) -> None:
    """Translate every Adsorbate-tagged site down so the anchor lands at ``target_z``.

    The Slab is untouched: this is the plumbing of an Adsorbate placed on the wrong
    part of the surface, not a broken cell.
    """
    _, adsorbate_indices = split_roles(structure)
    shift = target_z - cartesian_z(structure, anchor_index)
    structure.translate_sites(adsorbate_indices, [0, 0, shift], frac_coords=False)


def test_an_anchor_exactly_at_the_slab_top_layer_passes() -> None:
    """The boundary is inclusive: on the surface is on the surface, not inside it.

    ``structure[1]`` sits at the same z as the top layer (fractional z 0.50 in a
    cell whose top layer is c/2), which is the boundary an Adsorbate bound on an
    ``ontop`` site with ``distance = 0`` occupies.
    """
    structure = flat_cell(0.50)

    result = check_adsorbate_anchor_above_surface(structure, anchor_index=1)

    assert result.name == "adsorbate_anchor_above_surface"
    assert result.passed is True
    assert "anchor at z=1.9615 A" in result.detail
    assert "slab top z=1.9615 A" in result.detail


@pytest.mark.parametrize(
    ("anchor_frac_z", "depth"),
    [
        pytest.param(0.45, 0.1962, id="just-below-0.1962A"),
        pytest.param(0.25, 0.9808, id="buried-0.9808A"),
    ],
)
def test_an_anchor_below_the_slab_top_layer_fails_and_states_the_depth(
    anchor_frac_z: float, depth: float
) -> None:
    """Slightly below fails, and the detail says how far below.

    §G.2 verified this boundary on a cell whose top layer is 1.96155 A: fractional
    z 0.45 is 0.1962 A under it and fractional z 0.25 is 0.9808 A under it. Both
    must fail -- the check is not a fudge band.
    """
    structure = flat_cell(anchor_frac_z)

    result = check_adsorbate_anchor_above_surface(structure, anchor_index=1)

    assert result.passed is False
    assert f"{depth:.4f} A BELOW" in result.detail
    assert "slab top layer z=1.9615 A" in result.detail
    assert "anchor site 1 (C)" in result.detail


def test_a_correct_co_on_pt111_has_its_anchor_above_the_surface(
    pt_slab: Slab,
) -> None:
    """CO placed by ``place_adsorbate`` binds 1.85 A above the top layer: pass.

    The anchor is the carbon, at z = 16.560011 + 1.85 = 18.410011 A, above the
    slab top layer read from the structure rather than hard-coded (§J.6).
    """
    ontop = next(
        site for site in find_adsorption_sites(pt_slab) if site.site_type == "ontop"
    )
    placed = place_adsorbate(pt_slab, ontop, molecule="CO", anchor="C")

    result = check_adsorbate_anchor_above_surface(
        placed.structure, anchor_index=placed.anchor_index
    )

    assert result.passed is True
    assert f"anchor at z={cartesian_z(placed.structure, placed.anchor_index):.4f}" in (
        result.detail
    )
    assert f"slab top z={slab_top_layer_z(placed.structure):.4f}" in result.detail


def test_co_pushed_inside_the_pt111_slab_is_caught_and_the_depth_is_reported(
    pt_slab: Slab,
) -> None:
    """The motivating failure: a buried Adsorbate in an otherwise valid structure.

    The Slab is untouched and its cell is fine, so ``minimum_interatomic_distance``
    and ``lattice_angles`` both still pass -- only the anchor check can see this.
    Measured (§G.2): the anchor ends up 5.7330 A below the top layer, still *inside*
    the stack (above the bottom layer), which is exactly why a z threshold could
    never classify it as an Adsorbate.
    """
    ontop = next(
        site for site in find_adsorption_sites(pt_slab) if site.site_type == "ontop"
    )
    placed = place_adsorbate(pt_slab, ontop, molecule="CO", anchor="C")
    structure = placed.structure.copy()
    bury_adsorbate(structure, placed.anchor_index, BURIED_ANCHOR_Z)

    result = check_adsorbate_anchor_above_surface(
        structure, anchor_index=placed.anchor_index
    )

    top = slab_top_layer_z(structure)
    slab_indices, _ = split_roles(structure)
    bottom = float(np.min(np.asarray(structure.cart_coords)[slab_indices, 2]))
    anchor_z = cartesian_z(structure, placed.anchor_index)

    assert result.passed is False
    assert result.name == "adsorbate_anchor_above_surface"
    # Buried *inside* the Slab, not merely below the whole cell.
    assert bottom < anchor_z < top
    assert anchor_z == pytest.approx(BURIED_ANCHOR_Z, abs=1e-9)
    assert f"{top - anchor_z:.4f} A BELOW" in result.detail
    assert f"{top - anchor_z:.4f}" == "5.7330"
    # The rest of the self-check cannot see this structure as broken.
    assert check_min_distance(structure).passed is True
    assert check_lattice_angles(structure).passed is True


def test_an_anchor_index_on_a_slab_atom_is_rejected() -> None:
    """An anchor that the ``role`` tag calls Slab is a caller error, not a verdict.

    Silently reporting "the anchor is below the top layer" for a Slab atom would
    make a wrong index look like a physics failure.
    """
    structure = flat_cell(0.50)

    with pytest.raises(InvalidParameterError) as excinfo:
        check_adsorbate_anchor_above_surface(structure, anchor_index=0)

    message = str(excinfo.value)
    assert "anchor_index=0" in message
    assert len(message) <= 210


def test_a_non_integer_anchor_index_is_rejected() -> None:
    """A float index is a caller error, not silently truncated to site 1."""
    structure = flat_cell(0.50)

    with pytest.raises(InvalidParameterError) as excinfo:
        check_adsorbate_anchor_above_surface(structure, anchor_index=1.0)

    message = str(excinfo.value)
    assert "anchor_index=1.0" in message
    assert len(message) <= 210


def test_a_structure_with_no_slab_sites_is_rejected() -> None:
    """Without a Slab there is no top layer, so the check cannot answer.

    Reporting a pass would be a false negative on a structure whose ``role`` tags
    are simply wrong.
    """
    structure = flat_cell(0.50)
    structure.add_site_property("role", [ADSORBATE_ROLE, ADSORBATE_ROLE])

    with pytest.raises(InvalidParameterError) as excinfo:
        check_adsorbate_anchor_above_surface(structure, anchor_index=1)

    message = str(excinfo.value)
    assert "no site is tagged 'slab'" in message
    assert len(message) <= 210


def test_an_anchor_index_outside_the_structure_is_rejected() -> None:
    """Index 7 in a 2-site structure must be an ``InvalidParameterError``.

    Without it the caller gets an ``IndexError`` from the coordinate array, or
    worse, a verdict about the *last* site if Python's negative indexing is used.
    """
    structure = flat_cell(0.50)

    for bad_index in (2, 7, -1):
        with pytest.raises(InvalidParameterError) as excinfo:
            check_adsorbate_anchor_above_surface(structure, anchor_index=bad_index)
        message = str(excinfo.value)
        assert f"anchor_index={bad_index}" in message
        assert len(message) <= 210
