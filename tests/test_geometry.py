"""Tests for the shared geometry primitives (``dsh4sci.geometry``).

These primitives are a contract for the later tickets: ``vacuum_gap`` is the
Vacuum definition quoted in CONVENTIONS.md §C.3, ``set_vacuum`` is the only
sanctioned way to make the Vacuum exact, ``count_z_levels`` is the layer-grouping
rule behind both the Termination fingerprint (§C.5) and the "is this slab-like?"
predicate (§E.5), and ``split_roles`` is the shared Slab/Adsorbate decision
(§G.4).

Every expected number here comes from CONVENTIONS.md §C.2/§C.3/§I, which records
values measured against the pinned stack. They are pinned as literals so the
test can disagree with the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab, SlabGenerator

from dsh4sci.errors import InvalidParameterError
from dsh4sci.geometry import (
    ADSORBATE_ROLE,
    SLAB_ROLE,
    count_z_levels,
    group_z_levels,
    set_vacuum,
    split_roles,
    vacuum_gap,
)

# --- pinned numbers, CONVENTIONS.md §C.2/§C.3/§I -----------------------------

PT111_D111 = 2.265003  # a / sqrt(3) for fcc Pt, a = 3.9231 A
PT111_AREA = 6.664374  # in-plane area of the primitive Pt(111) cell, A^2
ORTHO_C = 27.18003409269386  # Pt(111) c after get_orthogonal_c_slab(), A
ORTHO_GAP = 18.12002272846258  # its measured largest periodic gap, A
SLAB_THICKNESS = ORTHO_C - ORTHO_GAP  # cartesian extent of the atomic stack


def _straddle(slab: Slab, frac_z: float = 0.35) -> Slab:
    """Same slab translated along c so the atomic stack crosses the periodic boundary.

    ``frac_z`` is added to every fractional z and **wrapped into [0, 1)**, which is
    exactly what a lattice translation does. The atomic stack is unchanged as a
    physical object, so every Vacuum definition must report the same value.
    """
    coords = np.array(slab.frac_coords, dtype=float).copy()
    coords[:, 2] = (coords[:, 2] + frac_z) % 1.0
    return Slab(
        lattice=Lattice(np.array(slab.lattice.matrix, dtype=float)),
        species=slab.species_and_occu,
        coords=coords,
        miller_index=slab.miller_index,
        oriented_unit_cell=slab.oriented_unit_cell,
        shift=slab.shift,
        scale_factor=slab.scale_factor,
        reorient_lattice=slab.reorient_lattice,
        site_properties=slab.site_properties,
    )


@pytest.fixture
def ortho_slab(pt_bulk: Structure) -> Slab:
    """The raw Pt(111) slab with an orthogonal c axis, before any Vacuum rescale.

    ``pt_bulk`` is the function-scoped conventional-cell fixture from conftest.py;
    it is copied because ``SlabGenerator`` mutates its input (§C.1).
    """
    raw = SlabGenerator(
        pt_bulk.copy(),
        (1, 1, 1),
        10.0,
        15.0,
        primitive=True,
        reorient_lattice=True,
        center_slab=True,
        lll_reduce=False,
        max_normal_search=None,
    ).get_slabs()
    return raw[0].get_orthogonal_c_slab()


@pytest.fixture
def bulk_pt(pt_bulk: Structure) -> Structure:
    """The conventional fcc Pt bulk the slabs are cut from."""
    return pt_bulk


# --- vacuum_gap: the largest periodic gap, wrap-around included -------------


def test_vacuum_gap_matches_the_measured_orthogonal_slab(ortho_slab: Slab) -> None:
    """The orthogonal Pt(111) slab's gap is the §C.3 measured 18.120023 A."""
    assert float(ortho_slab.lattice.c) == pytest.approx(ORTHO_C, abs=1e-9)
    assert vacuum_gap(ortho_slab) == pytest.approx(ORTHO_GAP, abs=1e-9)


def test_vacuum_gap_uses_the_wraparound_gap_not_the_cartesian_extent(
    ortho_slab: Slab,
) -> None:
    """A slab translated across the periodic boundary keeps its Vacuum.

    This is the §C.3 counter-example. The two obvious definitions --
    ``c - (z.max() - z.min())`` and ``(1 - ptp(frac_z)) * c`` -- both silently
    assume the stack does not straddle z = 0. Translating the very same slab makes
    them report the *interlayer spacing* (2.265 A) instead of the Vacuum, while the
    periodic-largest-gap definition is invariant.
    """
    straddled = _straddle(ortho_slab, frac_z=0.35)
    cart_z = np.asarray(straddled.cart_coords, dtype=float)[:, 2]
    frac_z = np.asarray(straddled.frac_coords, dtype=float)[:, 2]
    c = float(straddled.lattice.c)

    # The stack really does straddle the boundary: one atom sits near z = 0 while
    # the rest of it is spread over the top of the cell.
    assert cart_z.min() < 0.5
    assert cart_z.max() > 0.5 * c

    # The physics is invariant: same Vacuum as the untranslated slab.
    assert vacuum_gap(straddled) == pytest.approx(vacuum_gap(ortho_slab), abs=1e-9)
    assert vacuum_gap(straddled) == pytest.approx(ORTHO_GAP, abs=1e-9)

    # Both wrong definitions collapse to the interlayer spacing instead.
    wrong_extent = c - (cart_z.max() - cart_z.min())
    wrong_frac = (1.0 - float(np.ptp(frac_z))) * c
    assert wrong_extent == pytest.approx(PT111_D111, abs=1e-5)
    assert wrong_frac == pytest.approx(PT111_D111, abs=1e-5)
    assert abs(vacuum_gap(straddled) - wrong_extent) > 15.0


def test_vacuum_gap_of_a_bulk_crystal_is_the_smallest_periodic_repeat(
    bulk_pt: Structure,
) -> None:
    """For a bulk crystal with no Vacuum the largest gap is just the layer repeat.

    §E.5 measured 1.962 A for conventional fcc Pt (c = 3.9231 A, two z-levels).
    The value is not zero -- which is precisely why an "is it a slab?" test needs a
    threshold (5.0 A) rather than an equality check.
    """
    assert vacuum_gap(bulk_pt) == pytest.approx(1.96155, abs=1e-5)


def test_vacuum_gap_reads_the_cartesian_z_not_the_fractional_z() -> None:
    """Fractional z is not a coordinate this library may rely on.

    pymatgen's slab operations routinely leave fractional coordinates outside
    ``[0, 1)``; the periodic gap then has to be measured on Cartesian z, where
    "the same atom wrapped back into the cell" is the same number. Here both atoms
    are reported at fractional z outside the cell but are two A apart in z.
    """
    lattice = Lattice.from_parameters(3.0, 3.0, 20.0, 90.0, 90.0, 90.0)
    # frac z 1.0 and 1.1 -> cartesian z 20.0 and 22.0 (i.e. 0.0 and 2.0 modulo c).
    structure = Structure(lattice, ["Pt", "Pt"], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.1]])
    assert vacuum_gap(structure) == pytest.approx(18.0, abs=1e-9)


# --- set_vacuum: exact, symmetric, and a no-op on the atomic stack ----------


def test_set_vacuum_makes_the_gap_exactly_the_requested_value(
    ortho_slab: Slab,
) -> None:
    """The §C.3 corrected result for Pt(111): 15.000000000 A to ~1e-15.

    The request is only satisfied to three decimals by the generator (18.120 A
    after orthogonalisation), so this is the step that earns the ticket's
    ``+/- 0.01 A`` acceptance criterion.
    """
    rescaled = set_vacuum(ortho_slab, 15.0)

    assert vacuum_gap(rescaled) == pytest.approx(15.0, abs=1e-12)
    # The tight bound actually reached, measured as 3.55e-15 in §C.3.
    assert abs(vacuum_gap(rescaled) - 15.0) < 1e-13
    assert float(rescaled.lattice.c) == pytest.approx(24.060011364, abs=1e-9)


def test_set_vacuum_splits_the_vacuum_equally_above_and_below(
    ortho_slab: Slab,
) -> None:
    """The stack is re-centred, not merely rescaled (§C.3).

    Preserving Cartesian z without re-centring gives the right total gap but a
    5.94 / 9.06 split instead of 7.5 / 7.5 -- see the next test.
    """
    rescaled = set_vacuum(ortho_slab, 15.0)
    cart_z = np.asarray(rescaled.cart_coords, dtype=float)[:, 2]
    c = float(rescaled.lattice.c)

    above = c - float(cart_z.max())
    below = float(cart_z.min())
    assert above == pytest.approx(7.5, abs=1e-9)
    assert below == pytest.approx(7.5, abs=1e-9)
    assert above == pytest.approx(below, abs=1e-9)


def test_set_vacuum_preserves_the_atomic_stack_geometry(ortho_slab: Slab) -> None:
    """Rescaling the Vacuum must not move atoms relative to each other.

    Pinned by the failure mode it prevents: scaling ``c`` while keeping
    ``frac_coords`` unchanged would stretch the stack and report a 16.04 A gap for
    a 15 A request. Here the interlayer spacing must stay ``a / sqrt(3)``.
    """
    rescaled = set_vacuum(ortho_slab, 15.0)

    before = np.asarray(ortho_slab.cart_coords, dtype=float)
    after = np.asarray(rescaled.cart_coords, dtype=float)
    # Same in-plane structure per layer, and the same set of z-levels up to the
    # rigid translation + the (here identity) stack thickness preservation.
    z_before = np.sort(before[:, 2])
    z_after = np.sort(after[:, 2])
    assert np.allclose(np.diff(z_before), np.diff(z_after), atol=1e-9)
    assert np.allclose(np.diff(z_after), PT111_D111, atol=1e-5)

    # The in-plane lattice is untouched: only c changed.
    m_before = np.array(ortho_slab.lattice.matrix, dtype=float)
    m_after = np.array(rescaled.lattice.matrix, dtype=float)
    assert np.allclose(m_before[:2], m_after[:2], atol=1e-12)


def test_set_vacuum_is_idempotent(ortho_slab: Slab) -> None:
    """Asking twice for the same Vacuum changes nothing the second time."""
    once = set_vacuum(ortho_slab, 15.0)
    twice = set_vacuum(once, 15.0)

    assert vacuum_gap(twice) == pytest.approx(15.0, abs=1e-12)
    assert float(twice.lattice.c) == pytest.approx(float(once.lattice.c), abs=1e-9)
    assert np.allclose(
        np.asarray(twice.cart_coords, dtype=float),
        np.asarray(once.cart_coords, dtype=float),
        atol=1e-9,
    )


def test_set_vacuum_keeps_the_slab_identity_and_site_properties(
    ortho_slab: Slab,
) -> None:
    """The Miller index, ``shift`` and site properties survive the rescale.

    The fingerprint is computed from ``shift`` and the layer composition
    downstream, so a rescale that dropped ``shift`` would silently repoint every
    Termination identity.
    """
    rescaled = set_vacuum(ortho_slab, 15.0)

    assert rescaled.miller_index == ortho_slab.miller_index
    assert float(rescaled.shift) == float(ortho_slab.shift)
    assert set(rescaled.site_properties) == set(ortho_slab.site_properties)
    assert [str(s.specie) for s in rescaled] == [
        str(s.specie) for s in ortho_slab
    ]
    assert isinstance(rescaled, Slab)


def test_set_vacuum_does_not_mutate_its_input(ortho_slab: Slab) -> None:
    """The input slab is left untouched; a new Slab is returned."""
    c_before = float(ortho_slab.lattice.c)
    gap_before = vacuum_gap(ortho_slab)
    coords_before = np.asarray(ortho_slab.cart_coords, dtype=float).copy()

    set_vacuum(ortho_slab, 5.0)

    assert float(ortho_slab.lattice.c) == c_before
    assert vacuum_gap(ortho_slab) == gap_before
    assert np.allclose(
        np.asarray(ortho_slab.cart_coords, dtype=float), coords_before, atol=1e-12
    )


def test_set_vacuum_also_centres_a_stack_that_straddles_the_boundary(
    ortho_slab: Slab,
) -> None:
    """The rescale is correct even when the input stacks across z = 0.

    ``center_slab=True`` normally prevents this, but the algorithm must not depend
    on it (§C.3 "Well-definedness"): the gap definition is periodic, so the
    rescale is too.
    """
    straddled = _straddle(ortho_slab, frac_z=0.35)
    rescaled = set_vacuum(straddled, 12.0)
    cart_z = np.asarray(rescaled.cart_coords, dtype=float)[:, 2]
    c = float(rescaled.lattice.c)

    assert vacuum_gap(rescaled) == pytest.approx(12.0, abs=1e-9)
    assert float(cart_z.min()) == pytest.approx(6.0, abs=1e-9)
    assert c - float(cart_z.max()) == pytest.approx(6.0, abs=1e-9)


@pytest.mark.parametrize("bad", [0.0, -5.0, -1e-9])
def test_set_vacuum_rejects_a_non_positive_vacuum(
    ortho_slab: Slab, bad: float
) -> None:
    """Upstream silently accepts a negative ``min_vacuum_size``; the library does not.

    The message follows CONVENTIONS.md §H.1: parameter name, the rejected value
    echoed with ``!r``, why, the valid range, and a pasteable example.
    """
    with pytest.raises(InvalidParameterError) as excinfo:
        set_vacuum(ortho_slab, bad)

    message = str(excinfo.value)
    assert "vacuum=" in message
    assert "is invalid" in message
    assert "must be > 0" in message
    assert "Example: " in message
    assert "\n" not in message
    assert len(message) < 200
    # The specific exception type, not just any ValueError.
    assert isinstance(excinfo.value, ValueError)


# --- count_z_levels: the layer-grouping rule -------------------------------


def test_count_z_levels_of_the_five_layer_pt111_slab(ortho_slab: Slab) -> None:
    """Pt(111) at 10 A minimum thickness is a 5-layer slab: 5 atoms, 5 z-levels."""
    rescaled = set_vacuum(ortho_slab, 15.0)

    assert len(rescaled) == 5
    assert count_z_levels(rescaled) == 5


def test_count_z_levels_is_not_the_atom_count() -> None:
    """A layer count must be measured, never inferred from the atom count (§D.4).

    A Si-C bilayer is two layers of two atoms each, spaced 0.629 A apart -- well
    inside the 0.1 A default tolerance's complement, and far enough apart to be
    resolved. So a 4-atom cell holds 2 atoms per layer, 2 layers.
    """
    lattice = Lattice.from_parameters(3.0, 3.0, 20.0, 90.0, 90.0, 90.0)
    structure = Structure(
        lattice,
        ["Si", "Si", "C", "C"],
        [[0.0, 0.0, 0.1], [0.5, 0.5, 0.1], [0.0, 0.0, 0.13145], [0.5, 0.5, 0.13145]],
    )

    assert count_z_levels(structure) == 2


def test_count_z_levels_groups_every_site_exactly_once(ortho_slab: Slab) -> None:
    """``group_z_levels`` partitions the sites, bottom layer first."""
    groups = group_z_levels(ortho_slab)

    assert len(groups) == count_z_levels(ortho_slab)
    assert sorted(i for group in groups for i in group) == list(
        range(len(ortho_slab))
    )
    assert all(group for group in groups)
    # Bottom layer first.
    layer_z = [float(ortho_slab.cart_coords[group, 2].mean()) for group in groups]
    assert layer_z == sorted(layer_z)


def test_count_z_levels_tolerance_separates_bilayers(ortho_slab: Slab) -> None:
    """The tolerance is the layer decision, so it must be honoured exactly.

    Re-using the Pt(111) slab here would be a poor witness (all its layers are
    2.265 A apart). Instead build a controlled cell with gaps of 0.05 and 0.5 A:
    a tolerance between them must resolve 2 layers, and one below both must
    resolve 3.
    """
    lattice = Lattice.from_parameters(3.0, 3.0, 20.0, 90.0, 90.0, 90.0)
    structure = Structure(
        lattice,
        ["Pt", "Pt", "Pt"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.05 / 20.0], [0.0, 0.0, 0.55 / 20.0]],
    )

    assert count_z_levels(structure, tol=0.1) == 2
    assert count_z_levels(structure, tol=0.01) == 3


def test_count_z_levels_of_an_empty_structure_is_zero() -> None:
    """A structure with no sites has no layers -- not an exception."""
    lattice = Lattice.from_parameters(3.0, 3.0, 20.0, 90.0, 90.0, 90.0)
    structure = Structure(lattice, [], [])

    assert count_z_levels(structure) == 0
    assert group_z_levels(structure) == []


def test_count_z_levels_of_a_bulk_crystal(bulk_pt: Structure) -> None:
    """Conventional fcc Pt has 2 z-levels (§E.5), which is what makes it not a slab."""
    assert count_z_levels(bulk_pt) == 2


# --- the role contract: which sites are Slab, which are Adsorbate ----------


def test_role_constants_are_the_two_agreed_strings() -> None:
    """§G.4 pins the exact values; they are a cross-ticket contract."""
    assert SLAB_ROLE == "slab"
    assert ADSORBATE_ROLE == "adsorbate"


def test_split_roles_partitions_a_tagged_structure(ortho_slab: Slab) -> None:
    """A tagged structure splits into ascending index lists, with no site lost."""
    tagged = ortho_slab.copy()
    n_slab = len(tagged)
    tagged.append("C", [0.0, 0.0, 0.9], coords_are_cartesian=True)
    tagged.append("O", [0.0, 0.0, 0.95], coords_are_cartesian=True)
    tagged.add_site_property(
        "role", [SLAB_ROLE] * n_slab + [ADSORBATE_ROLE] * 2
    )

    slab_indices, adsorbate_indices = split_roles(tagged)

    assert slab_indices == list(range(n_slab))
    assert adsorbate_indices == [n_slab, n_slab + 1]
    assert sorted(slab_indices + adsorbate_indices) == list(range(len(tagged)))


def test_split_roles_survives_a_supercell(ortho_slab: Slab) -> None:
    """The tag is what makes a supercell still classifiable (§G.4 durability)."""
    tagged = ortho_slab.copy()
    n_slab = len(tagged)
    tagged.append("C", [0.0, 0.0, 0.9], coords_are_cartesian=True)
    tagged.add_site_property("role", [SLAB_ROLE] * n_slab + [ADSORBATE_ROLE])

    supercell = tagged.copy()
    supercell.make_supercell([2, 1, 1])
    slab_indices, adsorbate_indices = split_roles(supercell)

    assert len(slab_indices) == 2 * n_slab
    assert len(adsorbate_indices) == 2
    assert sorted(slab_indices + adsorbate_indices) == list(range(len(supercell)))


def test_split_roles_raises_on_a_structure_without_the_property(
    ortho_slab: Slab,
) -> None:
    """§H.2: name the missing property, the valid values, and a pasteable example."""
    with pytest.raises(ValueError) as excinfo:
        split_roles(ortho_slab)

    message = str(excinfo.value)
    assert "role" in message
    assert "cannot separate Slab from Adsorbate" in message
    assert "Example: " in message
    assert "\n" not in message
    assert len(message) < 260


def test_split_roles_rejects_an_unknown_role_value(ortho_slab: Slab) -> None:
    """A third role value is a bug at the writer, so it must not be silently dropped."""
    tagged = ortho_slab.copy()
    tagged.add_site_property("role", ["slab"] * (len(tagged) - 1) + ["subsurface"])

    with pytest.raises(ValueError) as excinfo:
        split_roles(tagged)

    message = str(excinfo.value)
    assert "subsurface" in message
    assert "'slab'" in message or "slab" in message
    assert "Example: " in message
    assert "\n" not in message
