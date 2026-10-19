"""Tests for :func:`dsh4sci.constraints.fix_bottom_atoms`.

Every expectation here is a *pinned number* from ``CONVENTIONS.md`` §D.4, not a
value recomputed the same way the implementation computes it: the 4-layer Pt(111)
slab has z-levels ``7.500000, 9.765003, 12.030006, 14.295009``, ``c = 21.795`` and
a 15 A Vacuum.

The tests drive the behaviour through the public seam only -- the returned
structure and the POSCAR it writes -- never a private helper.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import pytest
from pymatgen.core import Structure
from pymatgen.core.surface import Slab
from pymatgen.io.vasp import Poscar

from dsh4sci.constraints import fix_bottom_atoms
from dsh4sci.errors import InvalidParameterError
from dsh4sci.geometry import count_z_levels, group_z_levels, vacuum_gap
from dsh4sci.slabs import generate_slabs

#: The four z-levels of the pinned 4-layer Pt(111) slab (§D.4), bottom first.
FOUR_LAYER_Z = (7.500000, 9.765003, 12.030006, 14.295009)


@pytest.fixture
def four_layer_slab(pt_bulk) -> Slab:
    """The pinned 4-layer Pt(111) slab: 4 atoms, 4 z-levels, a 15 A Vacuum (§D.4)."""
    slabs = generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=8.0, vacuum=15.0)
    assert len(slabs) == 1
    return slabs[0]


@pytest.fixture
def supercell_slab(four_layer_slab: Slab) -> Slab:
    """2x2 in-plane supercell of the pinned slab: 16 atoms, 4 per z-level.

    ``Slab.make_supercell`` mutates its receiver, so this copies first -- a mutated
    fixture would silently hand the next test a 16-atom "4-layer slab".
    """
    supercell = four_layer_slab.copy().make_supercell([2, 2, 1])
    assert len(supercell) == 16
    assert count_z_levels(supercell) == 4
    return supercell


def flags_of(structure: Structure) -> list[tuple[bool, bool, bool]]:
    """The ``selective_dynamics`` flags, one 3-tuple per site."""
    return [
        tuple(bool(flag) for flag in site_flags)
        for site_flags in structure.site_properties["selective_dynamics"]
    ]


def fixed_mask(structure: Structure) -> list[bool]:
    """Per-site ``True`` where all three directions are fixed (``F F F``)."""
    return [not any(flags) for flags in flags_of(structure)]


def fixed_indices(structure: Structure) -> set[int]:
    """Indices of the sites whose flags are all ``False``."""
    return {index for index, is_fixed in enumerate(fixed_mask(structure)) if is_fixed}


def fixed_coordinates(structure: Structure) -> frozenset[tuple[float, float, float]]:
    """The Cartesian positions of the fixed sites, as a set."""
    return frozenset(
        tuple(round(float(value), 6) for value in structure.cart_coords[index])
        for index in fixed_indices(structure)
    )


def reordered_slab(slab: Slab, order: list[int]) -> Slab:
    """A ``Slab`` holding ``slab``'s sites in ``order``.

    Pymatgen does not contract the order in which it returns sites, and a caller's
    slab may have been reordered (``make_supercell``, a hand-built cell, another
    generator). Rebuilding through the constructor is the only way to get a Slab
    whose site indices run in a chosen order.
    """
    return Slab(
        lattice=slab.lattice,
        species=[slab.sites[index].species for index in order],
        coords=[slab.cart_coords[index] for index in order],
        miller_index=slab.miller_index,
        oriented_unit_cell=slab.oriented_unit_cell,
        shift=slab.shift,
        scale_factor=slab.scale_factor,
        reorient_lattice=True,
        coords_are_cartesian=True,
    )


def test_fixes_floor_of_fraction_of_a_four_layer_slab(four_layer_slab: Slab) -> None:
    assert len(four_layer_slab) == 4
    assert sum(fixed_mask(fix_bottom_atoms(four_layer_slab, 0.5))) == 2


def test_uses_floor_rather_than_round(four_layer_slab: Slab) -> None:
    # 4 * 0.875 == 3.5: floor gives 3, round would give 4.
    assert sum(fixed_mask(fix_bottom_atoms(four_layer_slab, 0.875))) == 3


def test_fixed_atoms_are_the_lowest_z_atoms(four_layer_slab: Slab) -> None:
    fixed = fix_bottom_atoms(four_layer_slab, 0.5)
    mask = fixed_mask(fixed)
    fixed_z = sorted(
        round(float(coords[2]), 6)
        for coords, is_fixed in zip(fixed.cart_coords, mask)
        if is_fixed
    )
    assert fixed_z == pytest.approx(list(FOUR_LAYER_Z[:2]), abs=1e-6)


def test_poscar_output_carries_the_three_column_flags(
    four_layer_slab: Slab, tmp_path
) -> None:
    fixed = fix_bottom_atoms(four_layer_slab, 0.5)
    poscar_path = tmp_path / "POSCAR"
    Poscar(fixed).write_file(poscar_path)

    text = poscar_path.read_text()
    assert "Selective dynamics" in text
    assert text.count(" F F F") == 2
    assert text.count(" T T T") == 2

    # Round-trip through pymatgen, deliberately NOT through ASE: ASE drops
    # selective dynamics on read (CONVENTIONS.md §D.8), so an ASE round-trip would
    # pass even if the flags had never been written.
    back = Structure.from_file(poscar_path)
    assert flags_of(back) == flags_of(fixed)


def test_vacuum_is_unchanged_by_fixing(four_layer_slab: Slab) -> None:
    gap_before = vacuum_gap(four_layer_slab)
    fixed = fix_bottom_atoms(four_layer_slab, 0.5)

    # Only site properties may change: no coordinate or lattice mutation at all.
    # The Vacuum is asserted *exactly*; the arrays carry a 1e-12 tolerance because
    # Structure.copy() re-derives Cartesian coordinates from fractions, so the last
    # bit is not stable. That is float round-off, not a moved atom.
    assert vacuum_gap(fixed) == gap_before
    np.testing.assert_allclose(
        fixed.lattice.matrix, four_layer_slab.lattice.matrix, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        fixed.cart_coords, four_layer_slab.cart_coords, rtol=0, atol=1e-12
    )


def test_input_structure_is_not_mutated(four_layer_slab: Slab) -> None:
    fix_bottom_atoms(four_layer_slab, 0.5)
    assert "selective_dynamics" not in four_layer_slab.site_properties


def test_fraction_zero_fixes_nothing_and_writes_no_property(
    four_layer_slab: Slab,
) -> None:
    result = fix_bottom_atoms(four_layer_slab, 0.0)

    # Absent means "fully free" (§D.3): a list of T T T would be a second way to
    # say nothing is fixed, and the absent form makes fraction=0 a no-op.
    assert "selective_dynamics" not in result.site_properties
    assert len(result) == 4
    np.testing.assert_allclose(
        result.cart_coords, four_layer_slab.cart_coords, rtol=0, atol=1e-12
    )


def test_fraction_one_fixes_every_atom(four_layer_slab: Slab) -> None:
    result = fix_bottom_atoms(four_layer_slab, 1.0)
    assert flags_of(result) == [(False, False, False)] * 4


@pytest.mark.parametrize("bad_fraction", [-0.1, 1.5, -1.0, 2.0, float("nan")])
def test_out_of_range_fraction_is_rejected(
    four_layer_slab: Slab, bad_fraction: float
) -> None:
    # floor(4 * -0.1) == -1 and floor(4 * 1.5) == 6 > n_atoms: the range must be
    # validated before k is computed, not after (§D.3).
    with pytest.raises(InvalidParameterError) as excinfo:
        fix_bottom_atoms(four_layer_slab, bad_fraction)

    message = str(excinfo.value)
    assert message.startswith(f"fraction={bad_fraction!r} is invalid")
    assert "must be in [0.0, 1.0]" in message
    assert "Example: fix_bottom_atoms(slab, fraction=0.5)" in message
    assert "\n" not in message
    assert len(message) <= 210


def test_non_numeric_fraction_is_rejected(four_layer_slab: Slab) -> None:
    with pytest.raises(InvalidParameterError) as excinfo:
        fix_bottom_atoms(four_layer_slab, None)

    message = str(excinfo.value)
    assert message.startswith("fraction=None is invalid")
    assert "Example: fix_bottom_atoms(slab, fraction=0.5)" in message
    assert len(message) <= 210


def test_the_fixed_set_is_independent_of_site_order(supercell_slab: Slab) -> None:
    # 16 atoms, fraction 0.375 -> k = 6: the four atoms of the bottom layer plus
    # *two* of the next, so the cut falls inside a layer where every atom shares a
    # z. The x/y/index tie-breaks (not the input order) must decide which two.
    reversed_slab = reordered_slab(
        supercell_slab, list(range(len(supercell_slab)))[::-1]
    )

    from_original = fix_bottom_atoms(supercell_slab, 0.375)
    from_reversed = fix_bottom_atoms(reversed_slab, 0.375)

    assert len(fixed_indices(from_original)) == 6
    assert len(fixed_indices(from_reversed)) == 6
    # Same atoms in absolute space, whichever order they were listed in...
    assert fixed_coordinates(from_original) == fixed_coordinates(from_reversed)
    # ...even though the reordering really did move them to different indices.
    assert fixed_indices(from_original) != fixed_indices(from_reversed)

    # Sensitivity of the input above: following the (z-only, stable) input order
    # selects two *different* atoms of the split layer, so this test would fail for
    # an implementation that forgot the tie-breaks.
    z = [float(coords[2]) for coords in reversed_slab.cart_coords]
    input_order_pick = sorted(range(len(reversed_slab)), key=lambda i: z[i])[:6]
    naive = frozenset(
        tuple(round(float(value), 6) for value in reversed_slab.cart_coords[index])
        for index in input_order_pick
    )
    assert naive != fixed_coordinates(from_reversed)


def test_every_fixed_atom_lies_in_the_two_lowest_layers(supercell_slab: Slab) -> None:
    fixed = fix_bottom_atoms(supercell_slab, 0.375)
    layers = group_z_levels(fixed)
    assert len(layers) == 4

    def z_of(index: int) -> float:
        return round(float(fixed.cart_coords[index][2]), 6)

    # k = 6 of 16 atoms: the whole bottom layer plus half of the second one, and
    # nothing at all in the upper two.
    assert {z_of(index) for index in fixed_indices(fixed)} == {
        z_of(layers[0][0]),
        z_of(layers[1][0]),
    }


def test_is_exported_from_the_package_root() -> None:
    from dsh4sci import fix_bottom_atoms as exported

    assert exported is fix_bottom_atoms


def test_the_slab_type_variable_is_public() -> None:
    """The return type of ``fix_bottom_atoms`` is part of its public signature.

    ``fix_bottom_atoms(slab: SlabT, ...) -> SlabT`` is a "you get back what you
    gave me" claim, so the variable has to be importable: a private ``_SlabT``
    would force ``SKILL.md`` to document a name no caller can use, and a reader
    cannot tell what the claim means.
    """
    import dsh4sci
    from dsh4sci import constraints

    assert "SlabT" in constraints.__all__
    assert dsh4sci.SlabT is constraints.SlabT
    assert isinstance(constraints.SlabT, TypeVar)
    assert constraints.SlabT.__bound__ is Structure
    assert not hasattr(constraints, "_SlabT"), (
        "a private _SlabT is still declared; SKILL.md may only document public names"
    )
