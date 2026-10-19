"""Tests for ``place_adsorbate``: putting an Adsorbate on an Adsorption site (#6).

Expected values are the pinned measurements in CONVENTIONS.md §F.1, §F.3 and §J.6,
restated as literals here so these tests can disagree with the implementation.

The canonical Slab is Pt(111) cut by ``generate_slabs``: 5 atoms, 5 z-levels, an
exact 15 A Vacuum and a **top layer at z = 16.560011 A** (§J.6 — never the older
18.120023 from §C.2, which came from a mis-centred stack). Its ``ontop`` site
sits at that z with ``distance=0``, so the caller's ``height`` *is* the
anchor-to-surface distance.

``ase.build.molecule`` does not return atoms in the order a chemist writes the
formula: CO is ``['O', 'C']`` and OH is ``['O', 'H']`` (§F.1). Every anchor is
therefore looked up by element symbol, and these tests pin that by asserting the
*species at the anchor index* — an index-0 implementation places the wrong atom
at the binding height and inverts the molecule (§F.2 measured ``O_z - C_z =
-1.1503`` for that regression).
"""

from __future__ import annotations

import ase.io
import numpy as np
import pytest
from ase.build import molecule as ase_molecule
from ase.constraints import FixAtoms
from pymatgen.core import Structure
from pymatgen.core.surface import Slab, SlabGenerator
from pymatgen.io.vasp import Poscar

from dsh4sci.adsorption import (
    AdsorptionSite,
    PlacedAdsorbate,
    find_adsorption_sites,
    place_adsorbate,
)
from dsh4sci.constraints import fix_bottom_atoms
from dsh4sci.errors import Dsh4sciError
from dsh4sci.geometry import ADSORBATE_ROLE, SLAB_ROLE, split_roles
from dsh4sci.slabs import generate_slabs

# --- pinned numbers, CONVENTIONS.md §F.1 / §F.3 / §J.6 ----------------------

PT111_TOP_LAYER_Z = 16.560011  # top z-level of the canonical Pt(111) Slab, A
PT111_N_ATOMS = 5  # Pt5
CO_BOND_LENGTH = 1.15034  # C-O in ase.build.molecule("CO"), A
DEFAULT_HEIGHT = 1.85  # the documented default anchor-surface distance, A

# The ticket's acceptance tolerance: the anchor sits 1.85 +/- 0.05 A from its
# nearest Slab atom.
HEIGHT_TOL = 0.05

# The placement is exact arithmetic, so the height can be pinned much tighter
# than the acceptance band; measured deviation is ~1e-15 A.
EXACT_TOL = 1e-9

# CONVENTIONS.md §H.1 rule 7: one line, under 210 characters.
MAX_MESSAGE_LEN = 210

# --- helpers ---------------------------------------------------------------


def top_layer_z(structure: Structure) -> float:
    """Cartesian z of the highest atomic layer (§J.6: read it, never hard-code)."""
    return float(np.max(np.asarray(structure.cart_coords)[:, 2]))


def slab_top_layer_z(placed: PlacedAdsorbate) -> float:
    """``top_layer_z`` of the Slab part only -- the Adsorbate sits above it."""
    cart = np.asarray(placed.structure.cart_coords, dtype=float)
    return float(np.max(cart[: placed.n_slab_atoms, 2]))


def adsorbate_atoms(placed: PlacedAdsorbate) -> list:
    """The Adsorbate's sites, in :func:`ase.build.molecule` order."""
    return [
        placed.structure[index]
        for index in range(placed.n_slab_atoms, len(placed.structure))
    ]


def only_adsorbate_atom(placed: PlacedAdsorbate, symbol: str):
    """The single Adsorbate atom with element ``symbol`` (order-independent)."""
    matches = [atom for atom in adsorbate_atoms(placed) if atom.specie.symbol == symbol]
    assert len(matches) == 1, f"expected exactly one {symbol}, got {len(matches)}"
    return matches[0]


def nearest_slab_distance(n_slab_atoms: int, structure: Structure, index: int) -> float:
    """Distance from site ``index`` to the nearest Slab atom, minimum-image aware.

    ``Structure.distance_matrix`` folds periodic images (§G.1), which is required
    here: the ``ontop`` site is a periodic image of the top layer atom, so raw
    Cartesian subtraction would report ~16 A instead of 1.85 A (§F.3).
    """
    distances = np.asarray(structure.distance_matrix, dtype=float)
    return float(distances[index, :n_slab_atoms].min())


# --- fixtures (ticket-local: conftest.py belongs to ticket #2) --------------


@pytest.fixture
def pt_slab(pt_bulk: Structure) -> Slab:
    """The canonical Pt(111) Slab: ``Pt5``, 5 z-levels, exact 15 A Vacuum."""
    return generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=10.0, vacuum=15.0)[0]


@pytest.fixture
def sites(pt_slab: Slab) -> list[AdsorptionSite]:
    """Every Adsorption site of the canonical Slab, in the project's vocabulary."""
    return find_adsorption_sites(pt_slab)


@pytest.fixture
def ontop_site(sites: list[AdsorptionSite]) -> AdsorptionSite:
    """The single ``ontop`` site of the primitive Pt(111) cell."""
    return next(site for site in sites if site.site_type == "ontop")


# --- the ticket's acceptance criteria --------------------------------------


def test_the_anchor_binds_at_the_documented_height_above_the_surface(
    pt_slab: Slab, ontop_site: AdsorptionSite
) -> None:
    """C in CO ends up 1.85 +/- 0.05 A from the nearest Pt -- with no tuning.

    An ``ontop`` site sits directly above one Slab atom, so on a Slab built with
    ``distance=0`` the anchor-to-surface distance equals ``height`` exactly; the
    acceptance band is the loose bound, the exact value is asserted below it.
    """
    placed = place_adsorbate(pt_slab, ontop_site, molecule="CO", anchor="C")

    distance = nearest_slab_distance(
        placed.n_slab_atoms, placed.structure, placed.anchor_index
    )

    assert distance == pytest.approx(DEFAULT_HEIGHT, abs=HEIGHT_TOL)
    assert distance == pytest.approx(DEFAULT_HEIGHT, abs=1e-6)


def test_the_molecule_points_away_from_the_surface(
    pt_slab: Slab, ontop_site: AdsorptionSite
) -> None:
    """In CO the oxygen is *above* the carbon: ``O_z - C_z = +1.15034 A``.

    The sign is the whole point. Anchoring by index 0 (ASE returns ``['O', 'C']``)
    puts O at the binding height and C above it, which measures ``-1.1503`` and is
    a CO molecule hanging upside down into the surface (§F.2).
    """
    placed = place_adsorbate(pt_slab, ontop_site, molecule="CO", anchor="C")

    carbon = placed.structure[placed.anchor_index]
    oxygen = only_adsorbate_atom(placed, "O")

    assert carbon.specie.symbol == "C"
    assert oxygen.specie.symbol == "O"
    assert oxygen.coords[2] > carbon.coords[2]
    assert oxygen.coords[2] - carbon.coords[2] == pytest.approx(
        CO_BOND_LENGTH, abs=1e-5
    )


@pytest.mark.parametrize("height", [1.5, DEFAULT_HEIGHT, 2.5])
def test_the_height_parameter_controls_the_anchor_surface_distance(
    pt_slab: Slab, ontop_site: AdsorptionSite, height: float
) -> None:
    """``height`` is the anchor-surface distance, not an offset from something else.

    An ``ontop`` site sits exactly on the top layer, so the anchor's clearance
    above that layer and its distance to the atom below are the same number.
    """
    placed = place_adsorbate(
        pt_slab, ontop_site, molecule="CO", anchor="C", height=height
    )

    distance = nearest_slab_distance(
        placed.n_slab_atoms, placed.structure, placed.anchor_index
    )
    assert distance == pytest.approx(height, abs=1e-6)
    assert placed.structure[placed.anchor_index].coords[2] == pytest.approx(
        slab_top_layer_z(placed) + height, abs=1e-6
    )


@pytest.mark.parametrize("site_type", ["ontop", "bridge", "hollow"])
def test_the_anchor_lands_on_the_requested_site(
    pt_slab: Slab, sites: list[AdsorptionSite], site_type: str
) -> None:
    """The anchor atom -- not the molecule's centroid -- sits on the site's x, y.

    The ``ontop`` site is a periodic image of its top-layer atom (§F.3), so the
    site's Cartesian x, y are *not* the same numbers as the top atom's; using the
    atom's coordinates instead of the site's would be off by a lattice vector.
    """
    site = next(candidate for candidate in sites if candidate.site_type == site_type)

    placed = place_adsorbate(pt_slab, site, molecule="CO", anchor="C")

    anchor = np.asarray(placed.structure[placed.anchor_index].coords, dtype=float)
    assert anchor[0] == pytest.approx(site.position[0], abs=EXACT_TOL)
    assert anchor[1] == pytest.approx(site.position[1], abs=EXACT_TOL)
    assert anchor[2] == pytest.approx(PT111_TOP_LAYER_Z + DEFAULT_HEIGHT, abs=1e-6)


def test_a_molecule_whose_binding_direction_is_not_along_z_is_rotated(
    pt_slab: Slab, ontop_site: AdsorptionSite
) -> None:
    """Water binds through O, but ASE puts its hydrogens on the far side of O.

    ``ase.build.molecule("H2O")`` has both H at z = -0.477 relative to O, so a
    translate-only placement buries them ~0.48 A inside the Slab. The anchor-
    relative frame must be rotated so the anchor -> (mean of the other atoms)
    direction points along +z *before* it is translated (§F.2).
    """
    placed = place_adsorbate(pt_slab, ontop_site, molecule="H2O", anchor="O")

    oxygen = placed.structure[placed.anchor_index]
    hydrogens = [
        placed.structure[index]
        for index in range(placed.n_slab_atoms, len(placed.structure))
        if placed.structure[index].specie.symbol == "H"
    ]

    assert oxygen.specie.symbol == "O"
    assert len(hydrogens) == 2
    for hydrogen in hydrogens:
        assert hydrogen.coords[2] > oxygen.coords[2]


# The trap this parameterisation exists for: ASE's atom order is not the order a
# chemist writes, so "index 0" is the wrong atom for CO, NO, H2O and OH (§F.1).
# (molecule, anchor, index of the anchor in ase.build.molecule(molecule))
ANCHOR_IS_NOT_FIRST = [
    ("CO", "C", 1),
    ("NO", "O", 1),
    ("OH", "H", 1),
    ("H2O", "H", 1),
]


@pytest.mark.parametrize(("formula", "anchor", "ase_index"), ANCHOR_IS_NOT_FIRST)
def test_the_anchor_is_found_by_element_symbol_not_by_index(
    pt_slab: Slab,
    ontop_site: AdsorptionSite,
    formula: str,
    anchor: str,
    ase_index: int,
) -> None:
    """The atom that binds is the one named by symbol, whatever ASE's order is.

    ``ase.build.molecule`` returns ``['O', 'C']`` for CO, ``['N', 'O']`` for NO and
    ``['O', 'H']`` for OH, so an implementation that anchors ``atoms[0]`` binds the
    wrong element and puts the requested anchor *above* it -- a molecule standing
    on its head. Pinning the species at the returned index fails immediately for
    such an implementation, and the direction check catches the inversion too.
    """
    ase_symbols = list(ase_molecule(formula).get_chemical_symbols())
    assert ase_symbols.index(anchor) == ase_index  # premise: not the first atom
    assert ase_index != 0

    placed = place_adsorbate(pt_slab, ontop_site, molecule=formula, anchor=anchor)

    bound = placed.structure[placed.anchor_index]
    assert bound.specie.symbol == anchor
    assert bound.coords[2] == pytest.approx(
        slab_top_layer_z(placed) + DEFAULT_HEIGHT, abs=1e-6
    )
    for atom in adsorbate_atoms(placed):
        if atom is bound:
            continue
        assert atom.coords[2] > bound.coords[2]


def test_the_site_property_marks_every_atom_with_its_role(
    pt_slab: Slab, ontop_site: AdsorptionSite
) -> None:
    """The ``role`` tag lets a consumer separate Slab from Adsorbate (§G.4).

    Species membership cannot do it -- OH on TiO2 shares O with the Slab -- and
    neither can a z threshold, because the bug it must detect is an Adsorbate atom
    *below* the Slab's top layer.
    """
    placed = place_adsorbate(pt_slab, ontop_site, molecule="CO", anchor="C")

    roles = placed.structure.site_properties["role"]

    assert roles == [SLAB_ROLE] * PT111_N_ATOMS + [ADSORBATE_ROLE] * 2
    assert split_roles(placed.structure) == ([0, 1, 2, 3, 4], [5, 6])
    assert placed.n_slab_atoms == PT111_N_ATOMS
    assert len(placed.structure) == PT111_N_ATOMS + 2
    assert placed.anchor_index == 6
    assert placed.structure[placed.anchor_index].specie.symbol == "C"


def test_every_site_of_a_supercelled_slab_can_be_placed_on(pt_slab: Slab) -> None:
    """Each site of a 3x3 Slab gets its own Adsorbate, on its own x, y.

    ``make_supercell`` mutates in place by default on this stack and returns
    ``self`` (§J.8), so the supercell is built from a copy. On a supercell the
    ``ontop`` sites are periodic images at many different x, y, which is exactly
    where "use the site's own coordinates, not the top atom's" stops being
    indistinguishable from the wrong implementation.
    """
    s33 = pt_slab.copy().make_supercell([3, 3, 1], in_place=False)
    assert len(s33) == 9 * PT111_N_ATOMS  # the copy really is a 3x3 cell

    supercell_sites = find_adsorption_sites(s33)
    assert len(supercell_sites) == 54

    for site in supercell_sites:
        placed = place_adsorbate(s33, site, molecule="CO", anchor="C")
        anchor = np.asarray(placed.structure[placed.anchor_index].coords, dtype=float)
        assert placed.n_slab_atoms == 9 * PT111_N_ATOMS
        assert anchor[0] == pytest.approx(site.position[0], abs=EXACT_TOL)
        assert anchor[1] == pytest.approx(site.position[1], abs=EXACT_TOL)
        assert anchor[2] == pytest.approx(
            PT111_TOP_LAYER_Z + DEFAULT_HEIGHT, abs=1e-6
        )


def test_the_written_poscar_reads_back_through_ase_with_the_expected_atom_count(
    pt_slab: Slab, ontop_site: AdsorptionSite, tmp_path
) -> None:
    """The ticket's file-I/O criterion: ``Poscar`` out, ``ase.io.read`` back, 7 atoms.

    ASE keeps the atom count and the POSCAR block order (§F.4), so both are safe to
    assert here; its ``get_chemical_formula()`` alphabetises to ``COPt5`` and is
    deliberately not asserted.
    """
    placed = place_adsorbate(pt_slab, ontop_site, molecule="CO", anchor="C")
    path = tmp_path / "CO_on_Pt111.vasp"

    Poscar(placed.structure).write_file(str(path))
    atoms = ase.io.read(str(path))

    assert len(atoms) == PT111_N_ATOMS + 2
    assert len(atoms) == len(placed.structure)
    assert atoms.get_chemical_symbols() == ["Pt"] * PT111_N_ATOMS + ["O", "C"]


def test_the_slab_count_is_the_source_of_truth_after_a_file_round_trip(
    pt_slab: Slab, ontop_site: AdsorptionSite, tmp_path
) -> None:
    """The ``role`` tag is lost through POSCAR; ``n_slab_atoms`` is what survives.

    §G.4 measured this: a POSCAR or CIF round-trip drops site properties, so an
    agent that reads the file back must count atoms rather than ask for the tag.
    """
    placed = place_adsorbate(pt_slab, ontop_site, molecule="CO", anchor="C")
    path = tmp_path / "CO_on_Pt111.vasp"
    Poscar(placed.structure).write_file(str(path))

    reloaded = Structure.from_file(str(path))

    assert "role" not in reloaded.site_properties
    assert len(reloaded) == len(placed.structure)
    assert len(reloaded) - placed.n_slab_atoms == 2
    assert reloaded[placed.anchor_index].specie.symbol == "C"


def test_placement_keeps_the_fixed_bottom_and_leaves_the_adsorbate_free(
    pt_slab: Slab, ontop_site: AdsorptionSite, tmp_path
) -> None:
    """A Slab prepared by ``fix_bottom_atoms`` stays fixed; the Adsorbate is free.

    Two failures are covered at once: dropping the Slab's flags (which would let a
    relaxation move the bulk), and writing them as ``None`` for the Adsorbate
    atoms, which makes the POSCAR writer raise ``TypeError: 'NoneType' object is
    not iterable`` (measured on this stack).
    """
    fixed = fix_bottom_atoms(pt_slab, fraction=0.4)
    slab_flags = list(fixed.site_properties["selective_dynamics"])
    assert slab_flags[0] == (False, False, False)

    placed = place_adsorbate(fixed, ontop_site, molecule="CO", anchor="C")

    flags = placed.structure.site_properties["selective_dynamics"]
    assert flags[: len(fixed)] == slab_flags
    assert flags[len(fixed) :] == [(True, True, True)] * 2

    path = tmp_path / "fixed_CO_on_Pt111.vasp"
    Poscar(placed.structure).write_file(str(path))
    atoms = ase.io.read(str(path))
    assert len(atoms) == PT111_N_ATOMS + 2
    # §J.7: ASE *does* surface the fixed set as a FixAtoms constraint.
    assert isinstance(atoms.constraints[0], FixAtoms)
    assert sorted(atoms.constraints[0].get_indices().tolist()) == [0, 1]


def test_the_callers_slab_is_not_mutated(
    pt_slab: Slab, ontop_site: AdsorptionSite
) -> None:
    """Placement returns a new structure; the caller's Slab is untouched."""
    cart_before = np.array(pt_slab.cart_coords, copy=True)
    frac_before = np.array(pt_slab.frac_coords, copy=True)

    place_adsorbate(pt_slab, ontop_site, molecule="CO", anchor="C")

    assert len(pt_slab) == PT111_N_ATOMS
    assert np.array_equal(np.asarray(pt_slab.cart_coords), cart_before)
    assert np.array_equal(np.asarray(pt_slab.frac_coords), frac_before)
    assert "role" not in pt_slab.site_properties


# --- §H error messages: named parameter, rejected value, valid set, valid call --


def test_an_unknown_molecule_is_rejected_with_the_valid_set(
    pt_slab: Slab, ontop_site: AdsorptionSite
) -> None:
    """ASE raises a bare ``KeyError: 'XYZ'``; an agent needs the valid set instead."""
    with pytest.raises(ValueError) as excinfo:
        place_adsorbate(pt_slab, ontop_site, molecule="XYZ", anchor="C")

    message = str(excinfo.value)

    assert isinstance(excinfo.value, Dsh4sciError)
    assert message.startswith(
        "molecule='XYZ' is not a known ASE molecule. Valid: an ASE `g2` name such as "
    )
    assert "Example: place_adsorbate(" in message
    assert "\n" not in message
    assert len(message) <= MAX_MESSAGE_LEN
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_an_anchor_absent_from_the_molecule_is_rejected_with_the_elements(
    pt_slab: Slab, ontop_site: AdsorptionSite
) -> None:
    """Python raises ``'N' is not in list``; the message must name the valid set."""
    with pytest.raises(ValueError) as excinfo:
        place_adsorbate(pt_slab, ontop_site, molecule="CO", anchor="N")

    message = str(excinfo.value)

    assert isinstance(excinfo.value, Dsh4sciError)
    assert message.startswith(
        "anchor='N' is not present in Adsorbate 'CO' (elements: ['C', 'O']). "
        "Valid: one of ['C', 'O']."
    )
    assert "Example: place_adsorbate(" in message
    assert "'N' is not in list" not in message
    assert "\n" not in message
    assert len(message) <= MAX_MESSAGE_LEN


@pytest.mark.parametrize("height", [0.0, -1.85, float("nan"), float("inf")])
def test_a_height_that_is_not_a_positive_length_is_rejected(
    pt_slab: Slab, ontop_site: AdsorptionSite, height: float
) -> None:
    """A zero or negative height puts the Adsorbate inside the Slab."""
    with pytest.raises(ValueError) as excinfo:
        place_adsorbate(
            pt_slab, ontop_site, molecule="CO", anchor="C", height=height
        )

    message = str(excinfo.value)

    assert isinstance(excinfo.value, Dsh4sciError)
    assert message.startswith("height=")
    assert "must be a finite float > 0" in message
    assert "Example: place_adsorbate(" in message
    assert "\n" not in message
    assert len(message) <= MAX_MESSAGE_LEN


def test_a_position_that_is_not_an_adsorption_site_is_rejected(
    pt_slab: Slab,
) -> None:
    """A bare tuple is a plausible mistake: name the type, not AttributeError."""
    with pytest.raises(TypeError) as excinfo:
        place_adsorbate(pt_slab, (0.0, 0.0, 16.56), molecule="CO", anchor="C")

    message = str(excinfo.value)

    assert message.startswith(
        "site=(0.0, 0.0, 16.56) is invalid: expected an AdsorptionSite, got tuple. "
        "Valid: a find_adsorption_sites() record."
    )
    assert "Example: place_adsorbate(" in message
    assert "\n" not in message
    assert len(message) <= MAX_MESSAGE_LEN


def test_the_public_names_are_re_exported_from_the_package() -> None:
    """The package-level API is the contract agents import from."""
    import dsh4sci

    assert dsh4sci.place_adsorbate is place_adsorbate
    assert dsh4sci.PlacedAdsorbate is PlacedAdsorbate
    assert dsh4sci.DEFAULT_BINDING_HEIGHT == DEFAULT_HEIGHT


def test_a_slab_that_is_not_orthogonal_is_placed_in_the_orthogonal_frame(
    pt_bulk: Structure,
) -> None:
    """Sites live in the orthogonal-c frame (§E), so placement must use that frame.

    A raw ``SlabGenerator`` Slab has ``alpha = beta = 135 deg``; its Cartesian
    coordinates differ from the frame the sites were reported in by ~20 A. Placing
    the molecule on the site's coordinates in the raw frame would put the Adsorbate
    nowhere near the Slab, so the shared orthogonalisation step runs here too.
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
    ).get_slabs()[0]
    assert raw.lattice.angles[0] != pytest.approx(90.0)

    site = next(
        candidate
        for candidate in find_adsorption_sites(raw)
        if candidate.site_type == "ontop"
    )
    placed = place_adsorbate(raw, site, molecule="CO", anchor="C")

    anchor = np.asarray(placed.structure[placed.anchor_index].coords, dtype=float)
    assert anchor[0] == pytest.approx(site.position[0], abs=EXACT_TOL)
    assert anchor[1] == pytest.approx(site.position[1], abs=EXACT_TOL)
    assert nearest_slab_distance(
        placed.n_slab_atoms, placed.structure, placed.anchor_index
    ) == pytest.approx(DEFAULT_HEIGHT, abs=1e-6)
    assert placed.structure.lattice.angles == pytest.approx(
        (90.0, 90.0, 60.0), abs=1e-9
    )


def test_a_slab_that_is_not_a_structure_is_rejected(
    ontop_site: AdsorptionSite,
) -> None:
    """The same TypeError the site finder raises for the same kind of mistake."""
    with pytest.raises(TypeError) as excinfo:
        place_adsorbate("Pt111.poscar", ontop_site, molecule="CO", anchor="C")

    message = str(excinfo.value)

    assert message.startswith(
        "slab='str' is invalid: expected a pymatgen Structure. Valid: a Slab-like "
        "Structure, e.g. from generate_slabs()."
    )
    assert "\n" not in message
    assert len(message) <= MAX_MESSAGE_LEN
