"""Tests for ``dsh4sci.adsorption``: Adsorption site enumeration (ticket #5).

The expected values are the pinned measurements in CONVENTIONS.md §E and §J.4,
restated as literals here so these tests can disagree with the implementation.

``Pt(111)``, the canonical Slab from ``generate_slabs`` (``Pt5``, top layer
``z = 16.560011 A``, exact 15 A Vacuum), enumerated with ``height=0.9``,
``distance=0.0`` and ``symm_reduce=-1``:

============  ==========  ===============================
supercell     area / A^2  sites (ontop / bridge / hollow)
============  ==========  ===============================
1 x 1          6.6644     1 / 3 / 2
2 x 2         26.6575     4 / 12 / 8
3 x 3         59.9794     9 / 27 / 18
============  ==========  ===============================

``distance=0.0`` leaves every site exactly on the top layer, so "at or above the
top layer" holds with ``>=`` up to float noise (measured ``|dz| <= 3.6e-15``).
A bulk crystal is rejected (§E.5): conventional Pt has a largest periodic gap of
``1.962 A``, below the 5.0 A threshold, and the message must say so.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab, SlabGenerator

from dsh4sci.adsorption import (
    MIN_VACUUM,
    MIN_Z_LEVELS,
    SITE_TYPES,
    AdsorptionSite,
    find_adsorption_sites,
    is_slab_like,
)
from dsh4sci.errors import Dsh4sciError, NotASlabError
from dsh4sci.geometry import count_z_levels, group_z_levels, vacuum_gap
from dsh4sci.loading import load_structure
from dsh4sci.slabs import generate_slabs

# --- pinned numbers, CONVENTIONS.md §E.3 / §J.4 -----------------------------

PT111_TOP_LAYER = 16.560011  # top z-level of the canonical Pt(111) Slab, A
PT111_PRIMITIVE_AREA = 6.664374  # in-plane area of the primitive cell, A^2
SUPERCELLS = (1, 2, 3)
PT111_COUNTS: dict[int, dict[str, int]] = {
    1: {"ontop": 1, "bridge": 3, "hollow": 2},
    2: {"ontop": 4, "bridge": 12, "hollow": 8},
    3: {"ontop": 9, "bridge": 27, "hollow": 18},
}

# Sites are computed in floating point; measured |site z - top layer z| is
# <= 3.6e-15, six orders of magnitude below this bound, so it does not weaken
# "at or above the top layer" -- it only stops float noise reading as a physics
# violation.
Z_NOISE_TOL = 1e-9

# CONVENTIONS.md §H.1 rule 7: one line, no trailing period, under 210 characters.
MAX_MESSAGE_LEN = 210


# --- helpers ---------------------------------------------------------------


def top_layer_z(structure: Structure) -> float:
    """Cartesian z of the highest atomic layer."""
    return float(np.max(np.asarray(structure.cart_coords)[:, 2]))


def in_plane_area(structure: Structure) -> float:
    """Area of the in-plane (a, b) cell in A^2: ``|a x b|``."""
    a, b = structure.lattice.matrix[0], structure.lattice.matrix[1]
    return float(np.linalg.norm(np.cross(a, b)))


def site_counts(sites: list[AdsorptionSite]) -> dict[str, int]:
    """Site type -> number of sites, in the canonical type order."""
    counts = dict.fromkeys(SITE_TYPES, 0)
    for site in sites:
        counts[site.site_type] += 1
    return counts


def supercelled(slab: Slab, n: int) -> Slab:
    """An ``n x n`` in-plane supercell of an already orthogonal slab.

    ``Slab.make_supercell`` defaults to ``in_place=True`` on this stack, so calling
    it on the fixture would silently rewrite the fixture (measured: a 5-site slab
    reported 20 sites after ``slab.make_supercell([2, 2, 1])``). Copy first, and
    ask explicitly for the non-mutating behaviour.
    """
    if n == 1:
        return slab.copy()
    return slab.copy().make_supercell([n, n, 1], in_place=False)


def wrapped(fractional_delta: np.ndarray) -> np.ndarray:
    """Fold a fractional difference into ``[-0.5, 0.5)`` (minimum image in-plane)."""
    return fractional_delta - np.round(fractional_delta)


# --- fixtures (ticket-local: conftest.py belongs to ticket #2) --------------


@pytest.fixture
def pt_slab(pt_bulk: Structure) -> Slab:
    """The canonical Pt(111) Slab: ``Pt5``, 5 z-levels, exact 15 A Vacuum."""
    return generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=10.0, vacuum=15.0)[0]


@pytest.fixture
def raw_non_orthogonal_slab(pt_bulk: Structure) -> Slab:
    """The raw ``SlabGenerator`` output, before orthogonalisation (§D2).

    ``a_z = b_z = -1.96155 A`` and ``alpha = beta = 135 deg``: the in-plane vectors
    carry z, which is what makes a supercell smear the stack.
    """
    generator = SlabGenerator(
        pt_bulk.copy(),
        (1, 1, 1),
        10.0,
        15.0,
        primitive=True,
        reorient_lattice=True,
        center_slab=True,
        lll_reduce=False,
        max_normal_search=None,
    )
    return generator.get_slabs()[0]


# --- the ticket's acceptance criteria --------------------------------------


def test_a_three_by_three_slab_returns_every_site_type(pt_slab: Slab) -> None:
    """Pt(111) 3x3 returns all three site types (§E.3)."""
    sites = find_adsorption_sites(supercelled(pt_slab, 3))

    assert sites, "a 3x3 Pt(111) slab must have adsorption sites"
    assert {site.site_type for site in sites} == {"ontop", "bridge", "hollow"}


@pytest.mark.parametrize("n", SUPERCELLS)
def test_site_counts_match_the_pinned_pt111_supercells(pt_slab: Slab, n: int) -> None:
    """The counts that scale with area (§E.3), as literals."""
    sites = find_adsorption_sites(supercelled(pt_slab, n))

    assert site_counts(sites) == PT111_COUNTS[n]


def test_site_counts_scale_with_surface_area(pt_slab: Slab) -> None:
    """Every site type has a constant density, and counts grow exactly as n^2 (§E.3).

    One magic number is not evidence of scaling; this asserts the density is
    constant across 1x1, 2x2 and 3x3 *and* that each count is ``n^2`` times the
    primitive cell's count.
    """
    areas = {n: in_plane_area(supercelled(pt_slab, n)) for n in SUPERCELLS}
    counts = {
        n: site_counts(find_adsorption_sites(supercelled(pt_slab, n))) for n in SUPERCELLS
    }

    assert areas[1] == pytest.approx(PT111_PRIMITIVE_AREA, abs=1e-5)
    assert areas[3] == pytest.approx(9.0 * areas[1], rel=1e-9)
    assert counts[1] == PT111_COUNTS[1]

    for site_type in SITE_TYPES:
        density = counts[1][site_type] / areas[1]
        for n in SUPERCELLS:
            assert counts[n][site_type] / areas[n] == pytest.approx(density, rel=1e-6), (
                f"{site_type} density changed in the {n}x{n} cell"
            )
            assert counts[n][site_type] == n * n * counts[1][site_type]


def test_counts_grow_instead_of_collapsing_to_symmetry_equivalents(pt_slab: Slab) -> None:
    """All sites are enumerated, so 2x2 strictly exceeds 1x1 for every type (§D3).

    Upstream's default ``symm_reduce=0.01`` collapses Pt(111) 1x1..4x4 to the same
    ``{1, 1, 2}``; that is the bug this test guards.
    """
    small = site_counts(find_adsorption_sites(supercelled(pt_slab, 1)))
    large = site_counts(find_adsorption_sites(supercelled(pt_slab, 2)))

    for site_type in SITE_TYPES:
        assert large[site_type] > small[site_type], f"{site_type} did not grow with area"


@pytest.mark.parametrize("n", SUPERCELLS)
def test_every_site_sits_at_or_above_the_slab_top_layer(pt_slab: Slab, n: int) -> None:
    """No site may land inside the slab (§E.1, §E.3)."""
    cell = supercelled(pt_slab, n)
    z_top = top_layer_z(cell)
    sites = find_adsorption_sites(cell)

    assert sites
    for site in sites:
        assert site.position[2] >= z_top - Z_NOISE_TOL, (
            f"{site.site_type} site at z={site.position[2]:.6f} is below the "
            f"top layer {z_top:.6f}"
        )


def test_sites_lie_on_the_top_layer_because_distance_is_zero(pt_slab: Slab) -> None:
    """The lifting offset is the caller's, so sites sit exactly on the top layer (§E.2).

    Upstream's ``distance`` default of 2.0 A would put every site 2 A above it, and
    ``height=2.5`` would put "ontop" 2.265 A *below* it; both fail here.
    """
    z_top = top_layer_z(pt_slab)

    assert z_top == pytest.approx(PT111_TOP_LAYER, abs=1e-6)
    for site in find_adsorption_sites(pt_slab):
        assert site.position[2] == pytest.approx(z_top, abs=Z_NOISE_TOL)


def test_site_type_strings_are_exactly_the_project_vocabulary(pt_slab: Slab) -> None:
    """Exactly ``ontop``/``bridge``/``hollow``, never an alias or ``all`` (§E.4)."""
    sites = find_adsorption_sites(supercelled(pt_slab, 2))

    assert {site.site_type for site in sites} == set(SITE_TYPES)
    assert SITE_TYPES == ("ontop", "bridge", "hollow")
    for site in sites:
        assert isinstance(site.site_type, str)
        assert site.site_type == site.site_type.strip().lower()
        assert site.site_type != "all", "upstream's concatenation key must not leak"


def test_each_site_exposes_the_cartesian_position_a_caller_can_place_at(
    pt_slab: Slab,
) -> None:
    """The return type is a list of typed records carrying a 3-D position (ticket #6)."""
    sites = find_adsorption_sites(pt_slab)

    assert sites
    for site in sites:
        assert isinstance(site, AdsorptionSite)
        assert isinstance(site.position, tuple)
        assert len(site.position) == 3
        assert all(isinstance(value, float) for value in site.position)
        assert all(math.isfinite(value) for value in site.position)


def test_an_ontop_site_sits_directly_above_a_top_layer_atom(pt_slab: Slab) -> None:
    """The positions are real: the ontop site is a top atom's periodic image (§E.3)."""
    ontop = [site for site in find_adsorption_sites(pt_slab) if site.site_type == "ontop"]

    assert len(ontop) == 1
    site_frac = pt_slab.lattice.get_fractional_coords(np.asarray(ontop[0].position))
    top_layer = group_z_levels(pt_slab)[-1]
    matches = [
        index
        for index in top_layer
        if np.allclose(
            wrapped(site_frac[:2] - np.asarray(pt_slab.frac_coords[index])[:2]),
            0.0,
            atol=1e-6,
        )
    ]

    assert len(matches) == 1, "an ontop site must be above exactly one top-layer atom"


# --- orthogonal c axis (D2 / §E.1) ----------------------------------------


def test_a_non_orthogonal_slab_is_orthogonalised_before_enumerating(
    raw_non_orthogonal_slab: Slab,
) -> None:
    """The c axis is forced orthogonal before the site search, and it is load-bearing.

    The naive call on the raw slab puts the ontop site 1.96155 A *inside* the slab
    (§D2); after the orthogonalisation the library performs, every site is back on
    the top layer of the orthogonalised frame.
    """
    raw = raw_non_orthogonal_slab
    # The prerequisite §D2 describes: the in-plane vectors carry z.
    assert raw.lattice.matrix[0][2] == pytest.approx(-1.96155, abs=1e-5)
    assert raw.lattice.matrix[1][2] == pytest.approx(-1.96155, abs=1e-5)
    assert raw.lattice.angles[0] == pytest.approx(135.0, abs=1e-6)

    z_top_raw = top_layer_z(raw)
    naive = AdsorbateSiteFinder(raw.copy()).find_adsorption_sites(
        distance=0.0, symm_reduce=-1
    )
    assert min(site[2] for site in naive["ontop"]) == pytest.approx(
        z_top_raw - 1.96155, abs=1e-3
    )
    assert min(site[2] for site in naive["ontop"]) < z_top_raw - 1.0

    sites = find_adsorption_sites(raw)

    assert site_counts(sites) == PT111_COUNTS[1]
    z_top = top_layer_z(raw.get_orthogonal_c_slab())
    for site in sites:
        assert site.position[2] >= z_top - Z_NOISE_TOL


def test_an_already_orthogonal_slab_is_unchanged_by_the_orthogonalisation(
    pt_slab: Slab,
) -> None:
    """``generate_slabs`` output is already orthogonal, so the step is a no-op (§E.1)."""
    assert pt_slab.lattice.matrix[0][2] == pytest.approx(0.0, abs=1e-12)
    assert pt_slab.lattice.matrix[1][2] == pytest.approx(0.0, abs=1e-12)

    direct = find_adsorption_sites(pt_slab)
    explicit = find_adsorption_sites(pt_slab.get_orthogonal_c_slab())

    assert [site.site_type for site in direct] == [site.site_type for site in explicit]
    assert [site.position for site in direct] == pytest.approx(
        [site.position for site in explicit], abs=1e-12
    )


def test_a_slab_loaded_as_a_plain_structure_is_enumerated_too(pt_slab: Slab) -> None:
    """A Slab read back from a file is a plain ``Structure``; it must still work."""
    plain = Structure.from_sites(list(pt_slab))

    assert not isinstance(plain, Slab)
    sites = find_adsorption_sites(plain)

    assert site_counts(sites) == PT111_COUNTS[1]
    for site in sites:
        assert site.position[2] == pytest.approx(PT111_TOP_LAYER, abs=1e-6)


# --- rejecting input that is not a Slab (§E.5 / §H) -----------------------


def test_a_bulk_crystal_is_rejected_with_the_measured_vacuum(pt_bulk: Structure) -> None:
    """Bulk Pt fails the slab-like predicate, and the message says how far off it is."""
    with pytest.raises(NotASlabError) as excinfo:
        find_adsorption_sites(pt_bulk)

    message = str(excinfo.value)

    assert message.startswith("structure is not slab-like: ")
    assert "Vacuum 1.962 A < 5.0 A" in message
    assert "Example: find_adsorption_sites(" in message
    assert "\n" not in message
    assert not message.endswith(".")
    assert len(message) <= MAX_MESSAGE_LEN
    # Catchable as the library root and as the ValueError pymatgen callers use.
    assert isinstance(excinfo.value, Dsh4sciError)
    assert isinstance(excinfo.value, ValueError)


def test_the_rejection_message_holds_the_bound_when_both_conditions_fail() -> None:
    """The longest message is the two-reason one, and it must stay under 210 chars (§H.1).

    An empty cell violates both conditions; it also proves the rejection path never
    divides by zero or indexes an empty stack.
    """
    empty = Structure(Lattice.cubic(10.0), [], [])

    with pytest.raises(NotASlabError) as excinfo:
        find_adsorption_sites(empty)

    message = str(excinfo.value)

    assert "Vacuum 0.000 A < 5.0 A" in message
    assert "only 0 z-levels < 2" in message
    assert len(message) <= MAX_MESSAGE_LEN
    assert "\n" not in message
    assert not message.endswith(".")


def test_a_second_bulk_crystal_reports_its_own_measured_vacuum(
    fixture_path: Callable[[str], Path],
) -> None:
    """The number in the message is measured, not copied from a single example."""
    cu = load_structure(fixture_path("Cu.poscar"))

    with pytest.raises(NotASlabError) as excinfo:
        find_adsorption_sites(cu)

    assert "Vacuum 1.808 A < 5.0 A" in str(excinfo.value)


def test_a_slab_whose_vacuum_is_too_thin_is_rejected_too(pt_bulk: Structure) -> None:
    """The threshold applies to real slabs as well, not only to bulk crystals (§E.5)."""
    thin = generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=8.0, vacuum=4.0)[0]

    assert vacuum_gap(thin) == pytest.approx(4.0, abs=1e-6)
    with pytest.raises(NotASlabError) as excinfo:
        find_adsorption_sites(thin)

    assert "Vacuum 4.000 A < 5.0 A" in str(excinfo.value)


def test_a_non_structure_argument_raises_type_error(
    fixture_path: Callable[[str], Path],
) -> None:
    """A Structure reference is not a Structure; upstream would raise AttributeError."""
    with pytest.raises(TypeError) as excinfo:
        find_adsorption_sites(str(fixture_path("Pt.poscar")))

    message = str(excinfo.value)

    assert message.startswith("structure=")
    assert "expected a pymatgen Structure" in message
    assert len(message) <= MAX_MESSAGE_LEN


def test_the_slab_like_predicate_accepts_the_canonical_slab(pt_slab: Slab) -> None:
    """``is_slab_like`` says so explicitly, so the predicate is testable on its own."""
    assert is_slab_like(pt_slab) == (True, [])


def test_the_slab_like_predicate_accepts_a_thin_but_real_vacuum(
    pt_bulk: Structure,
) -> None:
    """6 A of Vacuum and 4 z-levels is a Slab: the threshold is 5.0 A, not 15 A."""
    thin = generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=8.0, vacuum=6.0)[0]

    assert vacuum_gap(thin) == pytest.approx(6.0, abs=1e-6)
    assert count_z_levels(thin) == 4
    assert is_slab_like(thin) == (True, [])


@pytest.mark.parametrize("name", ["Pt.poscar", "Cu.poscar", "Si.poscar", "Cu.cif"])
def test_the_slab_like_predicate_rejects_bulk_crystals(
    fixture_path: Callable[[str], Path], name: str
) -> None:
    """Every bulk fixture fails, with a reason naming its measured Vacuum (§E.5)."""
    bulk = load_structure(fixture_path(name))

    slab_like, reasons = is_slab_like(bulk)

    assert slab_like is False
    assert reasons
    assert f"Vacuum {vacuum_gap(bulk):.3f} A < 5.0 A" in reasons[0], reasons


def test_the_documented_thresholds_are_the_measured_ones() -> None:
    """The constants behind the predicate are the ones §E.5 calibrated."""
    assert SITE_TYPES == ("ontop", "bridge", "hollow")
    assert MIN_VACUUM == 5.0
    assert MIN_Z_LEVELS == 2


# --- behaviour a caller relies on -----------------------------------------


def test_the_callers_slab_is_not_mutated(pt_slab: Slab) -> None:
    """Enumeration must not rewrite the structure it was handed.

    ``AdsorbateSiteFinder`` and ``Slab.make_supercell`` both mutate their input on
    this stack, so the site search works on an orthogonalised copy.
    """
    frac_before = np.array(pt_slab.frac_coords, copy=True)
    lattice_before = np.array(pt_slab.lattice.matrix, copy=True)
    properties_before = {
        name: list(values) for name, values in pt_slab.site_properties.items()
    }

    find_adsorption_sites(pt_slab)

    assert np.array_equal(np.asarray(pt_slab.frac_coords), frac_before)
    assert np.array_equal(np.asarray(pt_slab.lattice.matrix), lattice_before)
    assert {
        name: list(values) for name, values in pt_slab.site_properties.items()
    } == properties_before


def test_the_public_names_are_re_exported_from_the_package() -> None:
    """Ticket #5's names are part of the package API, not module-only."""
    import dsh4sci
    import dsh4sci.adsorption as adsorption

    assert dsh4sci.find_adsorption_sites is adsorption.find_adsorption_sites
    assert dsh4sci.is_slab_like is adsorption.is_slab_like
    assert dsh4sci.AdsorptionSite is adsorption.AdsorptionSite
    assert dsh4sci.SITE_TYPES == ("ontop", "bridge", "hollow")
