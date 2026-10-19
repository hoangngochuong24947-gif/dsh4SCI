"""Tests for ``dsh4sci.slabs``: Slab generation and the Termination fingerprint.

The expected values are the pinned measurements in CONVENTIONS.md §C.2 and §I,
restated as literals so these tests can disagree with the implementation:

``Pt(111)``, conventional cell, ``min_slab_size=10``, ``vacuum=15``, then
``get_orthogonal_c_slab()`` + exact Vacuum:

- exactly **1** Slab, composition ``Pt5``, 5 sites, 5 distinct z-levels;
- interlayer spacing **2.265003 A** (``a / sqrt(3)``);
- final ``c = 24.060011364 A``, Vacuum **15.000000000 A** (error 3.55e-15);
- z-levels 7.500000 9.765003 12.030006 14.295009 16.560011.

``SiC(111)`` with the same arguments: **2** Slabs, each ``C4 Si4`` / 8 sites /
8 z-levels, shifts 0.375 and 0.875, final ``c`` 23.181053314 and 24.439676901.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable

import numpy as np
import pytest
from pymatgen.core import Structure
from pymatgen.core.surface import Slab

from dsh4sci.errors import InvalidMillerIndexError, InvalidParameterError
from dsh4sci.geometry import count_z_levels, vacuum_gap
from dsh4sci.loading import load_structure
from dsh4sci.slabs import (
    TerminationFingerprint,
    generate_slabs,
    termination_fingerprint,
)

# --- pinned numbers, CONVENTIONS.md §C.2 / §I -------------------------------

PT111_C = 24.060011364  # final c, A
PT111_D111 = 2.265003  # a / sqrt(3), the (111) interlayer spacing, A
PT111_AREA = 6.664374  # in-plane area of the primitive cell, A^2
PT111_Z_LEVELS = (7.5, 9.765003, 12.030006, 14.295009, 16.560011)
RHOMBUS_EDGE = 2.774051  # a / sqrt(2): what the primitive input cell yields

SIC111_C = (23.181053314, 24.439676901)
SIC111_SHIFT = (0.375, 0.875)
SIC111_AREA = 8.231398

# The measured fingerprint digests (§C.5). They pin the representation itself:
# a change to __str__ or to the quantisation changes these.
PT111_SHA = "1060cd6a0cc2325f"
SIC111_SHA = ("83b9a6ea38b16a48", "a972bed42046ed5a")

# --- fixtures (ticket-local: conftest.py belongs to ticket #2) --------------


@pytest.fixture
def sic_bulk(fixture_path: Callable[[str], object]) -> Structure:
    """Conventional 3C zincblende SiC (``Si4 C4``, a = 4.36 A)."""
    return load_structure(fixture_path("SiC.poscar"))


@pytest.fixture
def pt_slabs(pt_bulk: Structure) -> list[Slab]:
    """The canonical Pt(111) Slab list: 1 Slab, ``Pt5``, exact 15 A Vacuum."""
    return generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=10.0, vacuum=15.0)


@pytest.fixture
def sic_slabs(sic_bulk: Structure) -> list[Slab]:
    """The canonical SiC(111) Slab list: 2 Terminations."""
    return generate_slabs(sic_bulk, (1, 1, 1), min_slab_size=10.0, vacuum=15.0)


# --- generate_slabs: the pinned Pt(111) case -------------------------------


def test_pt111_yields_exactly_one_slab(pt_slabs: list[Slab]) -> None:
    """Pt(111) at 10 A is a single-Termination surface: 1 Slab, ``Pt5``, 5 sites."""
    assert len(pt_slabs) == 1
    slab = pt_slabs[0]
    assert isinstance(slab, Slab)
    assert str(slab.composition) == "Pt5"
    assert len(slab) == 5


def test_pt111_has_five_distinct_z_levels(pt_slabs: list[Slab]) -> None:
    """5 atoms in 5 layers, at the pinned z-levels, 2.265003 A apart."""
    slab = pt_slabs[0]

    assert count_z_levels(slab) == 5
    z = np.sort(np.asarray(slab.cart_coords, dtype=float)[:, 2])
    assert z == pytest.approx(PT111_Z_LEVELS, abs=1e-6)
    assert np.diff(z) == pytest.approx([PT111_D111] * 4, abs=1e-6)


def test_pt111_interlayer_spacing_is_d111_not_the_rhombus_edge(
    pt_slabs: list[Slab],
) -> None:
    """The conventional input cell is what makes this ``a/sqrt(3)`` (§C.1).

    The primitive rhombohedral cell produces 2.774051 A instead -- the in-plane
    rhombus edge -- which is the wrong physics. Asserting the value, and that it is
    *not* the rhombus edge, pins the choice of input cell.
    """
    z = np.sort(np.asarray(pt_slabs[0].cart_coords, dtype=float)[:, 2])
    spacing = float(np.mean(np.diff(z)))

    assert spacing == pytest.approx(PT111_D111, abs=1e-6)
    assert abs(spacing - RHOMBUS_EDGE) > 0.1


def test_pt111_has_an_orthogonal_c_axis(pt_slabs: list[Slab]) -> None:
    """``get_orthogonal_c_slab()`` is mandatory, not cosmetic (§C.1, §D2).

    Without it the in-plane vectors carry a z component and later supercell work
    smears the stack. ``alpha`` and ``beta`` are the angles between c and the
    in-plane vectors; both must be 90 degrees.
    """
    lattice = pt_slabs[0].lattice
    alpha, beta, _gamma = lattice.angles

    assert alpha == pytest.approx(90.0, abs=1e-9)
    assert beta == pytest.approx(90.0, abs=1e-9)
    # The c-axis is the third lattice vector and lies along +z.
    matrix = np.asarray(lattice.matrix, dtype=float)
    assert matrix[2][0] == pytest.approx(0.0, abs=1e-9)
    assert matrix[2][1] == pytest.approx(0.0, abs=1e-9)
    assert matrix[2][2] == pytest.approx(float(lattice.c), abs=1e-9)
    # And the in-plane vectors carry no z, which is what stops the smear.
    assert matrix[0][2] == pytest.approx(0.0, abs=1e-6)
    assert matrix[1][2] == pytest.approx(0.0, abs=1e-6)


def test_pt111_carries_the_pinned_cell_numbers(pt_slabs: list[Slab]) -> None:
    """Final ``c`` and in-plane area, as measured in §C.2."""
    slab = pt_slabs[0]

    assert float(slab.lattice.c) == pytest.approx(PT111_C, abs=1e-6)
    assert float(slab.surface_area) == pytest.approx(PT111_AREA, abs=1e-6)
    assert slab.miller_index == (1, 1, 1)


# --- Vacuum: exact, symmetric, and the ticket's 0.01 A bound --------------


def test_vacuum_equals_the_request_within_the_ticket_tolerance(
    pt_slabs: list[Slab],
) -> None:
    """The ticket's acceptance criterion: gap == requested +/- 0.01 A."""
    assert vacuum_gap(pt_slabs[0]) == pytest.approx(15.0, abs=0.01)


def test_vacuum_is_exact_to_float_precision(pt_slabs: list[Slab]) -> None:
    """The library's tighter bound: 3.55e-15 A, not merely 1e-2 (§C.3).

    ``min_vacuum_size`` is a lower bound and the generator padded to 18.120 A after
    orthogonalisation, so reaching 15 A at all is the rescale's doing.
    """
    gap = vacuum_gap(pt_slabs[0])

    assert gap == pytest.approx(15.0, abs=1e-12)
    assert abs(gap - 15.0) < 1e-13


def test_vacuum_is_split_equally_above_and_below(pt_slabs: list[Slab]) -> None:
    """``center_slab=True`` plus the re-centring in ``set_vacuum`` (§C.3)."""
    slab = pt_slabs[0]
    z = np.asarray(slab.cart_coords, dtype=float)[:, 2]
    c = float(slab.lattice.c)

    assert float(z.min()) == pytest.approx(7.5, abs=1e-9)
    assert c - float(z.max()) == pytest.approx(7.5, abs=1e-9)


def test_generation_does_not_mutate_the_callers_bulk(pt_bulk: Structure) -> None:
    """The caller's bulk is not mutated by generation (§C.1 mutation warning).

    ``SlabGenerator`` adds ``bulk_wyckoff`` / ``bulk_equivalent`` to the Structure
    it is handed. The library copies first, so a caller can reuse the bulk.
    """
    before = list(pt_bulk.site_properties)

    generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=10.0, vacuum=15.0)

    assert list(pt_bulk.site_properties) == before


@pytest.mark.parametrize("vacuum", [5.0, 10.0, 12.5, 20.0])
def test_any_requested_vacuum_is_applied_exactly(
    pt_bulk: Structure, vacuum: float
) -> None:
    """The rescale is general, not tuned to 15 A."""
    slab = generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=10.0, vacuum=vacuum)[0]

    assert vacuum_gap(slab) == pytest.approx(vacuum, abs=1e-12)
    z = np.asarray(slab.cart_coords, dtype=float)[:, 2]
    assert float(z.min()) == pytest.approx(vacuum / 2.0, abs=1e-9)
    assert float(slab.lattice.c) - float(z.max()) == pytest.approx(
        vacuum / 2.0, abs=1e-9
    )


def test_min_slab_size_controls_the_layer_count(pt_bulk: Structure) -> None:
    """``min_slab_size`` is a lower bound on thickness, measured in layers (§D4).

    8 A -> ``Pt4``, 10 A -> ``Pt5``, 13 A -> ``Pt6``: the generator rounds up to
    whole atomic layers, so the atom count follows the request.
    """
    counts = {
        request: len(
            generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=request, vacuum=15.0)[0]
        )
        for request in (8.0, 10.0, 13.0)
    }

    assert counts == {8.0: 4, 10.0: 5, 13.0: 6}


# --- generate_slabs: the SiC(111) multi-Termination case ------------------


def test_sic111_yields_two_slabs(sic_slabs: list[Slab]) -> None:
    """A compound surface has several Terminations -- the case the fingerprint serves."""
    assert len(sic_slabs) >= 2
    assert len(sic_slabs) == 2


def test_sic111_slabs_carry_the_pinned_numbers(sic_slabs: list[Slab]) -> None:
    """Each is ``C4 Si4`` / 8 sites / 8 z-levels, with the measured ``c`` and area."""
    assert [str(slab.composition) for slab in sic_slabs] == ["C4 Si4", "C4 Si4"]
    assert [len(slab) for slab in sic_slabs] == [8, 8]
    assert [count_z_levels(slab) for slab in sic_slabs] == [8, 8]
    assert [float(slab.lattice.c) for slab in sic_slabs] == pytest.approx(
        SIC111_C, abs=1e-6
    )
    assert [float(slab.surface_area) for slab in sic_slabs] == pytest.approx(
        [SIC111_AREA, SIC111_AREA], abs=1e-6
    )


def test_sic111_z_levels_match_the_pinned_table(sic_slabs: list[Slab]) -> None:
    """The §C.2 z-levels, verbatim, for both Terminations.

    Pinning the absolute levels (rather than only the gaps) is what proves the two
    SiC Terminations are genuinely different cuts: they share four z-levels and
    differ on the other four, and the two slabs have different raw ``c`` because
    the cut lands differently.
    """
    expected = (
        (7.5, 8.129312, 10.017247, 10.646559, 12.534494, 13.163806, 15.051742, 15.681053),
        (7.5, 9.387935, 10.017247, 11.905183, 12.534494, 14.42243, 15.051742, 16.939677),
    )
    for slab, levels in zip(sic_slabs, expected):
        z = np.sort(np.asarray(slab.cart_coords, dtype=float)[:, 2])
        assert z == pytest.approx(levels, abs=1e-6)


def test_sic111_layers_alternate_bilayer_and_interlayer_gaps(
    sic_slabs: list[Slab],
) -> None:
    """The Si-C bilayer gap and the C-Si inter-bilayer gap alternate.

    0.629312 + 1.887935 = 2.517247 A = d(111) = a/sqrt(3) for SiC at a = 4.36 A,
    so the pair is one (111) repeat. Slab[0] opens and closes on the short bilayer
    gap (it starts on a C and ends on a Si); slab[1] is the reverse.
    """
    expected_first = [0.629312, 1.887935, 0.629312, 1.887935, 0.629312, 1.887935, 0.629312]
    expected_second = [1.887935, 0.629312, 1.887935, 0.629312, 1.887935, 0.629312, 1.887935]

    for slab, expected in zip(sic_slabs, (expected_first, expected_second)):
        gaps = np.diff(np.sort(np.asarray(slab.cart_coords, dtype=float)[:, 2]))
        assert gaps == pytest.approx(expected, abs=1e-6)
        # The bilayer pair is one (111) repeat, which is the physics check.
        assert float(gaps[0] + gaps[1]) == pytest.approx(2.517247, abs=1e-6)


def test_sic111_reaches_the_exact_vacuum_too(sic_slabs: list[Slab]) -> None:
    """Both raw ``c`` values were different, and both still land on 15 A exactly."""
    for slab in sic_slabs:
        assert vacuum_gap(slab) == pytest.approx(15.0, abs=1e-12)


# --- fingerprints ---------------------------------------------------------


def test_fingerprints_of_the_sic_terminations_are_all_distinct(
    sic_slabs: list[Slab],
) -> None:
    """The ticket's acceptance criterion, and the reason the fingerprint exists."""
    fingerprints = [termination_fingerprint(slab) for slab in sic_slabs]

    assert all(isinstance(f, TerminationFingerprint) for f in fingerprints)
    assert len(set(fingerprints)) == len(fingerprints)


def test_fingerprints_record_both_faces_and_the_shift(
    sic_slabs: list[Slab], pt_slabs: list[Slab]
) -> None:
    """The fingerprint holds the terminating-layer composition and the offset.

    SiC(111) cuts end on different elements at the two faces (``Si1``/``C1`` and
    its reverse) and differ by 0.5 in ``shift``. The Pt(111) cut is
    ``Pt1``/``Pt1`` at shift 0.5.
    """
    sic_fingerprints = [termination_fingerprint(slab) for slab in sic_slabs]
    pt_fingerprint = termination_fingerprint(pt_slabs[0])

    assert [f.top_layer for f in sic_fingerprints] == [
        (("Si", 1),),
        (("C", 1),),
    ]
    assert [f.bottom_layer for f in sic_fingerprints] == [
        (("C", 1),),
        (("Si", 1),),
    ]
    assert [f.shift for f in sic_fingerprints] == pytest.approx(SIC111_SHIFT, abs=1e-9)
    assert [f.miller_index for f in sic_fingerprints] == [(1, 1, 1), (1, 1, 1)]

    assert pt_fingerprint.top_layer == (("Pt", 1),)
    assert pt_fingerprint.bottom_layer == (("Pt", 1),)
    assert pt_fingerprint.shift == pytest.approx(0.5, abs=1e-9)
    assert pt_fingerprint.miller_index == (1, 1, 1)


def test_fingerprint_quantises_the_raw_shift(
    pt_slabs: list[Slab], sic_slabs: list[Slab]
) -> None:
    """pymatgen returns ``shift`` as unrounded float arithmetic; we quantise it.

    The raw values are ``0.4999999999999998``, ``0.37500000000000006`` and
    ``0.875``. Comparing them exactly would make the identity depend on float
    noise, so the fingerprint stores ``round(shift * 10**6)`` as an integer and
    exposes the de-quantised value.
    """
    raw = [float(pt_slabs[0].shift)] + [float(slab.shift) for slab in sic_slabs]
    # The raw values really are the noisy ones the quantisation exists to collapse.
    assert raw[0] != 0.5
    assert abs(raw[0] - 0.5) < 1e-12
    assert raw[1] != 0.375

    fingerprints = [
        termination_fingerprint(slab) for slab in [pt_slabs[0], *sic_slabs]
    ]
    assert [f.shift_ndigits for f in fingerprints] == [500000, 375000, 875000]
    assert [f.shift for f in fingerprints] == pytest.approx([0.5, 0.375, 0.875])


def test_fingerprint_str_and_sha_match_the_pinned_values(
    pt_slabs: list[Slab], sic_slabs: list[Slab]
) -> None:
    """§C.5's verified outputs, verbatim -- the representation is a contract."""
    pt_fingerprint = termination_fingerprint(pt_slabs[0])
    sic_fingerprints = [termination_fingerprint(slab) for slab in sic_slabs]

    assert str(pt_fingerprint) == "(1, 1, 1):Pt1/Pt1:shift=500000e-6"
    assert [str(f) for f in sic_fingerprints] == [
        "(1, 1, 1):Si1/C1:shift=375000e-6",
        "(1, 1, 1):C1/Si1:shift=875000e-6",
    ]
    assert pt_fingerprint.sha() == PT111_SHA
    assert tuple(f.sha() for f in sic_fingerprints) == SIC111_SHA


def test_fingerprint_is_frozen_hashable_and_order_comparable(
    sic_slabs: list[Slab],
) -> None:
    """It must work as a dict key, in a set, and in ``sorted`` -- that is the point."""
    first, second = (termination_fingerprint(slab) for slab in sic_slabs)

    assert hash(first) == hash(termination_fingerprint(sic_slabs[0]))
    assert len({first, second}) == 2
    assert {first: "a", second: "b"}[first] == "a"
    assert first != second
    assert sorted([second, first]) == sorted([first, second])

    with pytest.raises(AttributeError):
        first.shift_ndigits = 1


def test_the_same_call_twice_gives_identical_fingerprints(
    pt_bulk: Structure, sic_bulk: Structure
) -> None:
    """The ticket's acceptance criterion: the call is deterministic."""
    first = [
        termination_fingerprint(slab).sha()
        for slab in generate_slabs(sic_bulk, (1, 1, 1), 10.0, 15.0)
    ]
    second = [
        termination_fingerprint(slab).sha()
        for slab in generate_slabs(sic_bulk, (1, 1, 1), 10.0, 15.0)
    ]

    assert first == second
    assert len(set(first)) == len(first)

    pt_first = termination_fingerprint(generate_slabs(pt_bulk, (1, 1, 1), 10.0, 15.0)[0])
    pt_second = termination_fingerprint(
        generate_slabs(pt_bulk, (1, 1, 1), 10.0, 15.0)[0]
    )
    assert pt_first == pt_second


def test_fingerprints_are_stable_across_processes(
    fixture_path: Callable[[str], object],
) -> None:
    """A fingerprint must not depend on ``PYTHONHASHSEED`` (§C.5).

    This is the test that fails if the digest is taken over a ``set`` of element
    names: set iteration order is randomised per process by the hash seed, so the
    "same" fingerprint would hash differently in every run. Hashing the built
    string is what makes the identity reproducible across processes, not just
    within one.
    """
    script = textwrap.dedent(
        """
        from dsh4sci.loading import load_structure
        from dsh4sci.slabs import generate_slabs, termination_fingerprint

        sic = load_structure({sic_path!r})
        for slab in generate_slabs(sic, (1, 1, 1), 10.0, 15.0):
            fingerprint = termination_fingerprint(slab)
            print(fingerprint.sha(), str(fingerprint))
        """
    ).format(sic_path=str(fixture_path("SiC.poscar")))

    digests: list[list[str]] = []
    for seed in ("0", "7", "14", "21"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        digests.append(completed.stdout.splitlines())

    assert len(digests) == 4
    assert all(run == digests[0] for run in digests)
    assert [line.split()[0] for line in digests[0]] == list(SIC111_SHA)


def test_fingerprint_of_a_copied_slab_is_identical(pt_slabs: list[Slab]) -> None:
    """The fingerprint depends on the Slab only -- no generator state is needed."""
    slab = pt_slabs[0]

    assert termination_fingerprint(slab) == termination_fingerprint(slab.copy())


# --- validation: the library owns these, pymatgen leaks them (§C.6, §H.2) --


@pytest.mark.parametrize(
    "bad_index",
    [(0, 0, 0), (0, 0), (1, 1), (1, 1, 1, 1), (1.5, 1, 1), ("a", 1, 1), 111],
)
def test_malformed_miller_index_is_rejected_before_pymatgen(
    pt_bulk: Structure, bad_index: object
) -> None:
    """Upstream raises ``ZeroDivisionError`` / ``TypeError`` / unaligned shapes.

    None of those name the parameter, the valid set, or a valid call, so the
    library validates first and raises its own error (§C.6).
    """
    with pytest.raises(InvalidMillerIndexError) as excinfo:
        generate_slabs(pt_bulk, bad_index, min_slab_size=10.0, vacuum=15.0)

    message = str(excinfo.value)
    assert "miller_index=" in message
    assert "is invalid" in message
    assert "Example: " in message
    assert "\n" not in message
    assert len(message) < 260
    # A Dsh4sciError *and* a ValueError, so existing `except ValueError` still works.
    assert isinstance(excinfo.value, ValueError)


def test_zero_miller_index_message_states_the_rule(pt_bulk: Structure) -> None:
    """§H.2's exact wording: name the rule broken, not the crash."""
    with pytest.raises(InvalidMillerIndexError) as excinfo:
        generate_slabs(pt_bulk, (0, 0, 0))

    message = str(excinfo.value)
    assert "miller_index=(0, 0, 0)" in message
    assert "at least one index must be non-zero" in message
    assert "ZeroDivisionError" not in message


def test_wrong_arity_message_counts_the_indices(pt_bulk: Structure) -> None:
    """§H.2: echo how many indices were given, so the caller sees what to fix."""
    with pytest.raises(InvalidMillerIndexError) as excinfo:
        generate_slabs(pt_bulk, (1, 1))

    message = str(excinfo.value)
    assert "miller_index=(1, 1)" in message
    assert "expected exactly 3 indices, got 2" in message


def test_float_index_message_names_the_float(pt_bulk: Structure) -> None:
    """§H.2: ``indices must be integers, got float 1.5``."""
    with pytest.raises(InvalidMillerIndexError) as excinfo:
        generate_slabs(pt_bulk, (1.5, 1, 1))

    message = str(excinfo.value)
    assert "indices must be integers" in message
    assert "1.5" in message


def test_a_negative_miller_index_is_accepted(pt_bulk: Structure) -> None:
    """A negative index names the same plane family, so it must not be rejected."""
    slabs = generate_slabs(pt_bulk, (-1, -1, -1), min_slab_size=10.0, vacuum=15.0)

    assert len(slabs) == 1
    assert str(slabs[0].composition) == "Pt5"


@pytest.mark.parametrize("bad", [0.0, -1.0, -10.0])
def test_non_positive_min_slab_size_is_rejected(
    pt_bulk: Structure, bad: float
) -> None:
    """Upstream leaks ``ValueError: min() iterable argument is empty``."""
    with pytest.raises(InvalidParameterError) as excinfo:
        generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=bad, vacuum=15.0)

    message = str(excinfo.value)
    assert "min_slab_size=" in message
    assert "is invalid" in message
    assert "must be > 0" in message
    assert "Example: " in message
    assert "\n" not in message
    assert "min() iterable" not in message


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_non_positive_vacuum_is_rejected(pt_bulk: Structure, bad: float) -> None:
    """pymatgen silently accepts ``vacuum=0.0`` and ``vacuum=-5.0`` (§H.2)."""
    with pytest.raises(InvalidParameterError) as excinfo:
        generate_slabs(pt_bulk, (1, 1, 1), min_slab_size=10.0, vacuum=bad)

    message = str(excinfo.value)
    assert "vacuum=" in message
    assert "is invalid" in message
    assert "must be > 0" in message
    assert "Example: " in message
    assert "\n" not in message


def test_validation_runs_before_pymatgen_is_called(pt_bulk: Structure) -> None:
    """An invalid Miller index must be reported even when other arguments are bad.

    Ordering is a contract (§C.6): validate the Miller index first, so a caller
    fixing one problem at a time sees the most specific one, and no pymatgen code
    runs on rejected input.
    """
    with pytest.raises(InvalidMillerIndexError):
        generate_slabs(pt_bulk, (0, 0, 0), min_slab_size=0.0, vacuum=-1.0)
