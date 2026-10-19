"""The workflow ``SKILL.md`` documents, executed verbatim (ticket #8).

A document that describes a five-step workflow and a library whose functions
behave differently is the failure mode this ticket is about, so the example in
``SKILL.md`` is not a separate copy of the workflow: it *is* the workflow. The
fenced block whose info string is ``python workflow`` is compiled and executed
here, from the repository root, and the objects it leaves behind are then checked
against the pinned numbers.

That means the example cannot rot silently. If a default changes, if a signature
gains a required parameter, if a site count moves, or if someone edits a number in
the document, one of these assertions fails. The numbers themselves come from
``CONVENTIONS.md`` §I and §J.4 -- they are literals here, not recomputed the way
the library computes them, so this test can disagree with the library.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import dsh4sci
from dsh4sci.geometry import count_z_levels, split_roles, vacuum_gap

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL = REPO_ROOT / "SKILL.md"

#: The one fenced block that is executed. Its info string is the contract.
_WORKFLOW_BLOCK = re.compile(
    r"^```python workflow[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL
)

#: The order the document promises: load, slabs, sites, place, validate. Each
#: function is looked up as a *call* (`name(`), so the import line does not count.
_WORKFLOW_ORDER = (
    "load_structure",
    "generate_slabs",
    "termination_fingerprint",
    "fix_bottom_atoms",
    "find_adsorption_sites",
    "place_adsorbate",
    "check_structure",
    "all_passed",
)


def _workflow_source() -> str:
    """The documented workflow's source, exactly as the file contains it."""
    blocks = _WORKFLOW_BLOCK.findall(SKILL.read_text(encoding="utf-8"))
    assert len(blocks) == 1, (
        f"SKILL.md must contain exactly one ```python workflow block (found "
        f"{len(blocks)}); it is the workflow the tests execute"
    )
    return blocks[0]


def _run_workflow(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Execute the documented workflow and return the namespace it populated.

    Run from the repository root: the example refers to a checked-in fixture by a
    repository-relative path, which is how a reader would run it from a checkout.
    """
    monkeypatch.chdir(REPO_ROOT)
    namespace: dict[str, Any] = {"__name__": "dsh4sci_skill_workflow"}
    source = _workflow_source()
    # The compiled text is this repository's own document; nothing external runs.
    exec(compile(source, str(SKILL), "exec"), namespace)
    return namespace


def test_the_workflow_block_covers_every_documented_step_in_order() -> None:
    """The runnable example is the whole workflow, not a fragment of it.

    Order, not mere presence: the five steps are load → generate → enumerate →
    place → validate, and an example that validated before placing would pass the
    numeric assertions while teaching the wrong sequence.
    """
    source = _workflow_source()

    positions: list[int] = []
    for name in _WORKFLOW_ORDER:
        match = re.search(rf"\b{re.escape(name)}\(", source)
        assert match is not None, (
            f"the ```python workflow block never calls {name}(); the documented "
            f"workflow is load → slabs → sites → place → validate"
        )
        positions.append(match.start())

    assert positions == sorted(positions), (
        f"the workflow calls its steps out of order: {list(zip(_WORKFLOW_ORDER, positions))}"
    )


def test_the_workflow_is_documented_as_executable() -> None:
    """A reader (and a future editor) must be told the block is run, not just shown."""
    text = SKILL.read_text(encoding="utf-8")

    assert "python workflow" in text
    assert "tests/test_end_to_end.py" in text


def test_the_documented_workflow_produces_the_pinned_slab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pt(111) at 10 Å / 15 Å is one Slab, Pt5, five z-levels, Vacuum 15.000000."""
    namespace = _run_workflow(monkeypatch)

    bulk = namespace["bulk"]
    slabs = namespace["slabs"]
    slab = namespace["slab"]
    fingerprint = namespace["fingerprint"]

    # The conventional cubic Pt cell is Pt4 (CONVENTIONS.md §I).
    assert len(bulk) == 4

    # One Termination at (1, 1, 1): Pt5, five distinct z-levels.
    assert len(slabs) == 1
    assert len(slab) == 5
    assert {site.specie.symbol for site in slab} == {"Pt"}
    assert count_z_levels(slab) == 5

    # The invariants are c, the Vacuum and the spacing -- never an absolute z
    # (§J.6): the stack is centred symmetrically, so the top layer is read from
    # the structure, not assumed.
    assert vacuum_gap(slab) == pytest.approx(15.0, abs=1e-9)
    assert slab.lattice.c == pytest.approx(24.060011364, abs=1e-9)
    assert slab.lattice.angles == pytest.approx((90.0, 90.0, 60.0), abs=1e-6)

    stack_z = np.sort(np.asarray(slab.cart_coords, dtype=float)[:, 2])
    assert np.diff(stack_z) == pytest.approx([2.265003] * 4, abs=1e-5)

    # The Termination is identified by content, not by list position.
    assert str(fingerprint) == "(1, 1, 1):Pt1/Pt1:shift=500000e-6"
    assert fingerprint.sha() == "1060cd6a0cc2325f"


def test_the_documented_workflow_fixes_the_bottom_and_enumerates_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fraction=0.5`` fixes 2 of 5 atoms; the primitive cell has 1/3/2 sites."""
    namespace = _run_workflow(monkeypatch)

    slab = namespace["slab"]
    sites = namespace["sites"]

    flags = slab.site_properties["selective_dynamics"]
    assert flags[:2] == [(False, False, False)] * 2
    assert flags[2:] == [(True, True, True)] * 3

    # §I / §E.3: the primitive Pt(111) cell has 1 ontop, 3 bridge and 2 hollow
    # sites -- and the counts are not symmetry-reduced, which is why they scale
    # with area instead of collapsing to 1/1/2.
    assert len(sites) == 6
    assert Counter(site.site_type for site in sites) == {
        "ontop": 1,
        "bridge": 3,
        "hollow": 2,
    }
    assert [site.site_type for site in sites] == [
        "ontop",
        "bridge",
        "bridge",
        "bridge",
        "hollow",
        "hollow",
    ]


def test_the_documented_workflow_places_co_with_carbon_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C binds 1.850000 Å above the top layer, O sits 1.15034 Å above C (§F.3)."""
    namespace = _run_workflow(monkeypatch)

    placed = namespace["placed"]
    structure = placed.structure
    cart = np.asarray(structure.cart_coords, dtype=float)

    assert placed.n_slab_atoms == 5
    assert len(structure) == 7
    # ASE's molecule("CO") is ordered ['O', 'C'] -- so this also proves the
    # library anchored by symbol rather than by position in the list (§J.10).
    assert [site.specie.symbol for site in structure] == ["Pt"] * 5 + ["O", "C"]

    anchor = placed.anchor_index
    assert structure[anchor].specie.symbol == "C"

    top_z = float(np.max(cart[: placed.n_slab_atoms, 2]))
    assert cart[anchor, 2] - top_z == pytest.approx(1.85, abs=1e-9)
    closest_pt = min(
        structure.distance_matrix[anchor, index]
        for index in range(placed.n_slab_atoms)
    )
    assert closest_pt == pytest.approx(1.85, abs=1e-9)

    oxygen = next(
        index
        for index in range(placed.n_slab_atoms, len(structure))
        if structure[index].specie.symbol == "O"
    )
    assert cart[oxygen, 2] > cart[anchor, 2]
    assert cart[oxygen, 2] - cart[anchor, 2] == pytest.approx(1.15034, abs=1e-4)

    # The role tag separates the two in memory, and the Slab's selective dynamics
    # survives placement with both Adsorbate atoms left free.
    slab_indices, adsorbate_indices = split_roles(structure)
    assert slab_indices == [0, 1, 2, 3, 4]
    assert adsorbate_indices == [5, 6]
    combined = structure.site_properties["selective_dynamics"]
    assert combined[:2] == [(False, False, False)] * 2
    assert combined[2:] == [(True, True, True)] * 5


def test_the_documented_workflow_passes_its_own_self_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three checks, in report order, all passing -- including the anchor check.

    The workflow's final statement is ``assert all_passed(results)``; asserting it
    again here proves the *documented* sequence reached that state, rather than the
    test having built its own.
    """
    namespace = _run_workflow(monkeypatch)

    results = namespace["results"]

    assert [result.name for result in results] == [
        "minimum_interatomic_distance",
        "adsorbate_anchor_above_surface",
        "lattice_angles",
    ]
    assert [result.passed for result in results] == [True, True, True]
    assert dsh4sci.all_passed(results) is True

    # The closest pair is the intramolecular C-O bond, well above the 0.5 A floor.
    assert "1.1503" in results[0].detail


def test_the_documented_workflow_only_names_public_api() -> None:
    """The example is importable by a reader: every name it imports is exported.

    A snippet that reaches into ``dsh4sci.slabs`` would document an internal
    surface; the package root is the public one.
    """
    source = _workflow_source()

    imported = re.search(
        r"from dsh4sci import \((?P<names>.*?)\)", source, re.DOTALL
    )
    assert imported is not None, "the workflow must import from the package root"

    names = [
        name.strip().rstrip(",")
        for name in imported.group("names").splitlines()
        if name.strip()
    ]
    assert names, "the workflow imports nothing"
    for name in names:
        assert hasattr(dsh4sci, name), (
            f"the workflow imports {name!r}, which dsh4sci does not export"
        )
