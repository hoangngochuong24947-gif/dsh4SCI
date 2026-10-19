"""Every refusal names the parameter, the value and the valid range (ticket #8).

An error message is part of the agent surface: a caller that only learns "invalid"
has to guess, so it typically guesses by writing another script. ``CONVENTIONS.md``
§H therefore fixes one shape, and these tests treat it as a contract rather than a
style guide:

    <parameter>=<rejected value> is invalid: <why>. Valid: <range or set>. Example: <call>

The assertions below are deliberately independent of the wording around them: each
failure mode pins the parameter name (which must be the name in the signature), a
substring that can only come from the *rejected value*, a substring that names the
concrete valid range or set, and the pasteable example. Nothing asserts a whole
message, so a message may be improved without rewriting the suite -- but it cannot
lose one of those four facts, and it cannot grow past the single-line bound.

The bound is **210 characters**. It is not a preference: §H.2's own worked example
for an unparseable file is 202 characters, so a shorter cap would contradict the
conventions document. Two messages failed the convention when this ticket started
— one was 239 characters, and one gave a valid *set* where the range is a pair of
indices — and both were fixed by editing the message string only, with no change
to an exception type, a return value or a threshold.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest
from pymatgen.core import Structure
from pymatgen.core.surface import Slab

from dsh4sci.adsorption import (
    AdsorptionSite,
    PlacedAdsorbate,
    find_adsorption_sites,
    place_adsorbate,
)
from dsh4sci.constraints import fix_bottom_atoms
from dsh4sci.errors import (
    Dsh4sciError,
    InvalidMillerIndexError,
    InvalidParameterError,
    NotASlabError,
    StructureLoadError,
    UnsupportedFormatError,
)
from dsh4sci.geometry import set_vacuum
from dsh4sci.loading import load_structure
from dsh4sci.slabs import generate_slabs
from dsh4sci.validation import (
    check_adsorbate_anchor_above_surface,
    check_min_distance,
    check_structure,
)

#: §H.1 rule 7, corrected for §H.2's own 202-character example: 210, not 200.
MAX_MESSAGE_LENGTH = 210

#: A path that cannot exist, used so the echo of the rejected value is stable.
MISSING = "/nonexistent/dsh4sci/Pt.poscar"


@dataclass
class Bench:
    """The objects every failure mode is provoked against, built once per test."""

    bulk: Structure
    slab: Slab
    sites: list[AdsorptionSite]
    placed: PlacedAdsorbate


@pytest.fixture
def bench(fixture_path: Callable[[str], Path]) -> Bench:
    """Load the Pt fixture and walk it to a validated CO/Pt(111) structure.

    Function-scoped like every other structure fixture, because ``SlabGenerator``
    mutates the bulk it is handed and a leaked mutation between tests is exactly
    the class of bug this library exists to remove.
    """
    bulk = load_structure(fixture_path("Pt.poscar"))
    slab = generate_slabs(bulk, (1, 1, 1))[0]
    sites = find_adsorption_sites(slab)
    placed = place_adsorbate(slab, sites[0], molecule="CO", anchor="C")
    return Bench(bulk=bulk, slab=slab, sites=sites, placed=placed)


@dataclass(frozen=True)
class Refusal:
    """One failure mode and the four facts its message is required to carry."""

    id: str
    error: type[BaseException]
    call: Callable[[Bench], object]
    parameter: str
    rejected: tuple[str, ...]
    valid: tuple[str, ...]


# The failure modes of the library's input validation, one row each. `rejected`
# holds substrings only the rejected *value* can produce; `valid` holds the
# concrete range or set. Both are the ticket's acceptance criterion, not prose.
REFUSALS: tuple[Refusal, ...] = (
    Refusal(
        id="missing-file",
        error=FileNotFoundError,
        call=lambda bench: load_structure(MISSING),
        parameter="structure_path",
        rejected=(f"'{MISSING}'", "does not exist"),
        valid=("['.poscar', '.vasp', '.cif']",),
    ),
    Refusal(
        id="unsupported-suffix",
        error=UnsupportedFormatError,
        call=lambda bench: load_structure("/nonexistent/dsh4sci/Pt.xyz"),
        parameter="structure_path",
        rejected=("'/nonexistent/dsh4sci/Pt.xyz'", "'.xyz'"),
        valid=("['.poscar', '.vasp', '.cif']",),
    ),
    Refusal(
        id="miller-index-zero",
        error=InvalidMillerIndexError,
        call=lambda bench: generate_slabs(bench.bulk, (0, 0, 0)),
        parameter="miller_index",
        rejected=("(0, 0, 0)", "at least one index must be non-zero"),
        valid=("three integers not all zero", "(1, 1, 1)"),
    ),
    Refusal(
        id="miller-index-arity",
        error=InvalidMillerIndexError,
        call=lambda bench: generate_slabs(bench.bulk, (1, 1)),
        parameter="miller_index",
        rejected=("(1, 1)", "got 2"),
        valid=("three integers not all zero",),
    ),
    Refusal(
        id="miller-index-type",
        error=InvalidMillerIndexError,
        call=lambda bench: generate_slabs(bench.bulk, (1.5, 1, 1)),
        parameter="miller_index",
        rejected=("(1.5, 1, 1)", "float 1.5"),
        valid=("three integers not all zero",),
    ),
    Refusal(
        id="min-slab-size",
        error=InvalidParameterError,
        call=lambda bench: generate_slabs(bench.bulk, (1, 1, 1), min_slab_size=0.0),
        parameter="min_slab_size",
        rejected=("min_slab_size=0.0",),
        valid=("must be > 0", "typically >= 10.0"),
    ),
    Refusal(
        id="vacuum",
        error=InvalidParameterError,
        call=lambda bench: generate_slabs(bench.bulk, (1, 1, 1), vacuum=-5.0),
        parameter="vacuum",
        rejected=("vacuum=-5.0",),
        valid=("must be > 0", "typically >= 5.0"),
    ),
    Refusal(
        id="vacuum-below-interlayer-gap",
        error=InvalidParameterError,
        call=lambda bench: set_vacuum(bench.slab, 1.0),
        parameter="vacuum",
        rejected=("vacuum=1.0", "interlayer gap"),
        valid=("> 2.265", "typically >= 5.0"),
    ),
    Refusal(
        id="not-slab-like",
        error=NotASlabError,
        call=lambda bench: find_adsorption_sites(bench.bulk),
        parameter="structure",
        rejected=("not slab-like", "Vacuum 1.962 A < 5.0 A"),
        valid=(">= 5.0 A Vacuum along c", ">= 2 atomic z-levels"),
    ),
    Refusal(
        id="fraction",
        error=InvalidParameterError,
        call=lambda bench: fix_bottom_atoms(bench.slab, -0.1),
        parameter="fraction",
        rejected=("fraction=-0.1",),
        valid=("[0.0, 1.0]", "floor(n_atoms * fraction)"),
    ),
    Refusal(
        id="unknown-molecule",
        error=InvalidParameterError,
        call=lambda bench: place_adsorbate(
            bench.slab, bench.sites[0], molecule="XYZ", anchor="C"
        ),
        parameter="molecule",
        rejected=("molecule='XYZ'",),
        valid=("`g2` name", "'CO'", "'OH'", "'H2O'"),
    ),
    Refusal(
        id="anchor-absent",
        error=InvalidParameterError,
        call=lambda bench: place_adsorbate(
            bench.slab, bench.sites[0], molecule="CO", anchor="N"
        ),
        parameter="anchor",
        rejected=("anchor='N'", "elements: ['C', 'O']"),
        valid=("one of ['C', 'O']",),
    ),
    Refusal(
        id="binding-height",
        error=InvalidParameterError,
        call=lambda bench: place_adsorbate(
            bench.slab, bench.sites[0], molecule="CO", anchor="C", height=-1.0
        ),
        parameter="height",
        rejected=("height=-1.0",),
        valid=("finite float > 0", "1.85"),
    ),
    Refusal(
        id="anchor-index-required",
        error=InvalidParameterError,
        call=lambda bench: check_structure(bench.placed.structure),
        parameter="anchor_index",
        rejected=("anchor_index=None", "Adsorbate-tagged"),
        valid=("an int in [0, 6]",),
    ),
    Refusal(
        id="anchor-index-out-of-range",
        error=InvalidParameterError,
        call=lambda bench: check_adsorbate_anchor_above_surface(
            bench.placed.structure, 99
        ),
        parameter="anchor_index",
        rejected=("anchor_index=99", "7 sites"),
        valid=("0 to 6",),
    ),
    Refusal(
        id="distance-threshold",
        error=InvalidParameterError,
        call=lambda bench: check_min_distance(bench.placed.structure, threshold=0.0),
        parameter="threshold",
        rejected=("threshold=0.0",),
        valid=("finite number > 0", "0.5"),
    ),
)


def assert_refusal_contract(
    message: str,
    *,
    parameter: str,
    rejected: tuple[str, ...],
    valid: tuple[str, ...],
) -> None:
    """Assert every fact §H requires of one refusal message.

    Shared by the table-driven cases and the ones that need a temporary file, so
    the contract is stated once. ``parameter`` must lead, because a message that
    opens with the parameter is the one a reader can scan for.
    """
    assert message.startswith(parameter), (
        f"message does not open with the parameter name {parameter!r}: {message!r}"
    )

    for fragment in rejected:
        assert fragment in message, (
            f"message does not echo the rejected value ({fragment!r}): {message!r}"
        )

    assert "Valid:" in message, f"message names no valid range or set: {message!r}"
    for fragment in valid:
        assert fragment in message, (
            f"message does not state the concrete valid range ({fragment!r}): "
            f"{message!r}"
        )
    assert message.count("Valid:") == 1, (
        f"message states more than one valid range: {message!r}"
    )

    assert "Example: " in message, f"message carries no pasteable call: {message!r}"
    assert message.count("Example:") == 1, (
        f"message carries more than one example: {message!r}"
    )

    assert "\n" not in message and "\r" not in message, (
        f"message is not one line: {message!r}"
    )
    assert message == message.strip(), f"message has surrounding whitespace: {message!r}"
    assert not message.endswith("."), (
        f"message ends with a period, which §H.1 rule 7 forbids: {message!r}"
    )
    assert len(message) <= MAX_MESSAGE_LENGTH, (
        f"message is {len(message)} characters, over the "
        f"{MAX_MESSAGE_LENGTH}-character bound: {message!r}"
    )


@pytest.mark.parametrize("refusal", REFUSALS, ids=[row.id for row in REFUSALS])
def test_each_failure_mode_names_parameter_value_and_valid_range(
    refusal: Refusal, bench: Bench
) -> None:
    """One row per failure mode: the right exception, and a message that can be acted on."""
    with pytest.raises(refusal.error) as excinfo:
        refusal.call(bench)

    assert_refusal_contract(
        str(excinfo.value),
        parameter=refusal.parameter,
        rejected=refusal.rejected,
        valid=refusal.valid,
    )


def test_a_missing_structure_reference_stays_the_builtin_file_not_found_error(
    bench: Bench,
) -> None:
    """Callers already write ``except FileNotFoundError``; the library must not rename it.

    ``CONVENTIONS.md`` §H.1 rule 6 makes this the one failure mode that is *not* a
    ``Dsh4sciError``, because the builtin's meaning is already exact.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        load_structure(MISSING)

    assert not isinstance(excinfo.value, Dsh4sciError)


def test_unparseable_content_names_the_path_and_the_format_and_chains_the_cause() -> None:
    """A parser failure is wrapped, never leaked: the agent needs the path and the reason.

    ``pymatgen`` answers with a bare ``IndexError`` (measured) or a bare
    ``ValueError``; neither names the parameter nor offers a next call.

    The file lives under a deliberately short temporary directory: §H's
    210-character bound is a bound on the *message*, and a 100-character macOS
    ``tmp_path`` would measure the filesystem's path length rather than the
    wording. (The rejected value is echoed verbatim, whatever its length.)
    """
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="dsh4sci-") as directory:
        broken = Path(directory) / "broken.poscar"
        broken.write_text("this file is not a POSCAR\n", encoding="utf-8")

        with pytest.raises(StructureLoadError) as excinfo:
            load_structure(broken)

        message = str(excinfo.value)
        assert_refusal_contract(
            message,
            parameter="structure_path",
            rejected=(str(broken), "could not be parsed as POSCAR"),
            valid=("a POSCAR/CIF file readable by pymatgen",),
        )
        assert excinfo.value.__cause__ is not None, (
            "the parser's own exception must be chained, not discarded"
        )


def test_an_unsupported_suffix_is_refused_before_the_file_is_looked_for(
    tmp_path: Path,
) -> None:
    """The suffix is the caller's first and clearest mistake, so it wins the report.

    A path that is both unsupported and missing must report the suffix: that is the
    fixable fact. (Ordered deliberately in :func:`dsh4sci.loading.load_structure`.)
    """
    with pytest.raises(UnsupportedFormatError) as excinfo:
        load_structure(tmp_path / "missing.xyz")

    assert "unsupported suffix" in str(excinfo.value)


@pytest.mark.parametrize("refusal", REFUSALS, ids=[row.id for row in REFUSALS])
def test_every_refusal_is_catchable_as_the_library_root_and_as_value_error(
    refusal: Refusal, bench: Bench
) -> None:
    """One root for "this library refused", and ``ValueError`` for pymatgen-shaped code.

    ``FileNotFoundError`` is the documented exception (it is a builtin whose meaning
    is already exact), so it is the only row that is not a ``Dsh4sciError``.
    """
    error_type = refusal.error
    if error_type is FileNotFoundError:
        assert not issubclass(error_type, Dsh4sciError)
        return

    assert issubclass(error_type, Dsh4sciError)
    assert issubclass(error_type, ValueError)
