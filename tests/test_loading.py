"""Acceptance tests for ``load_structure`` (ticket #2).

The ticket's acceptance criteria are the test list. The expected loads come from
CONVENTIONS.md §B.1 (measured on the pinned stack), never from the
implementation; every error message is asserted on its *stem*, because
CONVENTIONS.md §H makes the stem the contract an agent reads.

``load_structure`` is the only place in this library that accepts a path. Once a
Structure is loaded it travels as an object, so the behaviour pinned here is the
library's whole file-I/O surface.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from pathlib import Path

import pytest
from pymatgen.core import Structure

from dsh4sci.errors import StructureLoadError, UnsupportedFormatError
from dsh4sci.loading import load_structure

# CONVENTIONS.md §B.1: file -> (element amounts, n_sites, a = b = c in Angstrom).
# All five are conventional cubic cells with 90/90/90 angles.
FIXTURE_EXPECTATIONS: dict[str, tuple[dict[str, float], int, float]] = {
    "Pt.poscar": ({"Pt": 4.0}, 4, 3.9231),
    "Cu.poscar": ({"Cu": 4.0}, 4, 3.615),
    "Si.poscar": ({"Si": 8.0}, 8, 5.431),
    "SiC.poscar": ({"Si": 4.0, "C": 4.0}, 8, 4.36),
    "Cu.cif": ({"Cu": 4.0}, 4, 3.615),
}


@pytest.mark.parametrize("name", sorted(FIXTURE_EXPECTATIONS))
def test_loads_the_checked_in_fixtures(
    name: str, fixture_path: Callable[[str], Path]
) -> None:
    expected_species, expected_n_sites, expected_a = FIXTURE_EXPECTATIONS[name]

    structure = load_structure(fixture_path(name))

    assert isinstance(structure, Structure)
    assert structure.composition.get_el_amt_dict() == expected_species
    assert len(structure) == expected_n_sites
    assert structure.lattice.abc == pytest.approx((expected_a,) * 3)
    assert structure.lattice.angles == pytest.approx((90.0, 90.0, 90.0))


def test_missing_file_raises_file_not_found_naming_the_full_path(
    tmp_path: Path,
) -> None:
    # The caller handed over a path that is not there; the error has to name the
    # path it tried (not a bare suffix) so the next call can be corrected.
    missing = tmp_path / "Nope.poscar"

    with pytest.raises(FileNotFoundError) as excinfo:
        load_structure(missing)

    message = str(excinfo.value)
    assert str(missing) in message
    assert missing.name in message
    assert message.startswith(f"structure_path={str(missing)!r} does not exist")


def test_unsupported_extension_names_the_extension_and_the_supported_set(
    tmp_path: Path,
) -> None:
    # An existing file whose suffix we do not read: say which suffix was refused
    # *and* which ones are valid, so the agent does not have to guess.
    unsupported = tmp_path / "molecule.xyz"
    unsupported.write_text("1\ncomment\nH 0.0 0.0 0.0\n")

    with pytest.raises(UnsupportedFormatError) as excinfo:
        load_structure(unsupported)

    message = str(excinfo.value)
    quoted_path = repr(str(unsupported))
    expected_prefix = f"structure_path={quoted_path} has unsupported suffix '.xyz'"
    assert message.startswith(expected_prefix)
    for suffix in (".poscar", ".vasp", ".cif"):
        assert suffix in message


def test_unsupported_extension_is_reported_even_when_the_file_is_missing(
    tmp_path: Path,
) -> None:
    # pymatgen checks the extension before existence (CONVENTIONS.md §B.2), and so
    # do we: the caller asked for a format we do not read, so say that instead of
    # sending them looking for a file that would not help.
    with pytest.raises(UnsupportedFormatError) as excinfo:
        load_structure(tmp_path / "nothing.foo")

    message = str(excinfo.value)
    assert "unsupported suffix '.foo'" in message
    assert ".poscar" in message and ".cif" in message


def test_loads_every_poscar_spelling_the_readers_support(
    fixture_path: Callable[[str], Path], tmp_path: Path
) -> None:
    # `.poscar`, `.vasp` and the bare VASP names are the same format; a caller
    # holding a CONTCAR should not have to rename it first.
    source = fixture_path("Pt.poscar")
    spellings = ("Pt.vasp", "POSCAR", "CONTCAR", "Pt.POSCAR", "Pt.POSCAR".lower())
    for spelling in spellings:
        target = tmp_path / spelling
        target.write_text(source.read_text())

        structure = load_structure(target)

        assert structure.composition.get_el_amt_dict() == {"Pt": 4.0}, spelling
        assert len(structure) == 4


def test_cif_suffix_is_case_insensitive(
    fixture_path: Callable[[str], Path], tmp_path: Path
) -> None:
    upper = tmp_path / "Cu.CIF"
    upper.write_text(fixture_path("Cu.cif").read_text())

    assert load_structure(upper).composition.get_el_amt_dict() == {"Cu": 4.0}


def test_corrupt_poscar_raises_structure_load_error_naming_path_and_format(
    tmp_path: Path,
) -> None:
    # The parser's own error is a bare ValueError/IndexError with no path in it.
    # The library must say which file, which format, and why -- and keep the
    # original exception chained for a human debugging session.
    corrupt = tmp_path / "broken.poscar"
    corrupt.write_text("Pt fcc\n1.0\nnot a structure at all\n")

    with pytest.raises(StructureLoadError) as excinfo:
        load_structure(corrupt)

    message = str(excinfo.value)
    expected_prefix = f"structure_path={str(corrupt)!r} could not be parsed as POSCAR: "
    assert message.startswith(expected_prefix)
    assert "could not convert string to float" in message
    assert isinstance(excinfo.value.__cause__, ValueError)


@pytest.mark.filterwarnings("ignore::UserWarning")  # pymatgen logs the CIF reason too
def test_corrupt_cif_names_cif_as_the_expected_format(tmp_path: Path) -> None:
    corrupt = tmp_path / "broken.cif"
    corrupt.write_text("data_broken\n_cell_length_a  not-a-number\n")

    with pytest.raises(StructureLoadError) as excinfo:
        load_structure(corrupt)

    assert "could not be parsed as CIF" in str(excinfo.value)


def test_a_directory_with_a_supported_suffix_is_a_missing_file(tmp_path: Path) -> None:
    # `x.poscar/` is not a Structure reference; report it the way a missing file
    # is reported rather than leaking a bare IsADirectoryError.
    directory = tmp_path / "folder.poscar"
    directory.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        load_structure(directory)

    assert str(directory) in str(excinfo.value)


def test_relative_path_is_reported_as_the_absolute_path_it_resolved_to(
    fixture_path: Callable[[str], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A relative path in an error message is only actionable if the caller knows
    # the working directory it was relative to; report the resolved path.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Pt.poscar").write_text(fixture_path("Pt.poscar").read_text())

    assert len(load_structure("Pt.poscar")) == 4

    with pytest.raises(FileNotFoundError) as excinfo:
        load_structure("./Missing.poscar")

    assert str(tmp_path / "Missing.poscar") in str(excinfo.value)


def test_loading_never_touches_the_network(
    fixture_path: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The load path is local file I/O only. Any socket attempt during a load is a
    # defect, so make one impossible and prove the fixtures still load.
    def forbid(*args: object, **kwargs: object) -> None:
        raise AssertionError("load_structure must not open a network connection")

    monkeypatch.setattr(socket, "socket", forbid)
    monkeypatch.setattr(socket, "create_connection", forbid)

    for name in FIXTURE_EXPECTATIONS:
        assert len(load_structure(fixture_path(name))) > 0


def test_pt_bulk_fixture_is_the_conventional_cubic_cell(pt_bulk: Structure) -> None:
    # The shared fixture later tickets cut Slabs from; pin what it is.
    assert isinstance(pt_bulk, Structure)
    assert pt_bulk.composition.get_el_amt_dict() == {"Pt": 4.0}
    assert pt_bulk.lattice.abc == pytest.approx((3.9231,) * 3)


def test_a_parser_error_that_is_not_a_value_error_is_still_a_structure_load_error(
    tmp_path: Path,
) -> None:
    # pymatgen lets an IndexError escape for a POSCAR too short to read (measured:
    # `IndexError: list index out of range`). A caller must never have to know
    # which upstream exception a given malformation produces.
    too_short = tmp_path / "short.poscar"
    too_short.write_text("not a structure at all\n")

    with pytest.raises(StructureLoadError) as excinfo:
        load_structure(too_short)

    assert isinstance(excinfo.value.__cause__, IndexError)
    assert "list index out of range" in str(excinfo.value)


def test_missing_file_message_matches_the_convention_verbatim(tmp_path: Path) -> None:
    # CONVENTIONS.md §H.2 fixes this message word for word; the shape is a contract
    # that later tickets and the agent-facing SKILL.md read. Built from tmp_path so
    # the test never depends on what exists at /data on this machine.
    missing = tmp_path / "Pt.poscar"

    with pytest.raises(FileNotFoundError) as excinfo:
        load_structure(missing)

    message = str(excinfo.value)
    assert message == (
        f"structure_path={str(missing)!r} does not exist. "
        "Valid: a readable file with suffix one of ['.poscar', '.vasp', '.cif']. "
        "Example: load_structure('/data/Pt.poscar')"
    )
    assert "\n" not in message
    assert not message.endswith(".")


def test_unsupported_suffix_message_matches_the_convention_verbatim(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "Pt.xyz"
    unsupported.write_text("")

    with pytest.raises(UnsupportedFormatError) as excinfo:
        load_structure(unsupported)

    message = str(excinfo.value)
    assert message == (
        f"structure_path={str(unsupported)!r} has unsupported suffix '.xyz'. "
        "Valid: one of ['.poscar', '.vasp', '.cif']. "
        "Example: load_structure('/data/Pt.poscar')"
    )
    assert "\n" not in message
    assert not message.endswith(".")


# The §H.2 canonical examples, verbatim. §H.1 caps the shape at ~200 characters;
# these are the messages the cap is about, so pin their length as a literal rather
# than measuring a message built from whatever tmp_path this machine hands out.
CANONICAL_H2_MESSAGES = (
    "structure_path='/data/Pt.poscar' does not exist. "
    "Valid: a readable file with suffix one of ['.poscar', '.vasp', '.cif']. "
    "Example: load_structure('/data/Pt.poscar')",
    "structure_path='/data/Pt.xyz' has unsupported suffix '.xyz'. "
    "Valid: one of ['.poscar', '.vasp', '.cif']. "
    "Example: load_structure('/data/Pt.poscar')",
    "structure_path='/data/broken.poscar' could not be parsed as POSCAR: "
    "could not convert string to float: 'at all'. "
    "Valid: a POSCAR/CIF file readable by pymatgen. "
    "Example: load_structure('/data/Pt.poscar')",
)


#: §H.1 says "under ~200 characters". §H.2's own parse-error example is 202, so the
#: cap is approximate; hold the messages to just past the spec's longest example.
MAX_MESSAGE_LENGTH = 210


@pytest.mark.parametrize("message", CANONICAL_H2_MESSAGES)
def test_the_documented_example_messages_respect_the_length_cap(message: str) -> None:
    assert len(message) < MAX_MESSAGE_LENGTH
    assert "\n" not in message
    assert not message.endswith(".")


def test_load_structure_is_reachable_from_the_package_root() -> None:
    # One entry point, exported once. `import dsh4sci` is what SKILL.md shows an
    # agent, so the name must be there without knowing the module layout.
    import dsh4sci

    assert dsh4sci.load_structure is load_structure
    assert ".poscar" in dsh4sci.SUPPORTED_SUFFIXES


@pytest.mark.parametrize("not_a_path", [123, None, 3.5])
def test_a_non_path_argument_names_the_parameter_instead_of_leaking_type_error(
    not_a_path: object,
) -> None:
    # §H.1 rule 6 forbids leaking a bare TypeError; os.fsdecode's own message names
    # neither the parameter nor a valid call, so the agent cannot recover from it.
    with pytest.raises(TypeError) as excinfo:
        load_structure(not_a_path)

    message = str(excinfo.value)
    assert message.startswith(f"structure_path={not_a_path!r} is invalid")
    assert "Valid: str or os.PathLike" in message
