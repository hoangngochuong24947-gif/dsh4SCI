"""Acceptance tests for the installable package skeleton (ticket #1).

These tests encode the ticket's acceptance criteria as executable spec: the
package imports, carries a version, and declares the exact dependency pins that
every measured number in ``CONVENTIONS.md`` was taken against. Expected values
come from the ticket and ``CONVENTIONS.md`` §A.3, never from the implementation.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import dsh4sci.errors as errors

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

_REQUIREMENT = re.compile(r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?P<specifier>.*)")


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _requirements() -> dict[str, str]:
    """Map distribution name -> version specifier for the runtime dependencies."""
    pinned: dict[str, str] = {}
    for requirement in _pyproject()["project"]["dependencies"]:
        match = _REQUIREMENT.fullmatch(requirement.strip())
        if match is None:
            pytest.fail(f"unparseable requirement: {requirement!r}")
        pinned[match.group("name").lower()] = match.group("specifier")
    return pinned


def test_package_imports_and_exposes_version() -> None:
    import dsh4sci

    assert dsh4sci.__version__ == "0.1.0"


def test_project_metadata_is_set_for_an_importable_library() -> None:
    project = _pyproject()["project"]

    assert project["name"] == "dsh4sci"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.12"


def test_declared_dependencies_are_exact_pins() -> None:
    deps = _requirements()

    assert deps["pymatgen"] == "==2026.9.24"
    # pymatgen is a shim declaring only pymatgen-core>=2026.9.23, so the real code
    # package must carry its own exact pin or a clean resolve drifts to 2026.10.16.
    assert deps["pymatgen-core"] == "==2026.10.2"
    assert deps["ase"] == "==3.29.0"
    assert deps["spglib"] == "==2.7.0"
    # numpy stays a range on purpose: pymatgen-core already constrains it.
    assert deps["numpy"] == ">=2.1,<3"


def test_dev_extra_pins_pytest() -> None:
    dev = _pyproject()["project"]["optional-dependencies"]["dev"]

    assert dev == ["pytest==8.4.2"]


def test_setuptools_packages_only_the_library_namespace() -> None:
    assert _pyproject()["tool"]["setuptools"]["packages"] == ["dsh4sci"]


def test_pytest_collects_from_the_tests_directory() -> None:
    ini_options = _pyproject()["tool"]["pytest"]["ini_options"]

    assert ini_options["testpaths"] == ["tests"]


# --- exception types -------------------------------------------------------
# Ticket #1 owns the exception *types*; ticket #8 owns their messages and its
# own test_errors.py. The contract asserted here is the hierarchy itself.


LIBRARY_ERRORS = (
    errors.StructureLoadError,
    errors.UnsupportedFormatError,
    errors.InvalidMillerIndexError,
    errors.InvalidParameterError,
    errors.NotASlabError,
)


def test_library_has_a_single_root_exception() -> None:
    assert issubclass(errors.Dsh4sciError, Exception)

    for error_type in LIBRARY_ERRORS:
        assert issubclass(error_type, errors.Dsh4sciError), f"{error_type.__name__} escapes the root"


def test_input_errors_remain_catchable_as_value_error() -> None:
    # Deliberate multiple inheritance: callers can keep the `except ValueError`
    # they would have written against pymatgen while the library still has one root.
    for error_type in LIBRARY_ERRORS:
        assert issubclass(error_type, ValueError), f"{error_type.__name__} is not a ValueError"


def test_unsupported_format_is_a_structure_load_error() -> None:
    assert issubclass(errors.UnsupportedFormatError, errors.StructureLoadError)


def test_each_error_type_is_distinct_so_failure_modes_stay_separable() -> None:
    # One exception type per failure mode (CONVENTIONS.md §H.1 rule 8).
    assert len({*LIBRARY_ERRORS, errors.Dsh4sciError}) == len(LIBRARY_ERRORS) + 1


@pytest.mark.parametrize("error_type", [errors.Dsh4sciError, *LIBRARY_ERRORS])
def test_errors_carry_a_message(error_type: type[Exception]) -> None:
    error = error_type("structure_path='/data/Pt.poscar' is invalid")

    assert str(error).startswith("structure_path=")


def test_missing_files_are_not_absorbed_into_the_library_root() -> None:
    # `FileNotFoundError` stays the builtin: callers expect to catch it unchanged.
    assert not issubclass(errors.Dsh4sciError, FileNotFoundError)

    with pytest.raises(FileNotFoundError):
        raise FileNotFoundError("missing")
