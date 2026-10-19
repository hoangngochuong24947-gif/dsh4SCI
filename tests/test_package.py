"""Acceptance tests for the installable package skeleton (ticket #1).

These tests encode the ticket's acceptance criteria as executable spec: the
package imports, carries a version, and declares the exact dependency pins that
every measured number in ``CONVENTIONS.md`` was taken against. Expected values
come from the ticket and ``CONVENTIONS.md`` §A.3, never from the implementation.
"""

from __future__ import annotations

import re
import tomllib
import types
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


# --- the declared public surface -------------------------------------------
# `__all__` is what makes the re-exports in `dsh4sci/__init__.py` a contract
# instead of an accident of import order. It is written out here as a literal --
# not derived from the module -- so adding, removing or renaming a public symbol
# has to move this list.

PUBLIC_SURFACE = frozenset(
    {
        # ticket #2 -- structure loading
        "SUPPORTED_SUFFIXES",
        "load_structure",
        # ticket #3 -- the shared geometry primitives
        "ADSORBATE_ROLE",
        "ROLE_PROPERTY",
        "SLAB_ROLE",
        "count_z_levels",
        "group_z_levels",
        "set_vacuum",
        "split_roles",
        "vacuum_gap",
        # ticket #3 -- slab generation and the Termination fingerprint
        "TerminationFingerprint",
        "generate_slabs",
        "termination_fingerprint",
        # ticket #4 -- selective dynamics
        "SlabT",
        "fix_bottom_atoms",
        # ticket #5 -- adsorption site enumeration
        "MIN_VACUUM",
        "MIN_Z_LEVELS",
        "SITE_TYPES",
        "AdsorptionSite",
        "SiteType",
        "find_adsorption_sites",
        "is_slab_like",
        # ticket #6 -- adsorbate placement
        "DEFAULT_BINDING_HEIGHT",
        "PlacedAdsorbate",
        "place_adsorbate",
        # ticket #7 -- the structure self-check
        "ANGLE_MAX",
        "ANGLE_MIN",
        "MIN_CELL_VOLUME",
        "MIN_INTERATOMIC_DISTANCE",
        "CheckResult",
        "all_passed",
        "check_adsorbate_anchor_above_surface",
        "check_lattice_angles",
        "check_min_distance",
        "check_structure",
    }
)


def test_the_package_declares_its_public_surface() -> None:
    import dsh4sci

    assert isinstance(dsh4sci.__all__, list), (
        f"dsh4sci.__all__ is {type(dsh4sci.__all__).__name__}; it must be a list so "
        f"the surface is explicit"
    )
    assert all(isinstance(name, str) for name in dsh4sci.__all__)
    assert len(dsh4sci.__all__) == len(set(dsh4sci.__all__)), (
        "dsh4sci.__all__ declares a name twice"
    )
    assert set(dsh4sci.__all__) == PUBLIC_SURFACE, (
        f"missing: {sorted(PUBLIC_SURFACE - set(dsh4sci.__all__))}; "
        f"unexpected: {sorted(set(dsh4sci.__all__) - PUBLIC_SURFACE)}"
    )


def test_every_declared_name_exists_on_the_package() -> None:
    import dsh4sci

    missing = [name for name in dsh4sci.__all__ if not hasattr(dsh4sci, name)]

    assert missing == [], f"{missing} are declared in __all__ but not importable"


def test_star_import_yields_the_declared_surface_and_nothing_else() -> None:
    """`from __future__ import annotations` binds a module-level `annotations` name.

    Without `__all__` that name is part of the star-import surface, so a caller
    would get a `_Feature` object as if it were library API. The declared surface
    is the whole of what a star import may bring: nothing withheld, nothing leaked.
    """
    import dsh4sci

    namespace: dict[str, object] = {}
    exec("from dsh4sci import *", namespace)

    exported = {name for name in namespace if not name.startswith("__")}
    assert "annotations" not in namespace
    assert exported == set(dsh4sci.__all__), (
        f"star import leaked: {sorted(exported - set(dsh4sci.__all__))}"
    )


def test_the_declared_surface_holds_no_dunders_or_submodules() -> None:
    import dsh4sci

    for name in dsh4sci.__all__:
        value = getattr(dsh4sci, name)
        assert not name.startswith("_"), f"{name} is private"
        assert not isinstance(value, types.ModuleType), (
            f"{name} is a submodule, not part of the API"
        )


_DOCUMENTED_IMPORT = re.compile(
    r"^from dsh4sci import (?P<rest>\([^)]*\)|[^\n#]+)", re.MULTILINE
)


def _documented_package_imports() -> set[str]:
    """Every name ``SKILL.md`` imports from the package root.

    SKILL.md says only names exported from the package root may be documented, so
    its import lines are the documented promise this check holds `__all__` to.
    """
    text = (REPO_ROOT / "SKILL.md").read_text(encoding="utf-8")
    names: set[str] = set()
    for match in _DOCUMENTED_IMPORT.finditer(text):
        rest = match.group("rest").strip()
        if rest.startswith("("):
            rest = rest[1 : rest.index(")")]
        for chunk in rest.replace("\n", " ").split(","):
            name = chunk.split(" as ")[0].strip()
            if name:
                names.add(name)
    return names


def test_every_documented_package_import_is_in_the_public_surface() -> None:
    import dsh4sci

    documented = _documented_package_imports()

    assert documented, "SKILL.md imports nothing from the package root"
    assert documented <= set(dsh4sci.__all__), (
        f"documented but not exported: {sorted(documented - set(dsh4sci.__all__))}"
    )
