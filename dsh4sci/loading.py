"""Turn a Structure reference on disk into a ``Structure`` object.

This module is the library's **entire** file-I/O surface. A Structure reference
(CONTEXT.md) is a path used only to load a structure; every other module takes a
``pymatgen.core.Structure`` and never sees a filename. Keeping the path-accepting
entry point to exactly one function is what lets the rest of the library stay
pure functions over objects.

The supported set is deliberately small and explicit — POSCAR (``.poscar``,
``.vasp``, and the bare ``POSCAR``/``CONTCAR`` names VASP writes) and CIF
(``.cif``). Guessing at further formats would be a promise the library cannot
keep, so an unknown suffix is a clean refusal rather than a silent attempt.

**Validation happens before pymatgen is called, in this order**, because upstream
checks the extension *before* the file exists and reports neither clearly
(CONVENTIONS.md §B.2): resolve the suffix, reject an unsupported one, reject a
missing file, and only then delegate. Each failure names the path, the format
expected and the supported set, in one line, so an agent that reads the error can
correct its next call without asking a human (§H).

Nothing here touches the network: the only I/O is reading the local file, and a
URL is just a string that will fail the existence check.
"""

from __future__ import annotations

import os
from pathlib import Path

from pymatgen.core import Structure

from dsh4sci.errors import StructureLoadError, UnsupportedFormatError

__all__ = ["SUPPORTED_SUFFIXES", "load_structure"]

# Suffixes this library reads, in the order the error messages list them. `.vasp`
# is the same POSCAR writer under another name; both map to the same reader.
_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".poscar": "POSCAR",
    ".vasp": "POSCAR",
    ".cif": "CIF",
}

# VASP writes its structure files with no suffix at all. Accepting them costs one
# set membership test, and it is the name a caller will most often be holding.
_BARE_POSCAR_NAMES: frozenset[str] = frozenset({"POSCAR", "CONTCAR"})

#: The suffixes accepted by :func:`load_structure`, as reported in error messages.
SUPPORTED_SUFFIXES: tuple[str, ...] = tuple(_FORMAT_BY_SUFFIX)

_SUPPORTED_SET = repr(list(SUPPORTED_SUFFIXES))
_SUPPORTED_TEXT = f"one of {_SUPPORTED_SET}"
_EXAMPLE = "load_structure('/data/Pt.poscar')"


def load_structure(structure_path: str | os.PathLike[str]) -> Structure:
    """Load a Structure reference into a ``Structure`` object.

    Args:
        structure_path: Path to a POSCAR (``.poscar``, ``.vasp``, ``POSCAR``,
            ``CONTCAR``) or CIF (``.cif``) file. Relative paths are resolved
            against the current working directory.

    Returns:
        The structure stored at ``structure_path``.

    Raises:
        UnsupportedFormatError: The suffix is not one the library reads. Raised
            before the existence check, so a bad suffix is reported as a bad
            suffix even when the file is also missing.
        FileNotFoundError: The path is not an existing regular file. A relative
            path is reported as the absolute path it resolved to.
        StructureLoadError: The file exists but pymatgen could not parse it; the
            message carries the underlying parser reason and the original
            exception is chained.

    Example:
        ``load_structure("fixtures/Pt.poscar")``
    """
    try:
        path = os.path.abspath(os.fsdecode(structure_path))
    except TypeError:
        # §H.1 rule 6: never leak a bare TypeError. os.fsdecode's message names
        # neither the parameter nor a valid call.
        raise TypeError(
            f"structure_path={structure_path!r} is invalid: expected a path, "
            f"got {type(structure_path).__name__}. Valid: str or os.PathLike. "
            f"Example: {_EXAMPLE}"
        ) from None
    name = Path(path).name
    suffix = Path(path).suffix.lower()

    format_name = _FORMAT_BY_SUFFIX.get(suffix)
    is_bare_poscar = suffix == "" and name.upper() in _BARE_POSCAR_NAMES
    if format_name is None and not is_bare_poscar:
        raise UnsupportedFormatError(
            f"structure_path={path!r} has unsupported suffix {suffix!r}. "
            f"Valid: {_SUPPORTED_TEXT}. Example: {_EXAMPLE}"
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"structure_path={path!r} does not exist. "
            f"Valid: a readable file with suffix {_SUPPORTED_TEXT}. Example: {_EXAMPLE}"
        )

    if is_bare_poscar:
        format_name = "POSCAR"

    try:
        return Structure.from_file(path)
    except Exception as exc:  # noqa: BLE001 - any parser failure becomes one library error
        raise StructureLoadError(
            f"structure_path={path!r} could not be parsed as {format_name}: {exc}. "
            f"Valid: a POSCAR/CIF file readable by pymatgen. Example: {_EXAMPLE}"
        ) from exc
