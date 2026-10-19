"""Exception types raised by this library.

The library has a single root, :class:`Dsh4sciError`, so a consumer can catch
everything this library raises with one ``except``. Every concrete type *also*
inherits from :class:`ValueError`, deliberately and not by accident:

upstream ``pymatgen`` signals bad input with ``ValueError`` (``Structure.from_file``
raises ``ValueError("Unrecognized extension ...")``, ``SlabGenerator`` raises
``ValueError`` on a malformed Miller index). A caller written against pymatgen
therefore already has ``except ValueError`` surrounding these calls, and keeping
that inheritance means those call sites keep working unchanged — while a caller
that wants precision can catch the specific subclass, and one that wants to catch
*only* this library's failures catches ``Dsh4sciError``.

``FileNotFoundError`` is intentionally **not** subclassed here. It is a builtin
whose meaning (a missing *Structure reference*) is already exact, and callers
expect to catch it as the builtin. The library raises it directly for missing
files rather than wrapping it.

Messages are written by the call sites, not here. The single-line shape —
``<parameter>=<value!r> is invalid: <why>. Valid: <range-or-set>. Example: <call>``
— is specified in ``CONVENTIONS.md`` §H.
"""

from __future__ import annotations

__all__ = [
    "Dsh4sciError",
    "InvalidMillerIndexError",
    "InvalidParameterError",
    "NotASlabError",
    "StructureLoadError",
    "UnsupportedFormatError",
]


class Dsh4sciError(Exception):
    """Base class for every exception raised by this library."""


class StructureLoadError(Dsh4sciError, ValueError):
    """A Structure reference could not be read into a ``Structure``.

    Covers an unparseable file, a corrupt payload, and any other failure to turn
    stored input into a structure. Missing files stay ``FileNotFoundError``, and a
    wrong suffix is the more specific :class:`UnsupportedFormatError`.
    """


class UnsupportedFormatError(StructureLoadError):
    """The Structure reference's suffix is not a format this library reads.

    A subtype of :class:`StructureLoadError` because the caller's recovery is the
    same — supply a different path — but named separately so the message can list
    the supported suffixes.
    """


class InvalidMillerIndexError(Dsh4sciError, ValueError):
    """A Miller index is not three integers naming a real crystal plane.

    Raised instead of letting upstream leak ``ZeroDivisionError`` for ``(0, 0, 0)``
    or ``TypeError`` for a non-integer index.
    """


class InvalidParameterError(Dsh4sciError, ValueError):
    """A numeric parameter is outside its valid range.

    The motivating case is Vacuum: upstream silently accepts a zero or negative
    ``min_vacuum_size``, which produces a slab that is not physically meaningful.
    """


class NotASlabError(Dsh4sciError, ValueError):
    """The input is not a Slab.

    Raised when an operation that only means something on a Slab — enumerating
    Adsorption sites, checking a surface — receives a bulk structure or a cell
    without a Vacuum.
    """
