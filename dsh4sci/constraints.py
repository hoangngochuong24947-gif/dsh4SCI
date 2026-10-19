"""Selective dynamics: marking the bottom of a Slab as fixed for relaxation.

``pymatgen`` never populates ``selective_dynamics`` -- a ``Slab`` has no such
attribute (``hasattr(...)`` is ``False``), and downstream relaxation tools read the
flags from a POSCAR block that only exists when a site property of that name is
present (CONVENTIONS.md §D.1). This module is the one place that writes it.

The rule is a *fraction of the Slab's atoms, taken from the bottom*, because that
is how relaxation is specified in practice: hold the deep layers at their bulk
positions and let the surface reconstruct. ``floor`` rather than ``round`` keeps
the count monotone in ``fraction`` and guarantees ``fraction=0`` fixes nothing.
"""

from __future__ import annotations

import math
from typing import Final, TypeVar

import numpy as np
from pymatgen.core import Structure

from dsh4sci.errors import InvalidParameterError

__all__ = ["fix_bottom_atoms"]

_SlabT = TypeVar("_SlabT", bound=Structure)

# One valid call, used by every error message in this module so the snippet a
# caller is told to paste is the documented invocation (§H.1 rule 5).
_FIX_EXAMPLE = "fix_bottom_atoms(slab, fraction=0.5)"

# Decimal places the Cartesian coordinates are rounded to before they are ordered.
# Pymatgen does not contract the ordering of the sites it returns, and layers that
# are equal in exact arithmetic are not necessarily equal in the last float bit, so
# the comparison is quantised rather than exact. Six decimals is one part in 1e6 of
# an angstrom: far coarser than float noise, far finer than any real displacement.
_COMPARE_NDIGITS: Final[int] = 6

#: The three flags mean "free to move along x, y, z" and "fixed".
_FREE: Final[tuple[bool, bool, bool]] = (True, True, True)
_FIXED: Final[tuple[bool, bool, bool]] = (False, False, False)

# The site property Poscar reads to emit the three-column flag block (§D.1). Private
# because callers never write it by hand -- this module is the one writer.
_SELECTIVE_DYNAMICS: Final[str] = "selective_dynamics"


def _validate_fraction(fraction: float) -> float:
    """Validate ``fraction`` into ``[0.0, 1.0]`` before ``floor`` can be applied.

    ``floor(n_atoms * fraction)`` is only a meaningful count for a fraction in the
    unit interval: ``floor(4 * -0.1) == -1`` would fix nothing while looking like a
    request, and ``floor(4 * 1.5) == 6`` would index past the end of the site list.
    The check is therefore ordered before the floor, not after it (§D.3).

    Raises:
        InvalidParameterError: ``fraction`` is not a real number in ``[0.0, 1.0]``
            -- including ``nan``, which passes neither ``>=`` nor ``<=``.
    """
    if isinstance(fraction, bool) or not isinstance(
        fraction, (int, float, np.floating)
    ):
        raise InvalidParameterError(
            f"fraction={fraction!r} is invalid: expected a number in [0.0, 1.0]. "
            f"Valid: a float in [0.0, 1.0]; floor(n_atoms * fraction) atoms are "
            f"fixed. Example: {_FIX_EXAMPLE}"
        )
    if not 0.0 <= fraction <= 1.0:
        raise InvalidParameterError(
            f"fraction={fraction!r} is invalid: must be in [0.0, 1.0]. "
            f"Valid: a float in [0.0, 1.0]; floor(n_atoms * fraction) atoms are "
            f"fixed. Example: {_FIX_EXAMPLE}"
        )
    return float(fraction)


def fix_bottom_atoms(slab: _SlabT, fraction: float) -> _SlabT:
    """Return a copy of ``slab`` with the bottom ``floor(n_atoms * fraction)`` atoms fixed.

    "Fixed" is expressed the only way pymatgen understands: a
    ``selective_dynamics`` site property holding ``[False, False, False]`` for a
    site held in place and ``[True, True, True]`` for a free one. ``Poscar(slab)``
    then emits the ``Selective dynamics`` block with no extra keyword, and
    ``Structure.from_file`` reads the flags back (§D.1, §D.2).

    Args:
        slab: The Slab to mark up. Its Cartesian coordinates must be stacked along
            ``z`` -- true for every Slab from :func:`dsh4sci.slabs.generate_slabs`,
            whose c axis is orthogonal to the in-plane vectors.
        fraction: The fraction of atoms to fix, in ``[0.0, 1.0]``.

    Returns:
        A new structure of the same type as ``slab`` (a ``Slab`` in, a ``Slab``
        out). Coordinates, lattice and every other site property are carried over
        unchanged, so the Vacuum is bit-identical.

    Raises:
        InvalidParameterError: ``fraction`` is outside ``[0.0, 1.0]``. Validated
            before ``floor`` is applied, because ``floor(4 * -0.1) == -1`` would
            silently fix nothing and ``floor(4 * 1.5) == 6`` would index past the
            end of the site list.

    Example:
        ``fix_bottom_atoms(slab, fraction=0.5)`` fixes the two lowest layers of a
        4-layer slab.
    """
    n_atoms = len(slab)
    fraction = _validate_fraction(fraction)
    k = int(math.floor(n_atoms * fraction))

    fixed: set[int] = set()
    if k > 0:
        cart = np.asarray(slab.cart_coords, dtype=float)
        # A *total* order, independent of how the caller's site list happens to be
        # arranged: z ascending, then x, then y, then the site index. z alone is not
        # enough because every atom in one layer shares a z, and pymatgen does not
        # contract the order of the sites it returns -- so a z-only (stable) sort
        # would resolve equal-z atoms by input order and fix different atoms for a
        # reordered but physically identical slab (§D.3).
        order = sorted(
            range(n_atoms),
            key=lambda i: (
                round(float(cart[i, 2]), _COMPARE_NDIGITS),
                round(float(cart[i, 0]), _COMPARE_NDIGITS),
                round(float(cart[i, 1]), _COMPARE_NDIGITS),
                i,
            ),
        )
        fixed = set(order[:k])

    out = slab.copy()
    if fixed:
        out.add_site_property(
            _SELECTIVE_DYNAMICS,
            [_FIXED if i in fixed else _FREE for i in range(n_atoms)],
        )
    return out
