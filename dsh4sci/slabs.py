"""Slab generation and the Termination fingerprint.

Given a bulk Structure and a Miller index, :func:`generate_slabs` cuts the Slabs
that expose that surface; :func:`termination_fingerprint` gives each one a
reproducible, content-based identity.

**Why the fingerprint is ours to compute.** ``pymatgen.core.surface.Slab`` has no
``termination`` attribute -- only ``shift`` and ``miller_index`` -- and the order in
which ``get_slabs()`` enumerates Terminations is not a contract. A bare integer
would silently repoint when the dependency upgrades, so CONTEXT.md's Termination
fingerprint ("the composition of the terminating layer together with its in-plane
offset") must be derived from content. :class:`TerminationFingerprint` is that
derivation: layer composition at both faces plus the quantised in-plane ``shift``.

**The canonical pipeline** (CONVENTIONS.md §C.1), all four steps mandatory:

1. ``SlabGenerator`` with the **conventional** input cell, ``primitive=True``,
   ``reorient_lattice=True``, ``center_slab=True``, ``lll_reduce=False`` and
   ``max_normal_search=None``. The conventional cell is what makes the interlayer
   spacing come out as ``d(111) = a/sqrt(3)`` rather than the in-plane rhombus
   edge; ``lll_reduce`` and ``max_normal_search`` both change the atom count and
   the Termination set.
2. ``get_orthogonal_c_slab()`` on every slab, so the c axis is perpendicular to
   the in-plane vectors. Without it the in-plane vectors carry a z component and
   later supercell work smears the stack.
3. :func:`dsh4sci.geometry.set_vacuum` to make the Vacuum exact. ``min_vacuum_size``
   is only a lower bound; the generator pads to whole atomic layers.
4. ``center_slab=True`` keeps that Vacuum well defined by stopping the stack from
   straddling the periodic boundary.

**Upstream mutation.** ``SlabGenerator.__init__`` adds ``bulk_wyckoff`` and
``bulk_equivalent`` site properties to the Structure it is handed, so the caller's
bulk is copied first.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.surface import Slab, SlabGenerator

from dsh4sci.errors import InvalidMillerIndexError, InvalidParameterError
from dsh4sci.geometry import group_z_levels, set_vacuum

__all__ = [
    "LAYER_TOL",
    "SHIFT_NDIGITS",
    "TerminationFingerprint",
    "generate_slabs",
    "termination_fingerprint",
]

#: Number of decimal places the in-plane ``shift`` is quantised to (§C.5).
SHIFT_NDIGITS: Final[int] = 6

#: The z-gap in angstrom that separates two atomic layers (§C.5).
LAYER_TOL: Final[float] = 0.1

# One valid call, reused by every error message in this module so the snippet a
# caller is told to paste is always the documented invocation (§H.1 rule 5).
_GENERATE_EXAMPLE = (
    "generate_slabs(bulk, miller_index=(1, 1, 1), min_slab_size=10.0, vacuum=15.0)"
)


@dataclass(frozen=True, order=True)
class TerminationFingerprint:
    """Content-based identity of one Termination.

    A Termination is "a specific cut within one Miller-index family, distinguished
    by which atomic layer the exposed face ends on" (CONTEXT.md). The fingerprint
    records exactly that, plus the in-plane offset that distinguishes cuts which
    end on the same layer composition:

    Attributes:
        miller_index: The ``(hkl)`` family the cut belongs to.
        top_layer: The terminating layer at the upper face, as
            ``tuple(sorted((element, count), ...))``. Sorted so the fingerprint
            does not depend on site ordering.
        bottom_layer: The terminating layer at the lower face, same form.
        shift_ndigits: ``round(shift * 10**SHIFT_NDIGITS)`` -- the in-plane offset
            as a scaled integer.

    ``shift_ndigits`` is an integer rather than the raw float because pymatgen
    returns ``shift`` from matrix arithmetic: ``0.4999999999999998``,
    ``0.37500000000000006`` and ``0.875`` were all observed. Comparing those
    exactly is brittle and any internal reordering changes the last bits. 6
    decimal places is four orders of magnitude finer than the difference between
    two distinct Terminations here (the SiC pair differs by 0.5) and four orders
    coarser than float noise at this magnitude.

    The dataclass is frozen, hashable and order-comparable, so fingerprints work
    in sets and sort deterministically.

    Example:
        ``str(termination_fingerprint(slab))`` is
        ``"(1, 1, 1):Pt1/Pt1:shift=500000e-6"``.
    """

    miller_index: tuple[int, ...]
    top_layer: tuple[tuple[str, int], ...]
    bottom_layer: tuple[tuple[str, int], ...]
    shift_ndigits: int

    @property
    def shift(self) -> float:
        """The in-plane offset in fractional units, de-quantised."""
        return self.shift_ndigits / 10**SHIFT_NDIGITS

    def __str__(self) -> str:
        top = "".join(f"{element}{count}" for element, count in self.top_layer)
        bottom = "".join(
            f"{element}{count}" for element, count in self.bottom_layer
        )
        return (
            f"{self.miller_index}:{top}/{bottom}"
            f":shift={self.shift_ndigits}e-{SHIFT_NDIGITS}"
        )

    def sha(self) -> str:
        """A short, stable digest of this fingerprint: the first 16 hex of SHA-256.

        The digest is taken over the **built string**, never over a ``set`` of
        element names: set iteration order depends on ``PYTHONHASHSEED``, so
        hashing one would make the identity differ between processes even when the
        physics is identical. The string built by :meth:`__str__` is fully
        ordered, which is what makes this digest reproducible run to run.
        """
        return hashlib.sha256(str(self).encode()).hexdigest()[:16]


def _layer_composition(
    slab: Slab, layer_indices: Sequence[int]
) -> tuple[tuple[str, int], ...]:
    """Reduce one atomic layer to ``tuple(sorted((element, count), ...))``.

    Sorting is what makes the fingerprint independent of site ordering, and a
    tuple-of-tuples is what makes it hashable and JSON-visible. Note this is a
    *sorted tuple*, not a set: a set's iteration order depends on the per-process
    hash seed and would break ``TerminationFingerprint.sha`` across processes.
    """
    counts = Counter(str(slab[index].specie) for index in layer_indices)
    return tuple(sorted(counts.items()))


def _validate_miller_index(miller_index: Sequence[int]) -> tuple[int, int, int]:
    """Validate a Miller index before pymatgen can leak its own error (§C.6)."""
    if isinstance(miller_index, str) or not isinstance(miller_index, Sequence):
        raise InvalidMillerIndexError(
            f"miller_index={miller_index!r} is invalid: expected a sequence of 3 "
            f"integers. Valid: three integers not all zero. "
            f"Example: {_GENERATE_EXAMPLE}"
        )
    if len(miller_index) != 3:
        raise InvalidMillerIndexError(
            f"miller_index={tuple(miller_index)!r} is invalid: expected exactly 3 "
            f"indices, got {len(miller_index)}. Valid: three integers not all "
            f"zero. Example: {_GENERATE_EXAMPLE}"
        )
    indices: list[int] = []
    for index in miller_index:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise InvalidMillerIndexError(
                f"miller_index={tuple(miller_index)!r} is invalid: indices must be "
                f"integers, got {type(index).__name__} {index!r}. Valid: three "
                f"integers not all zero. Example: {_GENERATE_EXAMPLE}"
            )
        indices.append(int(index))
    if all(index == 0 for index in indices):
        raise InvalidMillerIndexError(
            f"miller_index={tuple(indices)!r} is invalid: at least one index must "
            f"be non-zero. Valid: three integers not all zero, e.g. (1, 1, 1). "
            f"Example: {_GENERATE_EXAMPLE}"
        )
    return (indices[0], indices[1], indices[2])


def _validate_positive(value: float, name: str, why: str) -> float:
    """Validate one strictly-positive numeric parameter with a §H message."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise InvalidParameterError(
            f"{name}={value!r} is invalid: expected a number in Angstrom. "
            f"Valid: a positive float. Example: {_GENERATE_EXAMPLE}"
        )
    if not value > 0:
        raise InvalidParameterError(
            f"{name}={value!r} is invalid: must be > 0. Valid: {why}. "
            f"Example: {_GENERATE_EXAMPLE}"
        )
    return float(value)


def generate_slabs(
    bulk: Structure,
    miller_index: Sequence[int],
    min_slab_size: float = 10.0,
    vacuum: float = 15.0,
) -> list[Slab]:
    """Cut every Slab that exposes the ``miller_index`` surface of ``bulk``.

    Args:
        bulk: The bulk Structure to cut. A **conventional** cell is what makes the
            Miller indices mean their usual crystallographic thing and the
            interlayer spacing come out right (CONVENTIONS.md §C.1); the caller's
            object is copied first because ``SlabGenerator`` mutates its input.
        miller_index: Three integers naming the plane family, e.g. ``(1, 1, 1)``.
            At least one must be non-zero; a negative index is accepted and names
            the same plane family.
        min_slab_size: Lower bound on the Slab thickness in angstrom. The
            generator rounds up to whole atomic layers, so the result is at least
            this thick.
        vacuum: The Vacuum in angstrom. This is a *request*: the generator's own
            padding is only a lower bound, so the exact value is imposed afterwards.

    Returns:
        One Slab per Termination, each orthogonalised (c perpendicular to the
        in-plane vectors) with an exactly ``vacuum``-thick Vacuum split equally
        above and below. Ordering is pymatgen's enumeration order and is **not** a
        contract -- identify a Slab by :func:`termination_fingerprint`.

    Raises:
        InvalidMillerIndexError: ``miller_index`` is not three integers, or all
            three are zero. Raised instead of leaking upstream's
            ``ZeroDivisionError`` / ``TypeError`` / unaligned-shape ``ValueError``.
        InvalidParameterError: ``min_slab_size`` or ``vacuum`` is not strictly
            positive. Upstream silently accepts a negative Vacuum and leaks
            ``ValueError: min() iterable argument is empty`` for a zero thickness.
        TypeError: ``bulk`` is not a ``Structure``.

    Example:
        ``generate_slabs(load_structure("Pt.poscar"), (1, 1, 1), 10.0, 15.0)``
        returns one ``Pt5`` Slab.
    """
    # Validate everything before pymatgen runs: it leaks ZeroDivisionError for
    # (0, 0, 0), TypeError for a float index, and ValueError for a zero thickness
    # (CONVENTIONS.md §C.6, §H.2).
    indices = _validate_miller_index(miller_index)
    thickness = _validate_positive(
        min_slab_size,
        "min_slab_size",
        "a positive thickness in Angstrom, typically >= 10.0",
    )
    requested_vacuum = _validate_positive(
        vacuum,
        "vacuum",
        "a positive thickness in Angstrom, typically >= 5.0",
    )
    if not isinstance(bulk, Structure):
        raise TypeError(
            f"bulk={type(bulk).__name__!r} is invalid: expected a "
            f"pymatgen Structure. Valid: a conventional bulk cell, e.g. from "
            f"load_structure('Pt.poscar'). Example: {_GENERATE_EXAMPLE}"
        )

    generator = SlabGenerator(
        # SlabGenerator.__init__ adds bulk_wyckoff/bulk_equivalent site properties
        # to the Structure it is handed; copy so the caller's bulk is untouched.
        bulk.copy(),
        indices,
        thickness,
        requested_vacuum,
        primitive=True,
        reorient_lattice=True,
        center_slab=True,
        lll_reduce=False,
        max_normal_search=None,
    )

    # Both post-processing steps are mandatory (§C.1, §C.3): orthogonalise so the
    # c axis really is perpendicular to the in-plane vectors, then impose the
    # Vacuum exactly -- min_vacuum_size is only a lower bound.
    return [
        set_vacuum(raw.get_orthogonal_c_slab(), requested_vacuum)
        for raw in generator.get_slabs()
    ]


def termination_fingerprint(slab: Slab) -> TerminationFingerprint:
    """Compute the content-based identity of ``slab``'s Termination.

    The terminating layers are found by grouping the sites into atomic layers along
    z (:func:`dsh4sci.geometry.group_z_levels`); the top face is the last group and
    the bottom face the first, each reduced to a sorted ``(element, count)`` tuple
    so the result does not depend on site ordering. The in-plane offset comes from
    ``slab.shift``, quantised to :data:`SHIFT_NDIGITS` decimals.

    Args:
        slab: A Slab, normally one returned by :func:`generate_slabs`.

    Returns:
        The frozen, hashable fingerprint of that Termination.

    Raises:
        ValueError: ``slab`` has no sites, so it has no terminating layer.
        TypeError: ``slab`` is not a ``Slab``.

    Example:
        ``termination_fingerprint(generate_slabs(sic, (1, 1, 1))[0]).sha()``
    """
    if not isinstance(slab, Slab):
        raise TypeError(
            f"slab={type(slab).__name__!r} is invalid: expected a pymatgen Slab. "
            f"Valid: a Slab from generate_slabs(). "
            f"Example: termination_fingerprint(generate_slabs(bulk, (1, 1, 1))[0])"
        )

    layers = group_z_levels(slab, LAYER_TOL)
    if not layers:
        raise ValueError(
            "slab has no sites: it has no terminating layer, so it has no "
            "Termination fingerprint. Valid: a Slab with at least one site. "
            "Example: termination_fingerprint(generate_slabs(bulk, (1, 1, 1))[0])"
        )

    return TerminationFingerprint(
        miller_index=tuple(int(index) for index in slab.miller_index),
        top_layer=_layer_composition(slab, layers[-1]),
        bottom_layer=_layer_composition(slab, layers[0]),
        shift_ndigits=round(float(slab.shift) * 10**SHIFT_NDIGITS),
    )
