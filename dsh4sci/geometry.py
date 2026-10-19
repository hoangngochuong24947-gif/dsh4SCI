"""Shared geometric primitives for slabs and adsorbates.

This module holds the small number of decisions that more than one later module
needs, so that they are made **once**:

- :func:`vacuum_gap` -- what "the Vacuum" numerically means (CONVENTIONS.md §C.3).
- :func:`set_vacuum` -- the only sanctioned way to make that Vacuum exact.
- :func:`count_z_levels` -- how the atomic stack is grouped into layers; the
  Termination fingerprint (§C.5) and the "is this slab-like?" predicate (§E.5)
  both depend on it.
- :data:`SLAB_ROLE` / :data:`ADSORBATE_ROLE` / :func:`split_roles` -- the shared
  decision of which sites are Slab atoms and which are Adsorbate atoms (§G.4).

Vocabulary follows ``CONTEXT.md``: a **Slab** is a finite model of a surface and a
**Vacuum** is the empty gap that makes its periodic repeat non-interacting.

Two conventions are load-bearing and easy to get wrong:

**The Vacuum is measured on Cartesian z, as the largest periodic gap.** The two
obvious alternatives (``c - (z.max() - z.min())`` and ``(1 - ptp(frac_z)) * c``)
silently assume the atomic stack does not cross the periodic boundary along z, and
report the interlayer spacing when it does. Slab operations routinely produce
exactly that situation, so the periodic definition is the only stable one.

**The c axis is the third lattice vector.** After ``reorient_lattice=True`` the
third row of ``lattice.matrix`` is ``[0, 0, c]`` and ``lattice.c`` agrees with it.
No normal is ever recomputed from a Miller index here.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import Slab

from dsh4sci.errors import InvalidParameterError

__all__ = [
    "ADSORBATE_ROLE",
    "DEFAULT_LAYER_TOL",
    "SLAB_ROLE",
    "count_z_levels",
    "group_z_levels",
    "set_vacuum",
    "split_roles",
    "vacuum_gap",
]

#: Site property marking a site as part of the Slab. See §G.4.
SLAB_ROLE: Final[str] = "slab"

#: Site property marking a site as part of the Adsorbate. See §G.4.
ADSORBATE_ROLE: Final[str] = "adsorbate"

#: Default z-gap (A) that separates two distinct atomic layers. §C.5.
DEFAULT_LAYER_TOL: Final[float] = 0.1

_SET_VACUUM_EXAMPLE = "set_vacuum(raw.get_orthogonal_c_slab(), 15.0)"

# Tolerance for comparing a requested Vacuum against the slab's internal
# interlayer gap: the two are floats derived by different arithmetic paths.
_VACUUM_SLACK: Final[float] = 1e-9

# The site property carrying the Slab/Adsorbate split. Its name is part of the
# public contract: callers set it themselves to feed split_roles (§G.4).
_ROLE_PROPERTY: Final[str] = "role"


def vacuum_gap(structure: Structure) -> float:
    """Largest periodic gap between consecutive atomic z-levels, wrap-around included.

    This is the library's definition of the Vacuum (CONTEXT.md): project every site
    onto the c axis, sort the resulting Cartesians, and take the largest successive
    difference -- including the gap from the top of the stack back to the bottom
    through the periodic image (``z[0] + c``).

    The periodic form matters because the atomic stack may straddle the periodic
    boundary. A slab translated along c is physically identical but would make
    ``c - (z.max() - z.min())`` report the interlayer spacing instead of the Vacuum;
    the wrap-around gap is invariant under that translation (CONVENTIONS.md §C.3).

    It is measured on ``cart_coords[:, 2]``, never on fractional z: slab operations
    leave fractional coordinates outside ``[0, 1)`` (``-5.333 .. 0.753`` was
    observed), while Cartesian z stays meaningful.

    Args:
        structure: Any structure with a c axis as its third lattice vector -- a
            Slab, or a bulk crystal (for which this returns the smallest periodic
            layer repeat rather than a Vacuum).

    Returns:
        The largest gap, in angstrom. Zero for an empty structure, and never
        negative.

    Example:
        ``vacuum_gap(slab)`` returns ``15.0`` for a slab built with a 15 A Vacuum.
    """
    if len(structure) == 0:
        return 0.0
    z = np.sort(np.asarray(structure.cart_coords, dtype=float)[:, 2])
    c = float(structure.lattice.c)
    gaps = np.diff(np.concatenate([z, [z[0] + c]]))
    return float(np.max(gaps))


def set_vacuum(slab: Structure, vacuum: float) -> Slab:
    """Return the slab rescaled so its largest periodic z-gap is exactly ``vacuum``.

    ``min_vacuum_size`` is only a *lower bound* upstream: the generator pads to
    whole atomic layers, so a 15 A request produced gaps of 31.385 A (raw),
    18.120 A (after orthogonalisation) and only reaches 15 A after this rescale
    (CONVENTIONS.md §C.1, §C.3). The library therefore owns the Vacuum and applies
    it here.

    The atomic stack itself is **not** distorted: only ``c`` is rescaled and the
    whole stack is translated so that the remaining Vacuum is split equally above
    and below. Three details make that true and each one is a silent bug if
    omitted:

    1. ``c`` alone is scaled (by ``new_c / c``) so the in-plane lattice is
       untouched. Scaling ``c`` while leaving ``frac_coords`` alone moves the atoms
       in space and yields a 16.04 A gap for a 15 A request.
    2. The fractional coordinates are recomputed from the shifted Cartesians
       through the *new* lattice matrix, which is why the Cartesian positions are
       preserved exactly.
    3. The stack is re-centred on ``(new_c - thickness) / 2``. Preserving Cartesian
       z without re-centring gives the right total but an asymmetric 5.94/9.06 A
       split instead of 7.5/7.5 A.

    Args:
        slab: A ``pymatgen.core.Slab`` (its ``site_properties``, ``miller_index``,
            ``shift`` and ``scale_factor`` are carried over unchanged).
        vacuum: The requested Vacuum in angstrom.

    Returns:
        A new ``Slab`` whose :func:`vacuum_gap` equals ``vacuum`` and whose Vacuum
        is split equally above and below the stack.

    Raises:
        InvalidParameterError: ``vacuum`` is not strictly positive. Upstream
            silently accepts a zero or negative value and returns a slab that is
            not physically meaningful.
        TypeError: ``slab`` is not a ``Slab``.

    Example:
        ``set_vacuum(raw.get_orthogonal_c_slab(), 15.0)``
    """
    if isinstance(vacuum, bool) or not isinstance(vacuum, (int, float, np.floating)):
        raise InvalidParameterError(
            f"vacuum={vacuum!r} is invalid: expected a number in angstrom. "
            f"Valid: a positive float, typically >= 5.0. Example: {_SET_VACUUM_EXAMPLE}"
        )
    if not vacuum > 0:
        raise InvalidParameterError(
            f"vacuum={vacuum!r} is invalid: must be > 0. "
            f"Valid: a positive thickness in Angstrom, typically >= 5.0. "
            f"Example: {_SET_VACUUM_EXAMPLE}"
        )
    if not isinstance(slab, Slab):
        raise TypeError(
            f"slab={type(slab).__name__!r} is invalid: expected a pymatgen Slab. "
            f"Valid: a Slab from generate_slabs(). "
            f"Example: {_SET_VACUUM_EXAMPLE}"
        )

    c = float(slab.lattice.c)
    cart = np.asarray(slab.cart_coords, dtype=float).copy()

    # Split the stack at the largest periodic gap: the atom immediately after that
    # gap is the *bottom* of the stack, and the wrap-around gap is its Vacuum. This
    # is periodic, so it holds whether or not the stack straddles the boundary --
    # a bare `z.min()` would pick the wrapped atom instead of the real bottom and
    # then centre the stack on the wrong atom.
    z = cart[:, 2]
    order = np.argsort(z, kind="stable")
    z_sorted = z[order]
    gaps = np.diff(np.concatenate([z_sorted, [z_sorted[0] + c]]))
    gap = float(gaps.max())
    gap_index = int(np.argmax(gaps))
    thickness = c - gap
    # Unwrap so every site's z is >= the stack bottom (a lattice translation, so
    # the physical structure is unchanged) and the stack is contiguous.
    bottom = float(z_sorted[(gap_index + 1) % len(z_sorted)])
    cart[:, 2] = bottom + ((z - bottom) % c)

    # The Vacuum can only be *measured* as the largest periodic gap if it is at
    # least as large as the slab's own largest internal gap. Below that, the
    # periodic gap saturates at the internal spacing and the function would return
    # a plausible-looking wrong number. Refuse instead.
    if thickness > 0:
        internal = _largest_internal_gap(z_sorted, c, gap_index)
        if float(vacuum) < internal - _VACUUM_SLACK:
            raise InvalidParameterError(
                f"vacuum={vacuum!r} is invalid: below the slab's own "
                f"{internal:.3f} A interlayer gap, so the periodic gap would "
                f"saturate. Valid: > {internal:.3f}, typically >= 5.0. "
                f"Example: {_SET_VACUUM_EXAMPLE}"
            )

    new_c = thickness + float(vacuum)

    # Scale c only: the in-plane lattice and the stack's internal geometry are
    # facts about the physics, not about how much empty space we wrap around it.
    matrix = np.array(slab.lattice.matrix, dtype=float).copy()
    matrix[2] *= new_c / c

    # Centre the contiguous stack in the new cell so the Vacuum is split equally.
    cart[:, 2] += (new_c - thickness) / 2.0 - bottom

    new_lattice = Lattice(matrix)
    new_frac = new_lattice.get_fractional_coords(cart)

    return Slab(
        lattice=new_lattice,
        species=slab.species_and_occu,
        coords=new_frac,
        miller_index=slab.miller_index,
        oriented_unit_cell=slab.oriented_unit_cell,
        shift=slab.shift,
        scale_factor=slab.scale_factor,
        reorient_lattice=slab.reorient_lattice,
        site_properties=slab.site_properties,
    )


def _largest_internal_gap(z_sorted: np.ndarray, c: float, gap_index: int) -> float:
    """Largest gap *inside* the atomic stack, i.e. excluding the Vacuum itself.

    ``z_sorted`` holds the sorted Cartesians of one structure; ``gap_index`` is the
    index of the widest of the periodic gaps (the Vacuum, or ``len - 1`` for the
    wrap-around). Every other gap lies between two consecutive layers of the stack.
    """
    gaps = np.diff(np.concatenate([z_sorted, [z_sorted[0] + c]]))
    internal = [g for i, g in enumerate(gaps) if i != gap_index]
    return float(max(internal)) if internal else 0.0


def group_z_levels(
    structure: Structure, tol: float = DEFAULT_LAYER_TOL
) -> list[list[int]]:
    """Group site indices into atomic layers along z, bottom layer first.

    Sites are sorted by ``cart_coords[:, 2]`` and cut wherever the successive gap
    **exceeds** ``tol``; each group is one atomic layer (CONVENTIONS.md §C.5). A
    group is never empty, and the groups partition ``range(len(structure))``.

    Args:
        structure: The structure whose sites are grouped.
        tol: The z-gap in angstrom above which two sites belong to different
            layers. The default 0.1 A resolves ``d(111) = 2.265 A`` layers and the
            ``0.629 A`` Si-C bilayer gap, while collapsing nothing that belongs
            together.

    Returns:
        One list of site indices per layer, ordered from the lowest z upwards.

    Example:
        ``len(group_z_levels(slab))`` is the slab's layer count.
    """
    if tol < 0:
        raise InvalidParameterError(
            f"tol={tol!r} is invalid: must be >= 0. "
            f"Valid: a non-negative layer gap in Angstrom, typically 0.1. "
            f"Example: count_z_levels(slab, tol=0.1)"
        )
    n_sites = len(structure)
    if n_sites == 0:
        return []

    order = sorted(
        range(n_sites), key=lambda i: float(structure.cart_coords[i, 2])
    )
    levels: list[list[int]] = [[order[0]]]
    for previous, current in zip(order, order[1:]):
        z_previous = float(structure.cart_coords[previous, 2])
        z_current = float(structure.cart_coords[current, 2])
        if z_current - z_previous > tol:
            levels.append([current])
        else:
            levels[-1].append(current)
    return levels


def count_z_levels(structure: Structure, tol: float = DEFAULT_LAYER_TOL) -> int:
    """Number of distinct atomic layers along z.

    This is what "an N-layer slab" means (CONVENTIONS.md §D.4): for fcc Pt(111) a
    4-layer slab has 4 atoms, but for SiC(111) the primitive surface cell holds two
    atoms per Si-C bilayer, so the layer count must be measured rather than inferred
    from the atom count.

    Args:
        structure: The structure to measure.
        tol: The layer-separating z-gap in angstrom; see :func:`group_z_levels`.

    Returns:
        The number of distinct z-levels; ``0`` for an empty structure.

    Example:
        ``count_z_levels(generate_slabs(pt, (1, 1, 1))[0])`` is ``5``.
    """
    return len(group_z_levels(structure, tol))


def split_roles(structure: Structure) -> tuple[list[int], list[int]]:
    """Partition site indices into Slab atoms and Adsorbate atoms by the ``role`` tag.

    Species membership cannot make this call -- an OH adsorbate on a TiO2 slab
    shares O with the slab -- and neither can a z threshold, because the failure
    this exists to detect is an Adsorbate atom *below* the slab top (CONVENTIONS.md
    §G.4). A ``role`` site property with exactly the values :data:`SLAB_ROLE` and
    :data:`ADSORBATE_ROLE` can, and it survives ``copy()`` and ``make_supercell()``.

    The tag is a convenience for in-memory work, **not** a source of truth: it is
    lost on POSCAR/CIF round-trips, so anything that must survive file I/O carries
    the slab atom count explicitly alongside the structure.

    Args:
        structure: A structure tagged by e.g. ``place_adsorbate``.

    Returns:
        ``(slab_indices, adsorbate_indices)``, each in ascending site order. Either
        may be empty if the tag says so.

    Raises:
        ValueError: The structure has no ``role`` site property, or it holds a
            value other than :data:`SLAB_ROLE` / :data:`ADSORBATE_ROLE`.

    Example:
        ``slab_idx, ads_idx = split_roles(combined)``
    """
    roles = structure.site_properties.get(_ROLE_PROPERTY)
    if roles is None:
        raise ValueError(
            "structure has no 'role' site property; cannot separate Slab from "
            "Adsorbate. Valid: a Structure from place_adsorbate(), or set "
            "site_properties['role'] to 'slab'/'adsorbate'. Example: "
            "s.add_site_property('role', ['slab']*5 + ['adsorbate']*2)"
        )

    valid = {SLAB_ROLE, ADSORBATE_ROLE}
    unknown = sorted({str(role) for role in roles} - valid)
    if unknown:
        raise ValueError(
            f"role={unknown!r} is invalid: a role must be one of "
            f"{sorted(valid)}. Valid: 'slab' for Slab atoms, 'adsorbate' for "
            f"Adsorbate atoms. Example: "
            f"s.add_site_property('role', ['slab']*5 + ['adsorbate']*2)"
        )

    slab_indices: list[int] = []
    adsorbate_indices: list[int] = []
    for index, role in enumerate(roles):
        if str(role) == SLAB_ROLE:
            slab_indices.append(index)
        else:
            adsorbate_indices.append(index)
    return slab_indices, adsorbate_indices
