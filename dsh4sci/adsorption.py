"""Adsorption site enumeration: where an Adsorbate could sit on a Slab.

An **Adsorption site** is a position on a Slab's surface where an Adsorbate could
bind (CONTEXT.md). :func:`find_adsorption_sites` enumerates them and normalises
upstream's output into this project's vocabulary -- :data:`SITE_TYPES`, exactly
``ontop`` (above one atom), ``bridge`` (between two) and ``hollow`` (inside a
triangle of three or more). No alias, no raw upstream key, and never the ``all``
key upstream adds as a concatenation of the other three (§E.4): a caller that
iterated it would double-count every site.

**The return type is a list of typed records.** Tickets after this one must be
able to take one site and place an Adsorbate on it, and a plain
``dict[str, list[coordinates]]`` loses the type of a site once it is taken out of
the dict. :class:`AdsorptionSite` carries both the :attr:`~AdsorptionSite.site_type`
and the :attr:`~AdsorptionSite.position`, so
``[site for site in sites if site.site_type == "hollow"]`` names a site and
``site.position`` is what ``place_adsorbate`` (ticket #6) needs. The list is
ordered by :data:`SITE_TYPES` then by upstream's order within a type; **only the
type grouping is a contract**, not the order inside a type.

**Three upstream settings are fixed here, not by the caller** (CONVENTIONS.md §E):

1. ``symm_reduce=-1`` -- enumerate *every* site. The default ``symm_reduce=0.01``
   collapses symmetry-equivalent sites, which is exactly why the naive call never
   scales with area: Pt(111) 2x2, 3x3 and 4x4 all return ``{ontop: 1, bridge: 1,
   hollow: 2}`` instead of ``{4, 12, 8}``, ``{9, 27, 18}``, ``{16, 48, 32}`` (§D3).
2. ``distance=0.0`` -- do not lift the sites. ``distance`` displaces every site
   along the surface normal *after* it is computed (default 2.0 A), which is not
   the same thing as ``height``: ``height`` (on the finder, default 0.9 A) decides
   *which atoms count as surface* and moving it moves the site sideways through
   the stack instead (§E.2). With ``distance=0`` a site sits on the top layer and
   the caller owns the offset.
3. ``height=0.9``, passed explicitly so an upstream default change cannot silently
   pull the second layer into "surface".

**The c axis is forced orthogonal to the in-plane vectors first** (§D2, §E.1).
``generate_slabs`` already returns orthogonal slabs, and for those the
orthogonalisation is a no-op (measured: Cartesian positions agree to 2e-15 A), but
a Slab built elsewhere may keep ``alpha = beta = 135 deg`` with ``a_z = b_z =
-1.96155 A``. On such a slab the site search reports ``ontop`` 1.96 A *below* the
top layer and a supercell smears 5 z-levels into 13. **Every position returned is
Cartesian in the frame of ``structure.get_orthogonal_c_slab()``**, which is the
caller's own frame whenever the caller's Slab is already orthogonal.

**Rejecting input that is not a Slab** uses the calibrated predicate in
:func:`is_slab_like` (§E.5): a largest periodic z-gap of at least
:data:`MIN_VACUUM` (5.0 A) *and* at least :data:`MIN_Z_LEVELS` atomic z-levels.
5.0 A separates every real slab measured from every bulk crystal measured,
including the layered ones (graphite 3.36, h-BN 3.33, MoS2 2.97). It is a
geometric test, not ``Structure.is_valid()``, which is a minimum-distance check
and would accept a zero-volume cell (§D9). Caveat: a bulk crystal whose periodic
repeat exceeds 5 A along c *and* holds two distinct z-levels would be accepted --
a false accept, which is the safe direction for an error; the caller is told the
Vacuum it measured, so the mistake is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure
from pymatgen.core.surface import Slab

from dsh4sci.errors import NotASlabError
from dsh4sci.geometry import DEFAULT_LAYER_TOL, count_z_levels, vacuum_gap

__all__ = [
    "MIN_VACUUM",
    "MIN_Z_LEVELS",
    "SITE_TYPES",
    "AdsorptionSite",
    "SiteType",
    "find_adsorption_sites",
    "is_slab_like",
]

#: The closed set of Adsorption site types, in output order (§E.4).
SiteType = Literal["ontop", "bridge", "hollow"]

#: The Adsorption site types this library exposes, in output order (§E.4).
#: Lowercase and unaliased on purpose: these strings are the vocabulary.
SITE_TYPES: Final[tuple[SiteType, ...]] = ("ontop", "bridge", "hollow")

#: Smallest Vacuum (A) at which a structure still counts as a Slab (§E.5).
MIN_VACUUM: Final[float] = 5.0

#: Fewest atomic z-levels a structure needs to count as a Slab (§E.5).
MIN_Z_LEVELS: Final[int] = 2

# How far below the topmost atom an atom may sit and still count as "surface"
# (§E.2). Passed explicitly: it must not move with an upstream default.
_SURFACE_HEIGHT: Final[float] = 0.9

# One valid call, reused by every message in this module (§H.1 rule 5).
_FIND_EXAMPLE: Final[str] = "find_adsorption_sites(generate_slabs(bulk, (1, 1, 1))[0])"


@dataclass(frozen=True, order=True)
class AdsorptionSite:
    """One position where an Adsorbate could bind, with the site type that names it.

    Frozen and order-comparable so a list of sites can be sorted deterministically
    and compared in tests without comparing floats by eye.

    Attributes:
        site_type: ``"ontop"``, ``"bridge"`` or ``"hollow"`` -- always one of
            :data:`SITE_TYPES`, never an upstream alias and never ``"all"``.
        position: ``(x, y, z)`` Cartesian coordinates in angstrom, in the frame of
            ``structure.get_orthogonal_c_slab()`` (see the module docstring). This
            is a tuple rather than an array so the record stays hashable and two
            sites compare by value.

    Example:
        ``x, y, z = site.position`` is what ``place_adsorbate`` anchors on.
    """

    site_type: SiteType
    position: tuple[float, float, float]

    def __str__(self) -> str:
        x, y, z = self.position
        return f"{self.site_type}@({x:.6f}, {y:.6f}, {z:.6f})"


def is_slab_like(
    structure: Structure,
    min_vacuum: float = MIN_VACUUM,
    min_layers: int = MIN_Z_LEVELS,
) -> tuple[bool, list[str]]:
    """Decide whether a structure is a Slab, and say precisely why not if it is not.

    Both conditions must hold (§E.5), and both are measured with the shared
    geometry primitives rather than re-derived:

    1. the largest periodic z-gap (:func:`dsh4sci.geometry.vacuum_gap`) is at least
       ``min_vacuum`` -- i.e. there is a Vacuum, not just the crystal's own
       interlayer repeat;
    2. there are at least ``min_layers`` distinct atomic z-levels
       (:func:`dsh4sci.geometry.count_z_levels`) -- i.e. a stack of layers, not a
       unit cell with one.

    Returning the reasons, not a bare boolean, is what lets the caller's error
    message name the measured quantity and how far off it is (§H).

    Args:
        structure: The structure to classify.
        min_vacuum: Smallest Vacuum in angstrom that still counts as a Slab. 5.0 A
            is the calibrated discriminator; every bulk crystal measured is below
            it and every real slab measured is above it.
        min_layers: Fewest atomic z-levels that still counts as a Slab.

    Returns:
        ``(is_slab_like, reasons)``. ``reasons`` is empty exactly when the first
        element is ``True``; otherwise each entry is one violated condition.

    Example:
        ``is_slab_like(generate_slabs(bulk, (1, 1, 1))[0])`` reports ``(True, [])``.
    """
    gap = vacuum_gap(structure)
    levels = count_z_levels(structure, DEFAULT_LAYER_TOL)

    reasons: list[str] = []
    if gap < min_vacuum:
        reasons.append(f"Vacuum {gap:.3f} A < {min_vacuum} A")
    if levels < min_layers:
        reasons.append(f"only {levels} z-levels < {min_layers}")
    return (not reasons, reasons)


def find_adsorption_sites(structure: Structure) -> list[AdsorptionSite]:
    """Enumerate every Adsorption site on a Slab, in this project's vocabulary.

    A site is *ontop* when one atom lies under it, *bridge* when it sits between
    two, and *hollow* when it sits inside a triangle of three or more (§E.3). On
    fcc Pt(111) that yields a count per in-plane cell of 1 : 3 : 2 for the primitive
    cell, and every type's count scales exactly with the surface area --
    ``n^2`` times the primitive counts for an ``n x n`` supercell -- because
    symmetry reduction is switched off (§D3).

    Supercelling is the caller's job and must happen **after** the slab is
    orthogonal: supercell an orthogonal slab (``generate_slabs`` output already is
    one) and pass the result here. Every returned position lies at or above the
    slab's top z-level; with the fixed ``distance=0.0`` it lies exactly *on* it, so
    the caller applies the binding height when placing the Adsorbate (§E.2).

    Args:
        structure: The Slab to enumerate. A ``Slab`` from
            :func:`dsh4sci.slabs.generate_slabs`, or any ``Structure`` that is
            slab-like by :func:`is_slab_like` (a Slab read back from a file is a
            plain ``Structure``).

    Returns:
        One :class:`AdsorptionSite` per site, grouped by :data:`SITE_TYPES` in that
        order. The order inside a type follows upstream and is not a contract.

    Raises:
        NotASlabError: The structure is not slab-like -- no Vacuum of at least
            :data:`MIN_VACUUM`, or too few z-levels -- so "adsorption site" has no
            meaning. The message names the measured Vacuum and the threshold. This
            is rejected rather than silently "fixed" (§E.5).
        TypeError: ``structure`` is not a pymatgen ``Structure`` at all.

    Example:
        ``sites = find_adsorption_sites(slab); sites[0].position``
    """
    if not isinstance(structure, Structure):
        raise TypeError(
            f"structure={type(structure).__name__!r} is invalid: expected a "
            f"pymatgen Structure. Valid: a Slab-like Structure, e.g. from "
            f"generate_slabs(). Example: {_FIND_EXAMPLE}"
        )

    slab_like, reasons = is_slab_like(structure)
    if not slab_like:
        raise NotASlabError(
            f"structure is not slab-like: {'; '.join(reasons)}. "
            f"Valid: a Slab with >= {MIN_VACUUM} A Vacuum along c and "
            f">= {MIN_Z_LEVELS} atomic z-levels. Example: {_FIND_EXAMPLE}"
        )

    raw = AdsorbateSiteFinder(
        _orthogonalised(structure), height=_SURFACE_HEIGHT
    ).find_adsorption_sites(distance=0.0, symm_reduce=-1)

    sites: list[AdsorptionSite] = []
    for site_type in SITE_TYPES:
        for position in raw[site_type]:
            sites.append(
                AdsorptionSite(
                    site_type,
                    (float(position[0]), float(position[1]), float(position[2])),
                )
            )
    return sites


def _orthogonalised(structure: Structure) -> Slab:
    """The Slab with c perpendicular to its in-plane vectors (§D2).

    The mandatory first step, in one place so that anything else operating on the
    same sites (ticket #6) reaches the same frame. Calling it on an already
    orthogonal slab is a no-op up to float noise, and it never mutates the input:
    the result is a new object.
    """
    return _as_slab(structure).get_orthogonal_c_slab()


def _as_slab(structure: Structure) -> Slab:
    """Present any slab-like ``Structure`` as a ``Slab`` so the shared step can run.

    ``get_orthogonal_c_slab`` lives on ``Slab``, but a Slab read back from a file is
    a plain ``Structure``. The orthogonalisation reads only the lattice and the
    sites, so the ``Slab``-only metadata (Miller index, shift, oriented unit cell)
    is filled with neutral placeholders and is not observable in the result.
    """
    if isinstance(structure, Slab):
        return structure
    return Slab(
        lattice=structure.lattice,
        species=structure.species_and_occu,
        coords=structure.frac_coords,
        miller_index=(0, 0, 1),
        oriented_unit_cell=structure,
        shift=0.0,
        scale_factor=1.0,
        site_properties=structure.site_properties,
    )
