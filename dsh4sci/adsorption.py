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

**Placing an Adsorbate on one of those sites** is the other half of the module.
:func:`place_adsorbate` binds a molecule through a named element, by *symbol* --
never by index, because ``ase.build.molecule`` orders atoms as it likes
(``["O", "C"]`` for CO, so index 0 is oxygen) -- orients it along +z by rotating
the anchor-relative frame before translating it, and puts the anchor at the site's
own x, y and ``height`` above the Slab's top layer. The result is a
:class:`PlacedAdsorbate`: the combined structure tagged with the ``role`` site
property, the Slab's atom count, and the anchor's index. Both of the last two exist
because the tag alone is not durable -- it survives ``copy()`` and
``make_supercell()`` but is lost through POSCAR/CIF (§G.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
from ase.build import molecule as ase_molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Element, Structure
from pymatgen.core.surface import Slab

from dsh4sci.errors import InvalidParameterError, NotASlabError
from dsh4sci.geometry import (
    ADSORBATE_ROLE,
    DEFAULT_LAYER_TOL,
    ROLE_PROPERTY,
    SLAB_ROLE,
    count_z_levels,
    vacuum_gap,
)

__all__ = [
    "DEFAULT_BINDING_HEIGHT",
    "MIN_VACUUM",
    "MIN_Z_LEVELS",
    "SITE_TYPES",
    "AdsorptionSite",
    "PlacedAdsorbate",
    "SiteType",
    "find_adsorption_sites",
    "is_slab_like",
    "place_adsorbate",
]

#: Default anchor-to-surface distance (A) when placing an Adsorbate (§F.3).
DEFAULT_BINDING_HEIGHT: Final[float] = 1.85

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

# The direction an Adsorbate is oriented along: straight out of the surface.
_UP: Final[np.ndarray] = np.array([0.0, 0.0, 1.0])

# The site property selective dynamics is carried in (pymatgen's name for it).
_SELECTIVE_DYNAMICS: Final[str] = "selective_dynamics"

# An Adsorbate atom is free to move on all three axes: it is never part of the
# Slab's fixed bottom layer.
_FREE: Final[tuple[bool, bool, bool]] = (True, True, True)

# One valid call, reused by every message in this module (§H.1 rule 5).
_FIND_EXAMPLE: Final[str] = "find_adsorption_sites(generate_slabs(bulk, (1, 1, 1))[0])"
_PLACE_EXAMPLE: Final[str] = (
    "place_adsorbate(slab, site, molecule='CO', anchor='C', height=1.85)"
)


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


@dataclass(frozen=True)
class PlacedAdsorbate:
    """A Slab with an Adsorbate on it: the Structure plus what its tag cannot say.

    The ``role`` site property on :attr:`structure` is the convenient way to tell
    the Slab from the Adsorbate in memory, and it is not durable: it is dropped by
    a POSCAR or CIF round-trip (§G.4). After file I/O, :attr:`n_slab_atoms` is the
    source of truth -- the first ``n_slab_atoms`` sites are the Slab and the rest
    are the Adsorbate.

    Attributes:
        structure: The combined structure -- every Slab site first, then one site
            per Adsorbate atom in :func:`ase.build.molecule` order, tagged with the
            ``role`` site property.
        n_slab_atoms: How many leading sites belong to the Slab. Returned
            separately, rather than read back off the tag, so it survives file I/O.
        anchor_index: Index of the anchor atom in ``structure`` -- the atom that
            binds, for a check that compares its height with the Slab's top layer.
    """

    structure: Structure
    n_slab_atoms: int
    anchor_index: int


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


def place_adsorbate(
    slab: Structure,
    site: AdsorptionSite,
    *,
    molecule: str,
    anchor: str,
    height: float = DEFAULT_BINDING_HEIGHT,
) -> PlacedAdsorbate:
    """Put an Adsorbate on one Adsorption site, anchored by element symbol.

    The Adsorbate binds through ``anchor``, which sits at the site's own in-plane
    position: its x and y are ``site.position[:2]`` -- the *site's*, not the top
    atom's, which coincide only for an ``ontop`` site on a real top-layer atom --
    and its z is the Slab's top layer plus ``height``. Because
    :func:`find_adsorption_sites` fixes ``distance=0.0``, a site *is* the top-layer
    plane, so ``height`` is the anchor-to-surface distance: 1.85 A by default, and
    the anchor's nearest-neighbour distance on an ``ontop`` site is exactly it.

    **The anchor is found by element symbol, never by index.** ``ase.build.
    molecule`` orders atoms as it likes -- ``['O', 'C']`` for CO, ``['N', 'O']`` for
    NO (§F.1) -- so anchoring ``atoms[0]`` binds oxygen instead of carbon and
    leaves the molecule standing on its head.

    **The molecule is oriented before it is translated.** The anchor-relative frame
    is rotated so the anchor -> (mean of the other atoms) direction points along
    +z, i.e. out of the surface. Doing it the other way round -- anchoring first
    and flipping a z component afterwards -- silently inverts the molecule
    (measured: ``O_z - C_z = -1.1503`` for CO, §F.2).

    **Two things travel with the result, and they are not interchangeable.** The
    :data:`~dsh4sci.geometry.ROLE_PROPERTY` site property tags every site
    :data:`~dsh4sci.geometry.SLAB_ROLE` or
    :data:`~dsh4sci.geometry.ADSORBATE_ROLE` and survives ``copy()`` and
    ``make_supercell()``; it is **lost through POSCAR/CIF**, so for anything that
    must survive file I/O use :attr:`PlacedAdsorbate.n_slab_atoms`, the count of
    leading Slab atoms, which is why it is returned separately (§G.4). The
    ``selective_dynamics`` property of a Slab prepared by
    :func:`dsh4sci.constraints.fix_bottom_atoms` is carried over, with every
    Adsorbate atom free.

    Every site already in ``slab`` is tagged as Slab, so placement takes a Slab, not
    a structure that already carries an Adsorbate: place one Adsorbate per call.

    Args:
        slab: The Slab to adsorb onto -- a Slab-like ``Structure``, as returned by
            :func:`dsh4sci.slabs.generate_slabs`. It is not modified.
        site: The Adsorption site to bind on, from
            :func:`find_adsorption_sites` (or a :class:`AdsorptionSite` built by
            hand for a position of your own).
        molecule: An ASE ``g2`` molecule name, e.g. ``"CO"``, ``"OH"``, ``"H2O"``.
        anchor: Element symbol of the atom that binds to the surface, e.g. ``"C"``
            for CO and ``"O"`` for OH. Must be one of the molecule's elements.
        height: Anchor-to-surface distance in angstrom. Must be a finite, positive
            length; 1.85 A is the default.

    Returns:
        The combined :class:`PlacedAdsorbate`: the structure, its Slab atom count
        and the anchor's index. The Adsorbate's sites follow the Slab's in ASE
        molecule order, and their Cartesian coordinates are the ones that put the
        anchor on the site -- so they are reported in the frame of the Slab with c
        already perpendicular to its in-plane vectors (see the module docstring),
        and they may lie outside the unit cell along z.

    Raises:
        TypeError: ``slab`` is not a pymatgen ``Structure``, or ``site`` is not an
            :class:`AdsorptionSite`.
        InvalidParameterError: ``molecule`` is not an ASE ``g2`` name, ``anchor`` is
            not one of its elements, or ``height`` is not a finite positive length.
            Each message names the parameter, the rejected value, the valid set and
            a valid call (§H).

    Example:
        ``sites = find_adsorption_sites(slab);``
        ``placed = place_adsorbate(slab, sites[0], molecule="CO", anchor="C")``
    """
    if not isinstance(slab, Structure):
        raise TypeError(
            f"slab={type(slab).__name__!r} is invalid: expected a pymatgen "
            f"Structure. Valid: a Slab-like Structure, e.g. from "
            f"generate_slabs(). Example: {_PLACE_EXAMPLE}"
        )
    if not isinstance(site, AdsorptionSite):
        raise TypeError(
            f"site={site!r} is invalid: expected an AdsorptionSite, got "
            f"{type(site).__name__}. Valid: a find_adsorption_sites() record. "
            f"Example: {_PLACE_EXAMPLE}"
        )
    if not (isinstance(height, (int, float)) and math.isfinite(height) and height > 0):
        raise InvalidParameterError(
            f"height={height!r} is invalid: must be a finite float > 0. Valid: an "
            f"anchor-surface distance in Angstrom, e.g. 1.85. "
            f"Example: {_PLACE_EXAMPLE}"
        )

    symbols, positions = _load_molecule(molecule)
    anchor_index_local = _anchor_index(symbols, anchor, molecule)
    relative = _pointing_away(
        positions - positions[anchor_index_local], anchor_index_local
    )

    base = _orthogonalised(slab)
    top_z = float(np.max(np.asarray(base.cart_coords)[:, 2]))
    anchor_cart = np.array(
        [site.position[0], site.position[1], top_z + height], dtype=float
    )
    cart = relative + anchor_cart
    cart[anchor_index_local] = anchor_cart

    fractional = base.lattice.get_fractional_coords(cart)
    species = [base_site.species for base_site in base] + [
        Element(symbol) for symbol in symbols
    ]
    structure = Structure(
        base.lattice,
        species,
        np.vstack([np.asarray(base.frac_coords), fractional]),
        site_properties=_combined_properties(base, len(symbols)),
    )
    return PlacedAdsorbate(
        structure=structure,
        n_slab_atoms=len(base),
        anchor_index=len(base) + anchor_index_local,
    )


def _load_molecule(molecule: str) -> tuple[list[str], np.ndarray]:
    """The molecule's element symbols and Cartesian positions, in ASE's own order.

    ASE's ``molecule`` raises a bare ``KeyError`` for an unknown name (§H), which
    tells an agent nothing about the valid set or what to call instead.
    """
    try:
        atoms = ase_molecule(molecule)
    except KeyError as exc:
        raise InvalidParameterError(
            f"molecule={molecule!r} is not a known ASE molecule. Valid: an ASE "
            f"`g2` name such as 'CO', 'OH', 'H2O'. Example: {_PLACE_EXAMPLE}"
        ) from exc
    return (
        list(atoms.get_chemical_symbols()),
        np.asarray(atoms.get_positions(), dtype=float),
    )


def _anchor_index(symbols: list[str], anchor: str, molecule: str) -> int:
    """Index of the atom that binds, found by symbol -- never by position in the list.

    ``list.index`` raises ``'N' is not in list``, which names neither the parameter
    nor the valid set (§H).
    """
    if anchor not in symbols:
        elements = sorted(set(symbols))
        raise InvalidParameterError(
            f"anchor={anchor!r} is not present in Adsorbate {molecule!r} "
            f"(elements: {elements}). Valid: one of {elements}. "
            f"Example: {_PLACE_EXAMPLE}"
        )
    return symbols.index(anchor)


def _pointing_away(relative: np.ndarray, anchor_index: int) -> np.ndarray:
    """Rotate an anchor-centred molecule so it points along +z.

    ``relative`` holds the molecule's atoms with the anchor at the origin. The
    rotation is chosen to take the anchor -> (mean of the other atoms) direction
    onto :data:`_UP`, which points the bulk of the molecule away from the surface
    instead of through it. It is applied **before** the molecule is translated
    onto the site: re-anchoring first and flipping a z component afterwards
    silently inverts the molecule (measured ``O_z - C_z = -1.1503`` for CO, §F.2).

    A one-atom molecule, or one whose other atoms are centred on the anchor, has no
    direction to align and is returned unrotated; an exactly antiparallel target is
    a 180 deg rotation about any axis perpendicular to it, for which ``-I`` is the
    degenerate rotation matrix.
    """
    others = [index for index in range(len(relative)) if index != anchor_index]
    if not others:
        return relative
    target = relative[others].mean(axis=0)
    norm = float(np.linalg.norm(target))
    if norm < 1e-9:
        return relative
    target = target / norm
    if np.allclose(target, _UP):
        return relative
    cross = np.cross(target, _UP)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(target, _UP))
    if sine < 1e-12:
        rotation = np.eye(3) if cosine > 0 else -np.eye(3)
    else:
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ]
        )
        rotation = np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
    return relative @ rotation.T


def _combined_properties(
    slab: Structure, n_adsorbate: int
) -> dict[str, list[Any]]:
    """Per-atom site properties for the combined structure, Slab first.

    Two properties are written rather than inherited:

    * :data:`~dsh4sci.geometry.ROLE_PROPERTY` -- :data:`SLAB_ROLE` for every Slab
      site, :data:`ADSORBATE_ROLE` for every Adsorbate site (§G.4). It is the only
      tag that can separate the two when they share an element, and it survives
      ``copy()`` and ``make_supercell()`` -- but not a POSCAR/CIF round-trip, which
      is why :attr:`PlacedAdsorbate.n_slab_atoms` is returned as well.
    * ``selective dynamics``, when the Slab carries it -- an Adsorbate atom is free
      on all three axes. Filling it with ``None`` instead would make the POSCAR
      writer raise ``TypeError: 'NoneType' object is not iterable`` (measured), so
      the flags of a Slab prepared by :func:`dsh4sci.constraints.fix_bottom_atoms`
      survive placement with the Adsorbate left free.

    Any other property the Slab carries is kept for the Slab sites and holds no
    value for the Adsorbate's, where pymatgen's own convention for an appended
    site is ``None``.
    """
    properties: dict[str, list[Any]] = {
        key: list(values) + [None] * n_adsorbate
        for key, values in slab.site_properties.items()
    }
    if _SELECTIVE_DYNAMICS in slab.site_properties:
        properties[_SELECTIVE_DYNAMICS] = (
            list(slab.site_properties[_SELECTIVE_DYNAMICS]) + [_FREE] * n_adsorbate
        )
    properties[ROLE_PROPERTY] = (
        [SLAB_ROLE] * len(slab) + [ADSORBATE_ROLE] * n_adsorbate
    )
    return properties


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

    ``scale_factor`` is contracted as a 3x3 matrix (the oriented unit cell -> the
    surface cell), so its neutral placeholder is the identity matrix rather than
    the scalar ``1.0``: a float there violates the ``Slab`` type contract, and any
    later caller that reads the field would get a scalar where a matrix is
    promised.
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
        scale_factor=np.eye(3),
        site_properties=structure.site_properties,
    )
