"""The structure self-check: pass/fail rules for a structure that looks right.

A structure can be geometrically well formed -- every site has a species, the
lattice parses, pymatgen happily writes it out -- and still be physically wrong.
The motivating failure is real: an Adsorbate buried *inside* the Slab satisfies
every schema rule and is useless. This module states the physics rules that catch
that, as code rather than as comments, so a dependency upgrade cannot silently
drop them.

**The result of a check is a :class:`CheckResult`, and the result of a self-check
is a list of them -- never a bare boolean.** A caller that only learns "invalid"
has to re-measure to find out what is wrong; a caller that receives the name, the
flag and the measured detail can report it (CONVENTIONS.md §G). :func:`all_passed`
answers the boolean question when that is all the caller wants.

The three rules, each measured rather than assumed:

1. :func:`check_min_distance` -- the closest pair of atoms, **over periodic
   images**, must be further apart than 0.5 A. Two atoms at fractional z 0.001 and
   0.999 in a c = 3.9231 A cell are 0.0078 A apart; a naive Cartesian check reads
   3.9153 A and sees nothing (§G.1).
2. :func:`check_adsorbate_anchor_above_surface` -- the Adsorbate's anchor atom must
   sit at or above the Slab's top layer. This is the motivating failure, and it
   needs the shared Slab/Adsorbate split (:func:`dsh4sci.geometry.split_roles`),
   because neither species membership nor a z threshold can make that call (§G.2,
   §G.4).
3. :func:`check_lattice_angles` -- every lattice angle must be a real angle in
   ``[5, 175]`` degrees *and* the cell must have a finite, non-zero volume.
   ``Structure.is_valid()`` is the wrong tool: it is a 0.5 A distance check and
   returns ``True`` for a zero-volume cell (§G.3, D9). Angles of 60 and 120 degrees
   are normal for hexagonal surface cells and must pass.

Thresholds are constants of the physics, not tunables, but a caller may pass a
different one where a system justifies it (:data:`MIN_INTERATOMIC_DISTANCE`).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
from pymatgen.core import Structure

__all__ = [
    "ANGLE_MAX",
    "ANGLE_MIN",
    "MIN_CELL_VOLUME",
    "MIN_INTERATOMIC_DISTANCE",
    "CheckResult",
    "check_lattice_angles",
    "check_min_distance",
]

#: Closest allowed interatomic distance; ``<=`` this value fails (§G.1).
#:
#: 0.5 A is not arbitrary and must not be raised: a correct CO/Pt(111) has a
#: closest pair of 1.15034 A -- the intramolecular C-O bond -- so anything near
#: that would reject a chemically sound structure.
MIN_INTERATOMIC_DISTANCE: Final[float] = 0.5

#: Smallest valid lattice angle, degrees. Below this the cell is collapsed (§G.3).
ANGLE_MIN: Final[float] = 5.0

#: Largest valid lattice angle, degrees. ``gamma = 177`` is *inside* this range and
#: still leaves no volume -- which is why the lattice check also tests volume.
ANGLE_MAX: Final[float] = 175.0

#: Smallest valid cell volume, cubic angstrom. The skewed test cell measures ~4e-7.
MIN_CELL_VOLUME: Final[float] = 1e-6


@dataclass(frozen=True)
class CheckResult:
    """One check's verdict: what was checked, whether it passed, what was measured.

    Frozen because a verdict is evidence: a caller may collect results from
    several structures and report them, and a mutated verdict would be a lie.

    Attributes:
        name: The check's stable identifier, e.g. ``"minimum_interatomic_distance"``.
            Lowercase snake case, unique per check, and part of the public contract
            -- callers select failures by name rather than by parsing prose.
        passed: Whether the structure satisfies the rule.
        detail: One line stating the measured quantity, and how it missed the
            threshold when it did -- e.g. the distance and the two atoms involved.
    """

    name: str
    passed: bool
    detail: str


def check_min_distance(
    structure: Structure, threshold: float = MIN_INTERATOMIC_DISTANCE
) -> CheckResult:
    """Check the closest interatomic distance over periodic images.

    ``Structure.distance_matrix`` already folds periodic images (measured: atoms at
    fractional z 0.0 and 0.9 in a c = 3.0 A cell give 0.3 A, not 2.7 A), so the
    minimum of the off-diagonal entries *is* the minimum-image distance. The
    diagonal is zero (each atom to itself) and is masked before taking the
    minimum (§G.1).

    The comparison is ``<=``: "closer than 0.5 A fails" is read as "0.5 A and
    below fails", which is the safe direction for a collision check.

    Args:
        structure: The structure to measure.
        threshold: Closest allowed distance in angstrom. The default 0.5 A is the
            physics constant; see :data:`MIN_INTERATOMIC_DISTANCE`.

    Returns:
        A :class:`CheckResult` named ``"minimum_interatomic_distance"``. Its detail
        names the closest pair by index **and element symbol** -- an index pair
        alone does not tell a chemist which atoms collided. A structure with fewer
        than two sites has nothing to compare and passes.

    Example:
        ``check_min_distance(placed.structure)`` reports the C-O bond at 1.1503 A.
    """
    name = "minimum_interatomic_distance"
    n_sites = len(structure)
    if n_sites < 2:
        return CheckResult(
            name, True, f"only {n_sites} site(s): no pair to compare"
        )

    distances = np.array(structure.distance_matrix, dtype=float)
    np.fill_diagonal(distances, np.inf)
    first, second = np.unravel_index(np.argmin(distances), distances.shape)
    distance = float(distances[first, second])
    symbol_first = structure[int(first)].specie.symbol
    symbol_second = structure[int(second)].specie.symbol

    if distance <= threshold:
        return CheckResult(
            name,
            False,
            f"sites {first} ({symbol_first}) and {second} ({symbol_second}) are "
            f"{distance:.4f} A apart (<= {threshold} A)",
        )
    return CheckResult(
        name,
        True,
        f"closest pair {first} ({symbol_first})-{second} ({symbol_second}) at "
        f"{distance:.4f} A (> {threshold} A)",
    )


def check_lattice_angles(
    structure: Structure,
    min_angle: float = ANGLE_MIN,
    max_angle: float = ANGLE_MAX,
    min_volume: float = MIN_CELL_VOLUME,
) -> CheckResult:
    """Check that the lattice holds a real cell: plausible angles *and* real volume.

    Two conditions, and both are needed (§G.3):

    1. every angle lies in ``[min_angle, max_angle]`` -- 5 to 175 degrees by
       default. ``nan`` fails this, which is what catches a fully collapsed cell.
    2. the volume is finite and greater than ``min_volume``. A badly skewed cell
       keeps every angle inside the range and still has no volume, because
       ``gamma = 177`` degrees narrows the cell towards a plane.

    ``Structure.is_valid()`` is **not** used and must not be: it is a 0.5 A
    distance check (``min(distance_matrix) > 0.5``), so it reports ``True`` for a
    cell of zero volume -- measured on ``from_parameters(3, 3, 3, 89, 88, 177)``.
    ``Lattice`` has no ``is_valid`` to fall back on either.

    Angles of 60 and 120 degrees are ordinary hexagonal surface cells and pass;
    the rule is a validity range, not a preference for orthogonality.

    pymatgen divides by zero to measure a collapsed cell and emits a
    ``RuntimeWarning``; it is suppressed here so the check is quiet on the input it
    exists to reject.

    Args:
        structure: The structure whose lattice is checked.
        min_angle: Smallest valid lattice angle in degrees.
        max_angle: Largest valid lattice angle in degrees.
        min_volume: Smallest valid cell volume in cubic angstrom.

    Returns:
        A :class:`CheckResult` named ``"lattice_angles"``. Its detail states the
        measured angles (rounded to 3 decimals) and the volume to 4 decimals, so a
        caller can see how far the cell is from valid.

    Example:
        ``check_lattice_angles(slab)`` reports ``volume=160.3449 A^3`` for Pt(111).
    """
    name = "lattice_angles"
    # A degenerate cell makes pymatgen's own angle/volume properties divide by
    # zero; the check must reject that input, not warn about it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        angles = tuple(float(angle) for angle in structure.lattice.angles)
        volume = float(structure.lattice.volume)

    rounded = tuple(round(angle, 3) for angle in angles)
    out_of_range = [
        angle for angle in angles if not (min_angle <= angle <= max_angle)
    ]
    has_volume = math.isfinite(volume) and volume > min_volume

    if out_of_range or not has_volume:
        return CheckResult(
            name,
            False,
            f"invalid cell: angles={rounded}, valid range "
            f"[{min_angle}, {max_angle}] deg, volume={volume:.6g} A^3",
        )
    return CheckResult(
        name, True, f"angles={rounded}, volume={volume:.4f} A^3"
    )
