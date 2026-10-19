---
name: dsh4sci
description: Use when building or judging atomistic structures — surfaces, slabs, and adsorbates — including loading a structure, cutting slabs for a Miller index, choosing a termination, setting vacuum, enumerating adsorption sites, placing an adsorbate, and checking the result before it goes into a calculation.
---

# dsh4SCI

Domain logic for physics, chemistry, and materials science: the rules a
computational materials scientist applies when building and judging atomistic
structures — surfaces, slabs, and adsorbates.

It is a **plain Python library**, imported as `dsh4sci`. It is not an agent
framework, a service, or a storage layer; a consumer supplies those and calls
this library for the physics.

```python
import dsh4sci
```

## Call this library instead of writing pymatgen or ASE inline

**If the task is "cut a Pt(111) slab and put CO on it", call these functions.
Do not author the surface-generation Python yourself, and do not reproduce the
pymatgen or ASE API from memory or from a signature listing.** A script written
that way is *nondeterministic*: the same request produces a different script on a
different day, and the differences are not cosmetic — they change the physics.
This library exists so that one request has exactly one implementation.

Three reasons, in order of how much they matter:

1. **It is deterministic.** The same input gives the same Structure, the same
   Vacuum, the same fingerprint, the same site count, run to run. Every number in
   the docstrings was measured on the pinned stack and is *asserted by a test*, so
   a dependency upgrade that changes the physics fails the suite instead of
   silently changing your slab.
2. **The rules are pinned, not remembered.** The Vacuum is exact, the
   Termination has a content-based fingerprint, the bottom-layer rule is a total
   order, and the self-check knows what "the Adsorbate is inside the Slab" looks
   like. Upstream has no equivalent for most of that.
3. **The traps below are already handled.** Each one is a silent wrong answer, not
   an exception — the kind of bug that survives to a production calculation.

### What goes wrong when you hand-roll it

| hand-rolled pymatgen / ASE | what this library does instead |
|---|---|
| `SlabGenerator` adds `bulk_wyckoff` / `bulk_equivalent` site properties to the bulk you pass it | copies the bulk first; the caller's Structure is untouched |
| `min_vacuum_size` is only a **lower bound**: a 15 Å request came back at 31.385 Å raw, 18.120 Å after orthogonalisation | imposes the exact Vacuum afterwards, split equally above and below |
| centring that Vacuum on `z.min()` mis-centres a stack that straddles the periodic boundary — measured, one atom ends up 5.23 Å *outside* the cell | anchors after the largest *periodic* gap, then unwraps |
| the default `symm_reduce=0.01` collapses symmetry-equivalent sites, so Pt(111) 2x2 / 3x3 / 4x4 all return 1 ontop, 1 bridge, 2 hollow instead of 4/12/8, 9/27/18, 16/48/32 | enumerates every site (`symm_reduce=-1`), so counts scale with area |
| `distance` (default 2.0 Å) lifts each site along the normal, while `height` (default 0.9 Å) decides which atoms count as surface; confusing them moves the site sideways *through* the stack | fixes `distance=0`, `height=0.9`: a site lies on the top layer and the caller owns the offset |
| `ase.build.molecule("CO")` is ordered `['O', 'C']` — index 0 is oxygen, so anchoring by index stands the molecule on its head | anchors by element **symbol**, found in the molecule |
| re-anchoring first and flipping a z sign afterwards silently inverts the molecule (measured `O_z - C_z = -1.1503 Å`) | rotates the anchor-relative frame *before* translating |
| `Structure.make_supercell(..., in_place=True)` is the default **and returns `self`**: `s33 = slab.make_supercell([3, 3, 1])` leaves `slab` and `s33` the same 45-site object | not a library call — but this is why you should build the slab here and only supercell afterwards |
| the `role` tag survives `copy()` and `make_supercell()` but is **lost through POSCAR/CIF** | `place_adsorbate` returns the Slab atom count, which survives file I/O |
| `str(site.species)` is a `Composition`, so an oxygen site renders `'O1'` and `== 'O'` matches nothing | predicates use `site.specie.symbol` |
| ASE reads a POSCAR's flags back as `FixAtoms` and **drops the per-axis array**, so `[T, T, F]` is indistinguishable from `[F, F, F]` | pymatgen carries the flags; ASE is only good for the count |
| `Structure.is_valid()` is a 0.5 Å distance check and returns `True` for a zero-volume cell | judges angles **and** volume explicitly |
| a naive Cartesian distance misses periodic images: fractional z 0.001 and 0.999 in c = 3.9231 Å are 0.0078 Å apart, not 3.9153 Å | measures through `distance_matrix`, which already folds periodic images |

### When hand-rolling *is* the right call

This library stops at the structure. It does not write DFT inputs
(`INCAR`, `KPOINTS`), run a code, parse its output, relax, or compute band
structures — do that yourself, with the structure this library produced.

Two other cases where you should not use it:

- **A structure the workflow below cannot express.** A slab from a
  non-conventional cell, a heterostructure, a nanorod: build it in pymatgen, then
  call `check_structure` on the result. Do not re-derive the Vacuum, the layer
  grouping, or the Termination fingerprint — they live here.
- **An operation applied to something that is not a Slab.** `fix_bottom_atoms`
  assumes atoms stacked along z; `find_adsorption_sites` rejects anything without
  a Vacuum for exactly that reason.

## The workflow

Load → generate Slabs → (fix the bottom) → list Adsorption sites → place the
Adsorbate → validate. Each step below shows the signature the code actually has
and the call to make.

### 1. Load a Structure reference

```signatures
load_structure(structure_path: str | os.PathLike[str]) -> Structure
```

A **Structure reference** is a path and nothing else. Accepted suffixes are
`.poscar`, `.vasp`, `.cif`, plus the bare names `POSCAR` and `CONTCAR` that VASP
writes; anything else is refused with the supported set in the message. This is
the library's entire file-I/O surface — every other function takes a `Structure`,
so nothing downstream can be surprised by a filename.

```python
from dsh4sci import load_structure

bulk = load_structure("tests/fixtures/Pt.poscar")   # any POSCAR / CIF path
```

### 2. Generate Slabs, then fix the bottom

```signatures
generate_slabs(bulk: Structure, miller_index: Sequence[int], min_slab_size: float = 10.0, vacuum: float = 15.0) -> list[Slab]
termination_fingerprint(slab: Slab) -> TerminationFingerprint
fix_bottom_atoms(slab: SlabT, fraction: float) -> SlabT
```

`generate_slabs` returns **one Slab per Termination** — all of them, not a chosen
one. A **Termination** is a specific cut within one Miller-index family,
distinguished by which atomic layer the exposed face ends on; `pymatgen` has no
`termination` attribute and does not contract the order it enumerates them in, so
identify a Slab by `termination_fingerprint`, never by index.

The fingerprint is a frozen record of the top and bottom layer compositions plus
the in-plane offset, quantised to 6 decimals. `str()` gives a compact form and
`sha()` a 16-hex digest; both are stable across processes.

`SlabT` is a public type variable bound to `Structure`, exported from the package
root: give `fix_bottom_atoms` a Slab and you get a Slab back. It fixes the bottom
`floor(n_atoms * fraction)` atoms by writing pymatgen's `selective_dynamics` site
property — the only way `Poscar` will emit the flag block — and it orders atoms by
`(z, x, y, index)` so the same physical slab always fixes the same atoms, whatever
order the sites happen to arrive in. `fraction=0` fixes nothing; the Vacuum is
untouched.

```python
from dsh4sci import fix_bottom_atoms, generate_slabs, termination_fingerprint

slabs = generate_slabs(bulk, miller_index=(1, 1, 1), min_slab_size=10.0, vacuum=15.0)
slab = slabs[0]
print(termination_fingerprint(slab))      # (1, 1, 1):Pt1/Pt1:shift=500000e-6
slab = fix_bottom_atoms(slab, fraction=0.5)
```

### 3. List Adsorption sites

```signatures
find_adsorption_sites(structure: Structure) -> list[AdsorptionSite]
```

One `AdsorptionSite` per site, grouped by type in the order `ontop`, `bridge`,
`hollow` — those three strings are the vocabulary, with no upstream alias and
never the `all` key upstream adds as a concatenation of the other three. The order
*within* a type is not a contract.

Each site carries `site_type` and a Cartesian `position` on the top layer.
Supercelling is the caller's job and must happen **after** the Slab is orthogonal
(`generate_slabs` output already is); copy first, because `make_supercell`
mutates in place.

A structure that is not Slab-like — no Vacuum of at least 5.0 Å, or fewer than two
atomic z-levels — is refused, not silently repaired. That refusal names the
measured Vacuum, so you can see how far off the input is.

```python
from dsh4sci import find_adsorption_sites

sites = find_adsorption_sites(slab)
ontop = [site for site in sites if site.site_type == "ontop"]
```

### 4. Place an Adsorbate

```signatures
place_adsorbate(slab: Structure, site: AdsorptionSite, *, molecule: str, anchor: str, height: float = 1.85) -> PlacedAdsorbate
```

`molecule` is an ASE `g2` name (`"CO"`, `"OH"`, `"H2O"`). `anchor` is the
**element symbol** of the atom that binds — for CO that is `"C"`, and the library
finds it in the molecule rather than assuming a position in the list. `height` is
the anchor-to-surface distance in Å and it is keyword-only: `place_adsorbate(slab,
site, "CO", "C")` is a `TypeError` on purpose, because a positional fourth
argument is the kind of thing that silently becomes a different parameter.

The result is a `PlacedAdsorbate` with three fields, and the last two are not
redundant:

- `structure` — every Slab site first, then one site per Adsorbate atom, tagged
  with the `role` site property (`'slab'` / `'adsorbate'`). The tag makes the two
  separable **in memory**; it is lost through POSCAR/CIF.
- `n_slab_atoms` — how many leading sites belong to the Slab. After file I/O this
  is the source of truth.
- `anchor_index` — the binding atom's index, which `check_structure` needs.

Place one Adsorbate per call, on a Slab that does not already carry one.

```python
from dsh4sci import place_adsorbate

placed = place_adsorbate(slab, sites[0], molecule="CO", anchor="C", height=1.85)
```

### 5. Validate

```signatures
check_structure(structure: Structure, *, anchor_index: int | None = None) -> list[CheckResult]
all_passed(results: Iterable[CheckResult]) -> bool
```

Three rules run, and each reports a `CheckResult` — `name`, `passed`, and a
`detail` stating the measured quantity — never a bare boolean:

1. `minimum_interatomic_distance` — the closest pair over periodic images must
   exceed 0.5 Å. A correct CO/Pt(111) has that pair at the 1.15034 Å C–O bond.
2. `adsorbate_anchor_above_surface` — the anchor must sit at or above the Slab's
   top layer. This is the failure the module exists for: an Adsorbate buried
   inside the Slab satisfies every schema rule and is useless.
3. `lattice_angles` — every lattice angle in `[5, 175]` degrees **and** a real,
   non-zero volume. 60 and 120 degrees are normal hexagonal surface cells.

`anchor_index` is required whenever the structure carries Adsorbate-tagged sites:
skipping the anchor check and reporting "everything passed" is exactly the false
negative to prevent. `all_passed` is the boolean form when that is all you need.

```python
from dsh4sci import all_passed, check_structure

results = check_structure(placed.structure, anchor_index=placed.anchor_index)
all_passed(results)     # True
```

## The whole workflow, as one runnable script

The five steps again, in order and with nothing elided. On the checked-in Pt
fixture this gives one Slab, Pt5, five z-levels, an exact 15.000000 Å Vacuum, and
a CO bound 1.850000 Å above the top layer with the carbon down — all three checks
passing. Copy it and change the inputs.

```python workflow
from dsh4sci import (
    all_passed,
    check_structure,
    find_adsorption_sites,
    fix_bottom_atoms,
    generate_slabs,
    load_structure,
    place_adsorbate,
    termination_fingerprint,
)

# 1. load: a Structure reference is a path.
bulk = load_structure("tests/fixtures/Pt.poscar")

# 2. slabs: one Slab per Termination, c orthogonal, the Vacuum already exact.
slabs = generate_slabs(bulk, miller_index=(1, 1, 1), min_slab_size=10.0, vacuum=15.0)
slab = slabs[0]
fingerprint = termination_fingerprint(slab)      # identify a Termination by content

# freeze the bottom half of the atoms (2 of 5 here) for the relaxation
slab = fix_bottom_atoms(slab, fraction=0.5)

# 3. sites: every ontop / bridge / hollow position on the top layer.
sites = find_adsorption_sites(slab)

# 4. adsorbate: bind carbon, 1.85 A above the top layer.
placed = place_adsorbate(slab, sites[0], molecule="CO", anchor="C", height=1.85)

# 5. validate: three verdicts, and the boolean form of them.
results = check_structure(placed.structure, anchor_index=placed.anchor_index)
for result in results:
    print(result)
assert all_passed(results)
```

## Reading a refusal

Every refusal is one line and follows the same shape, so it can be read without a
traceback:

```
<parameter>=<rejected value> is invalid: <why>. Valid: <range or set>. Example: <a call you can paste>
```

```
fraction=-0.1 is invalid: must be in [0.0, 1.0]. Valid: a float in [0.0, 1.0]; floor(n_atoms * fraction) atoms are fixed. Example: fix_bottom_atoms(slab, fraction=0.5)
```

Four things to take from any message: which **parameter** was wrong, what value
was **rejected**, what the **valid range** is, and a **call that would work**. A
missing file is a `FileNotFoundError` (the builtin, so existing `except` clauses
keep working); every other input refusal is a `ValueError` subclass from
`dsh4sci.errors`, all sharing the root `Dsh4sciError`. No `ZeroDivisionError`, no
`KeyError`, no upstream `ValueError("Unrecognized extension ...")` reaches you.

## How the code in this file is checked

The signatures above are **not prose**. They are parsed by a test and compared
with `inspect.signature` of the real function, so a signature that drifts fails
the suite. The convention, which an editor must preserve:

- A **signature block** is a fenced code block whose info string is exactly
  `signatures`. Inside it, each non-blank line is one signature: the text that
  follows `def ` in Python source — parameter names, annotations, the keyword-only
  `*` marker, defaults and the return annotation — with no `def`, no body and no
  trailing colon.
- **One signature per line.** Never wrap a long one: a wrapped line is not a
  signature, so the parse fails and says which line it could not read.
- The annotation text must be the module's own text, because the comparison is
  exact. Most modules use `from __future__ import annotations`, so these are the
  strings the code carries.
- Only names exported from the `dsh4sci` package root may be documented: this file
  is the package's public surface, not a tour of its modules.

`tests/test_skill_signatures.py` enforces all of that, including that every
function the workflow needs is documented exactly once and that nothing stale is.
The runnable examples are checked the same way: a fenced block whose info string
is `python workflow` is executed verbatim by `tests/test_end_to_end.py`, against
the checked-in fixtures. Change a number in the workflow, or change the code it
calls, and one of those two tests fails.

## Vocabulary and decisions

Terms are defined in [`CONTEXT.md`](CONTEXT.md) — Structure, Structure reference,
Slab, Surface, Termination, Termination fingerprint, Vacuum, Adsorption site,
Adsorbate, Coverage. The founding decisions are in [`docs/adr/`](docs/adr/):
pure Python, reuse pymatgen and ASE, and **rules pinned by tests rather than
comments**.

## Install

```bash
pip install -e .
```

Requires Python 3.12+.
