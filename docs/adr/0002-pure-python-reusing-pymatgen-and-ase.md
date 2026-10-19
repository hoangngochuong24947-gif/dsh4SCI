# Pure Python, reusing pymatgen and ASE

Status: `accepted`

The library is pure Python and calls `pymatgen` for crystallography and `ASE` for molecular geometry, rather than reimplementing lattice, Miller-index, layer-detection, or slab-cutting maths.

Two forces decided this. First, these two libraries are the domain's de facto standard (pymatgen ~2000 stars, ASE ~517 on its canonical GitLab repository), and the hard part of this domain is not writing a lattice class — it is getting layer detection, termination ordering, and periodic-image handling right. Second, a TypeScript layer over Python would create a cross-process boundary, which is the single largest source of incidental complexity in comparable designs.

## Considered options

- **Reimplement the geometry engine from scratch.** Rejected: it buys control at the price of re-deriving rules that took the upstream projects years to stabilise, and it makes the physics harder to review because there is no authoritative reference to diff against.
- **TypeScript contracts with a Python worker per call.** Rejected: it exists only to serve a process boundary this library does not need. Worse, it makes the most common operation — pass a structure in, get a structure out — the most expensive one.

## Consequences

- `pymatgen` and `ASE` are hard dependencies. Their version pins matter and must be recorded precisely.
- The physics rules adapted from upstream (slab termination choice, adsorbate anchoring, strain accommodation) are protected by **tests that encode the rule**, not by comments. A rule that is not tested is a rule that will drift silently on the next dependency upgrade.
- Consumers must already be in the Python ecosystem. This is acceptable: every candidate consumer here is.
