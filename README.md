# dsh4SCI

Materials-science domain logic as a Python library: the rules a computational
materials scientist applies when building and judging atomistic structures —
surfaces, slabs, and adsorbates.

It is a library, not an agent framework, a service, or a storage layer. A
consumer supplies orchestration, tool registration, and persistence, and calls
this package for the physics. See [`CONTEXT.md`](CONTEXT.md) for the vocabulary
and [`docs/adr/0001-domain-logic-library-not-agent-plugin.md`](docs/adr/0001-domain-logic-library-not-agent-plugin.md)
for that boundary and why it was drawn there.

`pymatgen` supplies the crystallography and `ASE` the molecular geometry; this
library supplies the rules on top (see
[ADR-0002](docs/adr/0002-pure-python-reusing-pymatgen-and-ase.md)).

## Install

```bash
pip install -e .
```

Requires Python 3.12+. Add `.[dev]` for the test runner. Dependencies are pinned exactly — `pymatgen`,
`pymatgen-core`, `ase`, `spglib`, and a `numpy` range — because the rules are
pinned by tests that encode measured numbers, not by comments.

## Use

```python
import dsh4sci
```

See [`SKILL.md`](SKILL.md) for when this library applies and what it covers.

## Tests

```bash
pytest -q
```
