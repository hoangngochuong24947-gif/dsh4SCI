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

## What it is for

- Load a **Structure** from a stored file (a *Structure reference*).
- Build **Slabs** for a **Miller index**, distinguishing **Terminations** by a
  reproducible **Termination fingerprint**, with an exact **Vacuum**.
- Enumerate **Adsorption sites** and place an **Adsorbate** at one.
- Check a finished structure before it is used downstream.

Vocabulary is defined in [`CONTEXT.md`](CONTEXT.md); the founding decisions are
in [`docs/adr/`](docs/adr/).

## Install

```bash
pip install -e .
```

Requires Python 3.12+.
