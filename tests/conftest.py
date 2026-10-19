"""Shared pytest fixtures for the dsh4SCI test suite.

Only *cross-ticket* fixtures live here: the checked-in Structure references and
the bulk structures that later tickets cut Slabs from. Ticket-local fixtures
belong in that ticket's own test module (see the file-ownership table in
PLAYBOOK.md), so this file stays small and dependency-free.

The loaded structures are deliberately **function-scoped**: a ``Structure`` is
mutable, and downstream pymatgen code mutates its input (``SlabGenerator`` adds
``bulk_wyckoff``/``bulk_equivalent`` site properties to the Structure it is
handed, CONVENTIONS.md §C.1). A session-scoped bulk would leak that mutation
between tests.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pymatgen.core import Structure

from dsh4sci.loading import load_structure

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path of the directory holding the checked-in fixtures."""
    return FIXTURES_DIR


@pytest.fixture
def fixture_path() -> Callable[[str], Path]:
    """Map a fixture filename to its absolute path: ``fixture_path("Pt.poscar")``."""

    def resolve(name: str) -> Path:
        return FIXTURES_DIR / name

    return resolve


@pytest.fixture
def pt_bulk() -> Structure:
    """Conventional cubic Pt (``Pt4``, a = 3.9231 A) -- the bulk a Slab is cut from."""
    return load_structure(FIXTURES_DIR / "Pt.poscar")
