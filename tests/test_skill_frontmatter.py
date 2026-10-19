"""Acceptance tests for the agent-facing entry point (ticket #1).

``SKILL.md`` front matter is how an agent decides whether this library applies,
so the ``description`` must state the *situation* — building or judging atomistic
structures — rather than summarising the implementation. These tests parse the
front matter the same way an agent would, with a YAML loader, and assert on the
ticket's own wording instead of on whatever the file happens to say.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL = REPO_ROOT / "SKILL.md"

# The situation from the ticket, verbatim: an agent reading only this line must
# recognise that the library applies to structures, surfaces, slabs, adsorbates.
SITUATION_TERMS = ("building or judging", "atomistic structures", "surfaces", "slabs", "adsorbates")

# Internals must not appear in the description: it is a situation statement, not
# an implementation summary. Word boundaries keep "phase"/"database" from tripping
# the "ase" pattern.
IMPLEMENTATION_TERMS = re.compile(
    r"\b(pymatgen|ase|spglib|numpy|api|module|function|signature|pytest|wrapper)\b",
    re.IGNORECASE,
)


def _front_matter() -> dict:
    """Parse the leading ``---`` block of SKILL.md with a real YAML loader."""
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must open with a YAML front matter block"

    end = text.find("\n---\n", len("---\n"))
    assert end != -1, "SKILL.md front matter block is not closed by a '---' line"

    loaded = yaml.safe_load(text[len("---\n") : end])
    assert isinstance(loaded, dict), "SKILL.md front matter must parse as a YAML mapping"
    return loaded


def test_skill_front_matter_is_valid_yaml() -> None:
    assert _front_matter(), "SKILL.md front matter must not be empty"


def test_skill_front_matter_declares_name_and_description() -> None:
    front_matter = _front_matter()

    for key in ("name", "description"):
        assert key in front_matter, f"SKILL.md front matter is missing {key!r}"
        assert isinstance(front_matter[key], str), f"{key} must be a string"
        assert front_matter[key].strip(), f"{key} must not be empty"


def test_skill_name_is_a_stable_slug() -> None:
    name = _front_matter()["name"]

    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name), f"name {name!r} is not a lowercase slug"


def test_description_names_the_situation_not_the_implementation() -> None:
    description = _front_matter()["description"].lower()

    missing = [term for term in SITUATION_TERMS if term not in description]
    assert not missing, f"description does not name the situation; missing {missing}"


def test_description_does_not_summarise_internals() -> None:
    description = _front_matter()["description"]

    leaked = IMPLEMENTATION_TERMS.findall(description)
    assert not leaked, f"description summarises internals instead of the situation: {leaked}"
