"""Every signature ``SKILL.md`` shows is the signature the code has (ticket #8).

An agent drives this library by copying what ``SKILL.md`` shows it. A signature in
that file that has drifted from the code does not merely mislead: it produces a
call that raises, or -- worse -- a call that succeeds with a different meaning. So
the documented text is treated as an executable claim and checked against
``inspect.signature`` of the real function.

**The parsing convention** (documented in ``SKILL.md`` itself so a future editor
cannot silently break it):

* a *signature block* is a fenced code block whose info string is exactly
  ``signatures``;
* inside it, each non-blank line is one signature, written as the text that
  follows ``def `` in Python source -- parameters, annotations, keyword-only ``*``
  markers, defaults and the return annotation -- with no ``def`` keyword, no body
  and no trailing colon;
* one signature per line, never wrapped.

The check compiles each line into a throwaway function with ``from __future__
import annotations`` forced on, exactly as every module in this package compiles,
and compares the resulting ``inspect.Signature`` with the real one. ``Signature``
equality is element-wise over parameter name, *kind* (which is how a keyword-only
parameter is distinguishable from a positional one), default and annotation text,
plus the return annotation -- i.e. everything ``str(inspect.signature(...))``
renders. The documented annotation text must therefore be the module's own
annotation text, not a paraphrase of it.
"""

from __future__ import annotations

import __future__
import inspect
import re
from pathlib import Path

import pytest

import dsh4sci

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL = REPO_ROOT / "SKILL.md"

# The public surface the workflow in SKILL.md is built from. Expected, not derived
# from the file: dropping a signature from the document must fail this module.
EXPECTED_FUNCTIONS = frozenset(
    {
        "all_passed",
        "check_structure",
        "find_adsorption_sites",
        "fix_bottom_atoms",
        "generate_slabs",
        "load_structure",
        "place_adsorbate",
        "termination_fingerprint",
    }
)

# A fenced block whose info string is `signatures`. DOTALL so the body may span
# lines; the closing fence must be at the start of a line.
_SIGNATURE_BLOCK = re.compile(
    r"^```signatures[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL
)

# One documented signature: `name(parameters) -> return_annotation`.
_SIGNATURE_LINE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\((?P<parameters>.*)\)"
    r"\s*->\s*(?P<returns>\S.*)$"
)


def _documented_signatures() -> dict[str, tuple[int, str]]:
    """Map documented function name -> (line number in SKILL.md, signature text)."""
    text = SKILL.read_text(encoding="utf-8")
    documented: dict[str, tuple[int, str]] = {}

    for block in _SIGNATURE_BLOCK.finditer(text):
        first_body_line = text[: block.start("body")].count("\n") + 1
        for offset, raw_line in enumerate(block.group("body").splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            line_number = first_body_line + offset
            match = _SIGNATURE_LINE.fullmatch(line)
            assert match is not None, (
                f"SKILL.md:{line_number} is inside a ```signatures block but is not "
                f"a single-line signature of the form 'name(...) -> annotation': "
                f"{line!r}"
            )
            name = match.group("name")
            assert name not in documented, (
                f"SKILL.md documents {name!r} twice (lines "
                f"{documented[name][0]} and {line_number}); the two copies can "
                f"disagree and only one is checked"
            )
            documented[name] = (line_number, line)
    return documented


def _signature_from_document(signature_text: str) -> inspect.Signature:
    """Compile one documented line into the ``inspect.Signature`` it describes.

    The compilation forces ``from __future__ import annotations``, which every
    module of this package uses, so annotations stay the *source text* on both
    sides of the comparison instead of becoming resolved type objects on one side
    and strings on the other.
    """
    source = f"def {signature_text}: ...\n"
    code = compile(
        source,
        filename="<SKILL.md>",
        mode="exec",
        flags=__future__.annotations.compiler_flag,
    )
    namespace: dict[str, object] = {}
    # The compiled text is this repository's own document; nothing external runs.
    exec(code, namespace)
    function = namespace[_SIGNATURE_LINE.fullmatch(signature_text).group("name")]
    return inspect.signature(function)


@pytest.mark.parametrize("name", sorted(EXPECTED_FUNCTIONS))
def test_documented_signature_equals_the_real_signature(name: str) -> None:
    """The documented line is the signature ``inspect`` reports for the function.

    Both sides are obtained independently -- one from the document text by the
    compiler, one from the imported module by ``inspect`` -- so this cannot pass by
    construction.
    """
    documented = _documented_signatures()
    assert name in documented, (
        f"SKILL.md does not document {name!r} in a ```signatures block; the "
        f"workflow cannot be driven without its signature"
    )
    line_number, signature_text = documented[name]

    real = getattr(dsh4sci, name, None)
    assert real is not None, (
        f"SKILL.md documents {name!r} but dsh4sci does not export it; SKILL.md may "
        f"only document the package root's public surface"
    )

    assert _signature_from_document(signature_text) == inspect.signature(real), (
        f"SKILL.md:{line_number} shows {signature_text!r} but the real signature is "
        f"{str(inspect.signature(real))!r}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_FUNCTIONS))
def test_documented_signature_renders_identically_to_inspect(name: str) -> None:
    """The rendered forms agree, so what an agent reads is what Python prints.

    ``str(inspect.Signature)`` is what a caller sees in a traceback or in
    ``help()``; an agent comparing the document with either must get a character-
    for-character match apart from the quoting of string annotations, which
    ``inspect`` adds on both sides.
    """
    documented = _documented_signatures()
    assert name in documented
    signature_text = documented[name][1]
    real = getattr(dsh4sci, name)

    assert str(_signature_from_document(signature_text)) == str(
        inspect.signature(real)
    )


def test_every_documented_signature_is_a_required_public_function() -> None:
    """No stale entry: the documented set is exactly the workflow's surface."""
    documented = set(_documented_signatures())

    assert documented == EXPECTED_FUNCTIONS, (
        f"undocumented: {sorted(EXPECTED_FUNCTIONS - documented)}; "
        f"stale or unexpected: {sorted(documented - EXPECTED_FUNCTIONS)}"
    )


def test_the_parsing_convention_is_documented_in_skill_md() -> None:
    """A future editor must be able to learn the convention from the file itself.

    Without this, the first well-meaning reformat of the signature blocks would
    turn the check above into a silent no-op.
    """
    text = SKILL.read_text(encoding="utf-8")

    assert "signatures" in text, "SKILL.md never names the ```signatures block"
    assert "one signature per line" in text.lower(), (
        "SKILL.md does not state the one-signature-per-line rule"
    )
    assert "tests/test_skill_signatures.py" in text, (
        "SKILL.md does not say which test enforces the signature blocks"
    )


def test_keyword_only_markers_and_defaults_are_shown() -> None:
    """The two things an agent most easily gets wrong, pinned independently.

    These expectations are the conventions' numbers (1.85 A binding height, a
    10 A / 15 A Slab request), not values read back from either signature.
    """
    documented = _documented_signatures()

    placement = _signature_from_document(documented["place_adsorbate"][1])
    parameters = placement.parameters
    assert parameters["molecule"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["anchor"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["height"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["height"].default == 1.85
    assert parameters["height"].annotation == "float"
    assert parameters["molecule"].default is inspect.Parameter.empty
    assert parameters["anchor"].default is inspect.Parameter.empty

    validation = _signature_from_document(documented["check_structure"][1])
    anchor_index = validation.parameters["anchor_index"]
    assert anchor_index.kind is inspect.Parameter.KEYWORD_ONLY
    assert anchor_index.default is None

    slabs = _signature_from_document(documented["generate_slabs"][1])
    assert slabs.parameters["min_slab_size"].default == 10.0
    assert slabs.parameters["vacuum"].default == 15.0
    assert slabs.return_annotation == "list[Slab]"
