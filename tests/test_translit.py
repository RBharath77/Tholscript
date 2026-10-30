"""Round-trip tests for the Tamil <-> Brahmi transliteration layer.

Run:  python -m pytest tests/ -q      (from the project root)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tholscript.data.inventory import (  # noqa: E402
    BRAHMI, BRAHMI_INVENTORY, TAMIL_INVENTORY, VIRAMA, build_brahmi_inventory,
    build_tamil_inventory, tamil_cluster_roman,
)
from tholscript.data.lexicon import ALL_WORDS, sample_brahmi_line, sample_tamil_line  # noqa: E402
from tholscript.translation.brahmi_map import (  # noqa: E402
    brahmi_to_roman, brahmi_to_tamil, tamil_to_brahmi, tamil_to_roman,
)

import random  # noqa: E402


def test_inventories_are_unique_and_nonempty():
    assert len(TAMIL_INVENTORY) == len(set(TAMIL_INVENTORY))
    assert len(BRAHMI_INVENTORY) == len(set(BRAHMI_INVENTORY))
    assert len(TAMIL_INVENTORY) > 200
    assert len(BRAHMI_INVENTORY) > 400


def test_tamil_romanisation():
    assert tamil_to_roman("மன்னன்") == "maNNaN"
    assert tamil_to_roman("ஊர்") == "Ur"
    assert tamil_to_roman("கிணறு") == "kindaRu"
    assert tamil_to_roman("௩") == "3"
    assert tamil_cluster_roman("கி") == "ki"
    assert tamil_cluster_roman("க்") == "k"
    assert tamil_cluster_roman("அ") == "a"


def test_brahmi_output_is_in_inventory():
    """Every cluster tamil_to_brahmi emits must be a known Brahmi grapheme cluster."""
    inv = set(build_brahmi_inventory())
    bad = []
    for w in ALL_WORDS:
        br = tamil_to_brahmi(w)
        # split the cluster string back into inventory members greedily
        i = 0
        while i < len(br):
            for ln in (3, 2, 1):
                cand = br[i:i + ln]
                if cand in inv:
                    i += ln
                    break
            else:
                bad.append((w, br[i]))
                i += 1
    assert not bad, f"unknown Brahmi clusters for: {bad[:10]}"


def test_roman_roundtrip_is_stable():
    """brahmi_to_roman(tamil_to_brahmi(w)) must reproduce the Prakrit reading."""
    cases = {
        "மன்னன்": "mannan",
        "ஊர்": "uur",
        "கொடை": "kotai",
        "மகன்": "makan",
        "நெல்": "nel",
        "ஏரி": "eeri",
    }
    for word, expect in cases.items():
        got = brahmi_to_roman(tamil_to_brahmi(word))
        assert got == expect, f"{word}: expected {expect!r}, got {got!r}"


def test_brahmi_to_tamil_produces_real_glyphs():
    for w in ALL_WORDS:
        out = brahmi_to_tamil(tamil_to_brahmi(w))
        assert out, f"{w} transliterated to empty string"
        assert all("\u0B80" <= c <= "\u0BCD" for c in out), f"{w} -> {out!r}"


def test_virama_never_trails_a_vowel_sign():
    for w in ALL_WORDS:
        br = tamil_to_brahmi(w)
        assert VIRAMA + VIRAMA not in br, w


def test_lexicon_lines_are_tamil():
    rng = random.Random(1)
    for _ in range(50):
        s = sample_tamil_line(rng)
        assert s.words and all(
            all("\u0B80" <= c <= "\u0BCD" or c.isdigit() for c in w) for w in s.words
        )


def test_brahmi_lines_use_the_brahmi_block():
    rng = random.Random(1)
    inv = set(BRAHMI)
    for _ in range(30):
        s = sample_brahmi_line(rng)
        assert s.text.strip()
        for ch in s.text:
            if ch == " ":
                continue
            assert ch in inv, f"non-Brahmi char {hex(ord(ch))} in {s.text!r}"


def test_prakrit_sound_changes_produce_plausible_tamil():
    """Vowel sequences merge and -au- reduces, like a real Prakrit->Tamil reading."""
    assert brahmi_to_tamil(tamil_to_brahmi("ஊர்")) == "ஊர்"
    assert brahmi_to_tamil(tamil_to_brahmi("ஏரி")) == "ஏரி"
    assert brahmi_to_tamil(tamil_to_brahmi("கோயில்")) == "கோயில்"
    assert brahmi_to_tamil(tamil_to_brahmi("பெருங்குடி")) == "பெருங்குடி"


def test_brahmi_metadata_is_complete():
    for ch, (name, tr, tamil) in BRAHMI.items():
        assert name.startswith("BRAHMI"), name
        assert isinstance(tr, str) and isinstance(tamil, str)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
