"""Transliteration between Tamil orthography, Brahmi orthography and Latin.

Three conversions live here:

``tamil_to_brahmi``   modern Tamil word  ->  Brahmi clusters.  Tamil-Brahmi
                    cave inscriptions write the local language with the Brahmi
                    alphabet, so the Brahmi training corpus is produced by
                    running the Tamil lexicon through this map.
``brahmi_to_roman``   Brahmi clusters    ->  Latin transliteration.
``brahmi_to_tamil``   Brahmi clusters    ->  modern Tamil spelling, applying the
                    Prakrit -> Tamil sound changes an epigraphist would use.

Design note: everything is tokenised *before* it is converted.  The earlier
version of this module re-scanned the romanised string mid-loop, which
double-counted vowels (``மன்னன்`` -> ``mnanana`` instead of ``mannan``).  The
token list is now built once and iterated once, and ``tests/test_translit.py``
pins the round-trips.

The Prakrit -> Tamil rules are *conventional*, not authoritative; they are all
listed in ``PRAKRIT_TAMIL_RULES`` so they can be audited or swapped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from tholscript.data.inventory import BRAHMI, BRAHMI_CONSONANTS, TAMIL_DIGITS, VIRAMA

# --------------------------------------------------------------------------- #
# Tamil -> Latin
# --------------------------------------------------------------------------- #
_TAMIL_CONS_ROMAN = {
    "க": "k", "ங": "ng", "ச": "c", "ஞ": "ny", "ட": "t", "ண": "nd", "த": "th",
    "ந": "n", "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v",
    "ழ": "zh", "ள": "L", "ற": "R", "ன": "N",
}
_TAMIL_VOWEL_ROMAN = {
    "அ": "a", "ஆ": "A", "இ": "i", "ஈ": "I", "உ": "u", "ஊ": "U", "எ": "e",
    "ஏ": "E", "ஐ": "Y", "ஒ": "o", "ஓ": "O", "ஔ": "W",
}
_MAA_TO_VOWEL = {
    "": "a", "\u0BBE": "A", "\u0BBF": "i", "\u0BC0": "I", "\u0BC1": "u",
    "\u0BC2": "U", "\u0BC6": "e", "\u0BC7": "E", "\u0BC8": "Y", "\u0BCA": "o",
    "\u0BCB": "O", "\u0BCC": "W", "\u0BCD": "",
}
_DIGIT_TO_ASCII = {d: str(i) for i, d in enumerate(TAMIL_DIGITS)}


def tamil_to_tokens(word: str) -> List[Tuple[str, str]]:
    """மன்னன் -> [('C','m'), ('V','a'), ('C','N'), ('C','N'), ('V','a'), ('C','N')]

    'C' tokens are onsets, 'V' tokens are vowel nuclei, 'D' tokens are digits.
    A consonant with no following vowel mark gets the inherent short 'a';
    a consonant with a pulli gets no vowel token at all.
    """
    toks: List[Tuple[str, str]] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in _DIGIT_TO_ASCII:
            toks.append(("D", _DIGIT_TO_ASCII[ch]))
            i += 1
            continue
        if ch in _TAMIL_VOWEL_ROMAN:
            toks.append(("V", _TAMIL_VOWEL_ROMAN[ch]))
            i += 1
            continue
        if ch in _TAMIL_CONS_ROMAN:
            toks.append(("C", _TAMIL_CONS_ROMAN[ch]))
            maa = word[i + 1] if i + 1 < len(word) else ""
            if maa in _MAA_TO_VOWEL:
                v = _MAA_TO_VOWEL[maa]
                if v:
                    toks.append(("V", v))
                i += 2
            else:
                toks.append(("V", "a"))     # inherent short a
                i += 1
            continue
        i += 1                              # stray combining mark: ignore
    return toks


def tamil_to_roman(word: str) -> str:
    return "".join(v for _, v in tamil_to_tokens(word))


# --------------------------------------------------------------------------- #
# Latin tokens -> Brahmi
# --------------------------------------------------------------------------- #
_CONS_TO_BRAHMI = {
    "k": "\U00011013", "kh": "\U00011014", "g": "\U00011015", "gh": "\U00011016",
    "ng": "\U00011017", "c": "\U00011018", "ch": "\U00011019", "j": "\U0001101A",
    "jh": "\U0001101B", "ny": "\U0001101C", "t": "\U0001101D", "tth": "\U0001101E",
    "dd": "\U0001101F", "nd": "\U00011021", "th": "\U00011022", "d": "\U00011024",
    "dh": "\U00011025", "n": "\U00011026", "N": "\U00011026", "p": "\U00011027",
    "ph": "\U00011028", "b": "\U00011029", "bh": "\U0001102A", "m": "\U0001102B",
    "y": "\U0001102C", "r": "\U0001102D", "R": "\U0001102D", "l": "\U0001102E",
    "v": "\U0001102F", "zh": "\U00011032", "L": "\U00011034", "s": "\U00011032",
    "sh": "\U00011030", "h": "\U00011033",
}
_VOWEL_TO_BRAHMI_IND = {
    "a": "\U00011005", "A": "\U00011006", "i": "\U00011007", "I": "\U00011008",
    "u": "\U00011009", "U": "\U0001100A", "e": "\U0001100E", "E": "\U0001100F",
    "Y": "\U00011010", "o": "\U00011011", "O": "\U00011012",
}
_VOWEL_TO_BRAHMI_MAA = {
    "A": "\U00011036", "i": "\U00011037", "I": "\U00011038", "u": "\U00011039",
    "U": "\U0001103A", "e": "\U0001103E", "E": "\U0001103F", "Y": "\U00011040",
    "o": "\U00011041", "O": "\U00011042",
}
# Brahmi has no vowel SIGN for short a / long aa / au: those use the independent
# vowel letter (or the AA sign) instead.  Everything is derived from the
# inventory table so the two directions cannot drift apart.
_VOWEL_TO_BRAHMI_MAA["W"] = "\U00011006"     # au  -> AA sign (as in Prakrit orthography)
_BRAHMI_MAA_CHARS = {"\U00011036", "\U00011037", "\U00011038", "\U00011039",
                     "\U0001103A", "\U0001103E", "\U0001103F", "\U00011040",
                     "\U00011041", "\U00011042"}
_BRAHMI_TO_TAMIL_MAA = {info[2]: ch for ch, info in BRAHMI.items()
                        if info[2] and info[2] not in ("அ",) and len(info[2]) == 1
                        and ord(info[2]) >= 0x0BBE and ord(info[2]) <= 0x0BCC}
_BRAHMI_IND_TO_TAMIL_VOWEL = {ch: info[2] for ch, info in BRAHMI.items()
                              if len(info[2]) == 1 and "\u0B85" <= info[2] <= "\u0B94"}
_BRAHMI_IND_TO_TAMIL_VOWEL["\U0001100B"] = "ரு"
_BRAHMI_IND_TO_TAMIL_VOWEL["\U0001100C"] = "லு"
_BRAHMI_IND_TO_TAMIL_VOWEL["\U0001100D"] = "ளூ"
_DIGIT_TO_BRAHMI = {str(i): chr(0x11051 + i) for i in range(1, 10)}


def tamil_to_brahmi(word: str) -> str:
    """மன்னன் -> BRAHMI MA + NA/VIRAMA + NA + NA/VIRAMA (as a single cluster string)."""
    toks = tamil_to_tokens(word)
    out: List[str] = []
    k = 0
    while k < len(toks):
        kind, val = toks[k]
        if kind == "D":
            out.append(_DIGIT_TO_BRAHMI.get(val, ""))
            k += 1
            continue
        if kind == "V":                                  # independent vowel
            out.append(_VOWEL_TO_BRAHMI_IND.get(val, ""))
            k += 1
            continue
        # consonant: look at what follows it
        cons = _CONS_TO_BRAHMI.get(val)
        if cons is None:
            k += 1
            continue
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        if nxt is None:                                  # word-final bare onset
            out.append(cons + VIRAMA)
            k += 1
        elif nxt[0] == "V":                              # onset + vowel
            if nxt[1] == "a":
                out.append(cons)                         # inherent short a
            else:
                out.append(cons + _VOWEL_TO_BRAHMI_MAA.get(nxt[1], ""))
            k += 2
        else:                                            # onset + onset: cluster
            out.append(cons + VIRAMA)
            k += 1
    return "".join(c for c in out if c)


# --------------------------------------------------------------------------- #
# Brahmi -> Latin
# --------------------------------------------------------------------------- #
def brahmi_to_roman(text: str) -> str:
    out: List[str] = []
    for ch in text:
        if ch == " ":
            out.append(" ")
            continue
        info = BRAHMI.get(ch)
        if info is None:
            continue
        _, tr, _ = info
        if ch == VIRAMA:
            if out and out[-1].endswith("a") and len(out[-1]) > 1:
                out[-1] = out[-1][:-1]
            continue
        if ch == "\U00011044":                           # anusvara
            out.append("m")
            continue
        if ch in BRAHMI_CONSONANTS:
            out.append(tr)
        elif ch in _BRAHMI_MAA_CHARS:
            if out and out[-1].endswith("a") and len(out[-1]) > 1:
                out[-1] = out[-1][:-1] + tr
            else:
                out.append(tr)
        else:
            out.append(tr)
    return "".join(out)


# --------------------------------------------------------------------------- #
# Prakrit (roman) -> modern Tamil spelling
# --------------------------------------------------------------------------- #
# (source in Prakrit romanisation, Tamil roman token, kind)  -- longest first
PRAKRIT_TAMIL_RULES: List[Tuple[str, str, str]] = [
    # aspiration is not written in Tamil
    ("kh", "k", "C"), ("gh", "k", "C"), ("ch", "c", "C"), ("jh", "c", "C"),
    ("dh", "th", "C"), ("bh", "p", "C"), ("ph", "p", "C"), ("tth", "t", "C"),
    ("th", "th", "C"), ("sh", "c", "C"),
    # voiced stops reduce to their Tamil voiceless counterpart
    ("g", "k", "C"), ("j", "c", "C"), ("dd", "t", "C"), ("d", "th", "C"),
    ("b", "p", "C"),
    # Prakrit s -> Tamil h ; retroflex nasals/stops preserved
    ("s", "h", "C"), ("ng", "ng", "C"), ("ny", "ny", "C"), ("nd", "nd", "C"),
    ("zh", "c", "C"),
    ("k", "k", "C"), ("c", "c", "C"), ("t", "t", "C"), ("n", "n", "C"),
    ("p", "p", "C"), ("m", "m", "C"), ("y", "y", "C"), ("r", "r", "C"),
    ("l", "l", "C"), ("v", "v", "C"), ("h", "h", "C"),
    # Tamil has three nasals / two rhotics / two laterals that Prakrit merges.
    # A Brahmi reader cannot recover them, so the conventional Tamil spelling
    # picks one member of each merged set (and the spell-checker fixes the rest).
    ("N", "n", "C"), ("R", "r", "C"), ("L", "l", "C"),
    # Prakrit au has no Tamil counterpart and regularly becomes long o
    ("W", "O", "V"),
    ("Z", "c", "C"),
    ("A", "A", "V"), ("I", "I", "V"), ("U", "U", "V"), ("E", "E", "V"),
    ("Y", "Y", "V"), ("O", "O", "V"),
    ("a", "a", "V"), ("i", "i", "V"), ("u", "u", "V"), ("e", "e", "V"), ("o", "o", "V"),
]

_ROMAN_TO_TAMIL_CONS = {v: k for k, v in _TAMIL_CONS_ROMAN.items()}
_VOWEL_TO_MAA = {v: g for g, v in _MAA_TO_VOWEL.items() if g}


def _tokenise_roman(roman: str) -> List[Tuple[str, str]]:
    """Split Prakrit romanisation into (kind, tamil-roman) tokens, longest match first."""
    toks: List[Tuple[str, str]] = []
    i = 0
    while i < len(roman):
        for src, dst, kind in PRAKRIT_TAMIL_RULES:
            if roman.startswith(src, i):
                toks.append((kind, dst))
                i += len(src)
                break
        else:
            i += 1
    return toks


_TAMIL_INDEPENDENT_VOWEL = {
    "a": "அ", "A": "ஆ", "i": "இ", "I": "ஈ", "u": "உ", "U": "ஊ",
    "e": "எ", "E": "ஏ", "Y": "ஐ", "o": "ஒ", "O": "ஓ", "W": "ஔ",
}
# A word-final inherent short a is normally lengthened when a Prakrit word is
# taken into Tamil (uru -> ஊரு, matura -> மதுரை), so we do the same here.
_FINAL_A_LENGTHEN = {"a": "A", "i": "I", "u": "U"}


def _merge_vowel_sequences(toks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """'u','u' -> one long U, 'a','i' -> diphthong, 'a','u' -> au.

    Prakrit keeps a historical vowel sequence that Tamil spells as a single
    long vowel or diphthong.  Without this, ``uura`` reads as ``உஉர்``.
    Runs *after* the Prakrit->Tamil rewrite, so the merged diphthong is
    normalised here too (Tamil has no au where Prakrit does -> long o).
    """
    pairs = {"AA": "A", "II": "I", "UU": "U", "EE": "E", "OO": "O",
             "AI": "Y", "AY": "Y", "AU": "O", "AW": "O", "AO": "O", "OI": "Y"}
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(toks):
        cur = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if cur[0] == "V" and nxt is not None and nxt[0] == "V":
            merged = pairs.get(cur[1].upper() + nxt[1].upper())
            if merged:
                out.append(("V", merged))
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def _roman_to_tamil_word(roman: str) -> str:
    toks = _tokenise_roman(roman)
    # Prakrit h-loss between vowels:  ... V h V ... -> drop the h
    toks = [t for i, t in enumerate(toks)
            if not (t[1] == "h" and 0 < i < len(toks) - 1
                    and toks[i - 1][0] == "V" and toks[i + 1][0] == "V")]
    toks = _merge_vowel_sequences(toks)
    if not toks:
        return ""
    # lengthen the vowel of a final inherent-a syllable (k-a -> kaa)
    if toks[-1] == ("V", "a") and toks[-2][0] == "C":
        toks = toks[:-1] + [("V", _FINAL_A_LENGTHEN["a"])]

    out: List[str] = []
    k = 0
    while k < len(toks):
        kind, val = toks[k]
        if kind == "C":
            cons = _ROMAN_TO_TAMIL_CONS.get(val)
            if cons is None:
                k += 1
                continue
            nxt = toks[k + 1] if k + 1 < len(toks) else None
            if nxt is not None and nxt[0] == "V":
                out.append(cons + _VOWEL_TO_MAA.get(nxt[1], ""))
                k += 2
            else:
                out.append(cons + "\u0BCD")           # cluster / word-final
                k += 1
            continue
        out.append(_TAMIL_INDEPENDENT_VOWEL.get(val, ""))
        k += 1
    return "".join(g for g in out if g)


def brahmi_to_tamil(text: str) -> str:
    """Brahmi line -> modern Tamil spelling (via Prakrit transliteration)."""
    words: List[str] = []
    for w in brahmi_to_roman(text).split(" "):
        if not w:
            continue
        if w.isdigit():
            words.append("".join(TAMIL_DIGITS[int(d)] if d != "0" else "௦" for d in w))
            continue
        tw = _roman_to_tamil_word(w)
        if tw:
            words.append(tw)
    return " ".join(words)


if __name__ == "__main__":
    samples = ["மன்னன்", "கிணறு", "ஊர்", "கொடை", "மகன்", "வேள்", "ஐந்து",
               "பெருங்குடி", "கோயில்", "அழகன்", "நெல்", "ஏரி"]
    for w in samples:
        br = tamil_to_brahmi(w)
        print(f"{w:8s} roman={tamil_to_roman(w):10s} brahmi_chars={len(br):2d} "
              f"re-roman={brahmi_to_roman(br):10s} tamil={brahmi_to_tamil(br)}")
