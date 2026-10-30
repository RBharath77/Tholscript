"""Character inventories for the two scripts this project reads.

Tamil (U+0B80 block) is written with an *abugida* whose vowel signs (maatras)
are attached to the consonant.  The recognition vocabulary is therefore the set
of **grapheme clusters** a Tamil inscription actually contains:

    vowel          ->  அ ஆ இ ஈ உ ஊ எ ஏ ஐ ஒ ஓ ஔ  (12)
    consonant      ->  க ங ச ஞ ட ண த ந ப ம ய ர ல வ ழ ள ற ன  (18)
    cons+short maa ->  கி கு கெ கை ...                       (18 x 5)
    cons+long maa  ->  கா கீ கூ கே கோ ...                    (18 x 5)
    cons+au sign   ->  கௌ ...                                 (18)
    cons+viraama   ->  க் ...                                 (18)
    numerals       ->  ௦ ௧ ௨ ௩ ௪ ௫ ௬ ௭ ௮ ௯  (10) + ௰ ௱ ௲  (3)

Brahmi (U+11000 block, Unicode 4.1+) is the ancestor script used in the
Ashokan rock edicts and in the Tamil-Brahmi cave inscriptions of Tamil Nadu.
Every codepoint carries its Unicode name, its scholarly transliteration
(IAST-ish) and the Tamil glyph it is conventionally equated with, which is
what powers the Brahmi -> Tamil transliteration stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Tamil
# --------------------------------------------------------------------------- #
TAMIL_VOWELS = "அஆஇஈஉஊஎஏஐஒஓஔ"
TAMIL_CONSONANTS = "கஙசஞடணதநபமயரலவழளறன"
TAMIL_MAATRAS = {
    "a": "",        # inherent
    "aa": "\u0BBE",  # ா
    "i": "\u0BBF",   # ி
    "ii": "\u0BC0",  # ீ
    "u": "\u0BC1",   # ு
    "uu": "\u0BC2",  # ூ
    "e": "\u0BC6",   # ெ
    "ee": "\u0BC7",  # ே
    "ai": "\u0BC8",  # ை
    "o": "\u0BCA",   # ொ
    "oo": "\u0BCB",  # ோ
    "au": "\u0BCC",  # ௌ
    "": "\u0BCD",    # pulli / viraama
}
TAMIL_DIGITS = "௦௧௨௩௪௫௬௭௮௯"
TAMIL_NUMBER_SIGNS = "௰௱௲"

# romanisation of the Tamil inventory (used for readable reports & the LM)
_TAMIL_VOWEL_ROMAN = {"அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ii", "உ": "u", "ஊ": "uu",
                      "எ": "e", "ஏ": "ee", "ஐ": "ai", "ஒ": "o", "ஓ": "oo", "ஔ": "au"}
_TAMIL_CONS_ROMAN = {"க": "k", "ங": "ng", "ச": "c", "ஞ": "ny", "ட": "t", "ண": "n",
                     "த": "th", "ந": "n", "ப": "p", "ம": "m", "ய": "y", "ர": "r",
                     "ல": "l", "வ": "v", "ழ": "zh", "ள": "l2", "ற": "r2", "ன": "n2"}
_TAMIL_DIGIT_ROMAN = {d: str(i) for i, d in enumerate(TAMIL_DIGITS)}
_TAMIL_DIGIT_ROMAN.update({"௰": "10", "௱": "100", "௲": "1000"})


def tamil_cluster_roman(cluster: str) -> str:
    """கி -> 'ki', க் -> 'k', அ -> 'a', ௩ -> '3'."""
    if cluster in _TAMIL_VOWEL_ROMAN:
        return _TAMIL_VOWEL_ROMAN[cluster]
    if cluster in _TAMIL_DIGIT_ROMAN:
        return _TAMIL_DIGIT_ROMAN[cluster]
    if not cluster or cluster[0] not in _TAMIL_CONS_ROMAN:
        return "?"
    base = _TAMIL_CONS_ROMAN[cluster[0]]
    maa = cluster[1] if len(cluster) > 1 else ""
    for key, glyph in TAMIL_MAATRAS.items():
        if glyph == maa:
            return base if key == "" else base + key
    return base


def build_tamil_inventory() -> List[str]:
    inv: List[str] = list(TAMIL_VOWELS)
    inv += list(TAMIL_CONSONANTS)
    for c in TAMIL_CONSONANTS:
        for key in ("aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au", ""):
            inv.append(c + TAMIL_MAATRAS[key])
    inv += list(TAMIL_DIGITS) + list(TAMIL_NUMBER_SIGNS)
    # de-duplicate, keep order (க + "" is just க)
    seen, out = set(), []
    for ch in inv:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


# --------------------------------------------------------------------------- #
# Brahmi  (codepoint -> (unicode-ish name, transliteration, tamil equivalent))
# --------------------------------------------------------------------------- #
BRAHMI: Dict[str, Tuple[str, str, str]] = {
    "\U00011005": ("BRAHMI LETTER A", "a", "அ"),
    "\U00011006": ("BRAHMI LETTER AA", "aa", "ஆ"),
    "\U00011007": ("BRAHMI LETTER I", "i", "இ"),
    "\U00011008": ("BRAHMI LETTER II", "ii", "ஈ"),
    "\U00011009": ("BRAHMI LETTER U", "u", "உ"),
    "\U0001100A": ("BRAHMI LETTER UU", "uu", "ஊ"),
    "\U0001100B": ("BRAHMI LETTER VOCALIC R", "r", "ரு"),
    "\U0001100C": ("BRAHMI LETTER VOCALIC L", "l", "லு"),
    "\U0001100D": ("BRAHMI LETTER VOCALIC LL", "ll", "ளூ"),
    "\U0001100E": ("BRAHMI LETTER E", "e", "எ"),
    "\U0001100F": ("BRAHMI LETTER EE", "ee", "ஏ"),
    "\U00011010": ("BRAHMI LETTER AI", "ai", "ஐ"),
    "\U00011011": ("BRAHMI LETTER O", "o", "ஒ"),
    "\U00011012": ("BRAHMI LETTER AU", "au", "ஔ"),
    "\U00011013": ("BRAHMI LETTER KA", "ka", "க"),
    "\U00011014": ("BRAHMI LETTER KHA", "kha", "க"),
    "\U00011015": ("BRAHMI LETTER GA", "ga", "க"),
    "\U00011016": ("BRAHMI LETTER GHA", "gha", "க"),
    "\U00011017": ("BRAHMI LETTER NGA", "nga", "ங"),
    "\U00011018": ("BRAHMI LETTER CA", "ca", "ச"),
    "\U00011019": ("BRAHMI LETTER CHA", "cha", "ச"),
    "\U0001101A": ("BRAHMI LETTER JA", "ja", "ஜ"),
    "\U0001101B": ("BRAHMI LETTER JHA", "jha", "ஜ"),
    "\U0001101C": ("BRAHMI LETTER NYA", "nya", "ஞ"),
    "\U0001101D": ("BRAHMI LETTER TTA", "ta", "ட"),
    "\U0001101E": ("BRAHMI LETTER TTHA", "tha", "ட"),
    "\U0001101F": ("BRAHMI LETTER DDA", "da", "ட"),
    "\U00011020": ("BRAHMI LETTER DDHA", "dha", "ட"),
    "\U00011021": ("BRAHMI LETTER NNA", "na", "ண"),
    "\U00011022": ("BRAHMI LETTER TA", "ta", "த"),
    "\U00011023": ("BRAHMI LETTER THA", "tha", "த"),
    "\U00011024": ("BRAHMI LETTER DA", "da", "த"),
    "\U00011025": ("BRAHMI LETTER DHA", "dha", "த"),
    "\U00011026": ("BRAHMI LETTER NA", "na", "ந"),
    "\U00011027": ("BRAHMI LETTER PA", "pa", "ப"),
    "\U00011028": ("BRAHMI LETTER PHA", "pha", "ப"),
    "\U00011029": ("BRAHMI LETTER BA", "ba", "ப"),
    "\U0001102A": ("BRAHMI LETTER BHA", "bha", "ப"),
    "\U0001102B": ("BRAHMI LETTER MA", "ma", "ம"),
    "\U0001102C": ("BRAHMI LETTER YA", "ya", "ய"),
    "\U0001102D": ("BRAHMI LETTER RA", "ra", "ர"),
    "\U0001102E": ("BRAHMI LETTER LA", "la", "ல"),
    "\U0001102F": ("BRAHMI LETTER VA", "va", "வ"),
    "\U00011030": ("BRAHMI LETTER SHA", "sha", "ஸ"),
    "\U00011031": ("BRAHMI LETTER SSA", "ssa", "ஷ"),
    "\U00011032": ("BRAHMI LETTER SA", "sa", "ஸ"),
    "\U00011033": ("BRAHMI LETTER HA", "ha", "ஹ"),
    "\U00011034": ("BRAHMI LETTER LLA", "lla", "ள"),
    "\U00011035": ("BRAHMI VIRAMA", "", ""),
    "\U00011036": ("BRAHMI VOWEL SIGN AA", "aa", "\u0BBE"),
    "\U00011037": ("BRAHMI VOWEL SIGN I", "i", "\u0BBF"),
    "\U00011038": ("BRAHMI VOWEL SIGN II", "ii", "\u0BC0"),
    "\U00011039": ("BRAHMI VOWEL SIGN U", "u", "\u0BC1"),
    "\U0001103A": ("BRAHMI VOWEL SIGN UU", "uu", "\u0BC2"),
    "\U0001103B": ("BRAHMI VOWEL SIGN VOCALIC R", "r", "ரு"),
    "\U0001103C": ("BRAHMI VOWEL SIGN VOCALIC RR", "rr", "றூ"),
    "\U0001103D": ("BRAHMI VOWEL SIGN VOCALIC L", "l", "லு"),
    "\U0001103E": ("BRAHMI VOWEL SIGN E", "e", "\u0BC6"),
    "\U0001103F": ("BRAHMI VOWEL SIGN EE", "ee", "\u0BC7"),
    "\U00011040": ("BRAHMI VOWEL SIGN AI", "ai", "\u0BC8"),
    "\U00011041": ("BRAHMI VOWEL SIGN O", "o", "\u0BCA"),
    "\U00011042": ("BRAHMI VOWEL SIGN AU", "au", "\u0BCC"),
    "\U00011043": ("BRAHMI SIGN CANDRABINDU", "m", "\u0BAE\u0BCD"),
    "\U00011044": ("BRAHMI ANUSVARA", "m", "\u0BAE\u0BCD"),
    "\U00011045": ("BRAHMI VISARGA", "h", ""),
    "\U00011046": ("BRAHMI PUNCTUATION", ".", "."),
    "\U00011052": ("BRAHMI NUMBER ONE", "1", "௧"),
    "\U00011053": ("BRAHMI NUMBER TWO", "2", "௨"),
    "\U00011054": ("BRAHMI NUMBER THREE", "3", "௩"),
    "\U00011055": ("BRAHMI NUMBER FOUR", "4", "௪"),
    "\U00011056": ("BRAHMI NUMBER FIVE", "5", "௫"),
    "\U00011057": ("BRAHMI NUMBER SIX", "6", "௬"),
    "\U00011058": ("BRAHMI NUMBER SEVEN", "7", "௭"),
    "\U00011059": ("BRAHMI NUMBER EIGHT", "8", "௮"),
    "\U0001105A": ("BRAHMI NUMBER NINE", "9", "௯"),
    "\U0001105B": ("BRAHMI NUMBER TEN", "10", "௰"),
    "\U0001105C": ("BRAHMI NUMBER TWENTY", "20", ""),
    "\U0001105D": ("BRAHMI NUMBER THIRTY", "30", ""),
    "\U0001105E": ("BRAHMI NUMBER FORTY", "40", ""),
    "\U0001105F": ("BRAHMI NUMBER FIFTY", "50", ""),
    "\U00011060": ("BRAHMI NUMBER SIXTY", "60", ""),
    "\U00011061": ("BRAHMI NUMBER SEVENTY", "70", ""),
    "\U00011062": ("BRAHMI NUMBER EIGHTY", "80", ""),
    "\U00011063": ("BRAHMI NUMBER NINETY", "90", ""),
    "\U00011064": ("BRAHMI NUMBER ONE HUNDRED", "100", "௱"),
    "\U00011065": ("BRAHMI NUMBER ONE THOUSAND", "1000", "௲"),
}

BRAHMI_CONSONANTS = [ch for ch, (n, t, _) in BRAHMI.items() if t and t[-1] == "a" and len(t) <= 4]
BRAHMI_VOWELS = ["\U00011005", "\U00011006", "\U00011007", "\U00011008", "\U00011009",
                 "\U0001100A", "\U0001100E", "\U0001100F", "\U00011010", "\U00011011",
                 "\U00011012"]
BRAHMI_MAATRAS = ["\U00011036", "\U00011037", "\U00011038", "\U00011039", "\U0001103A",
                  "\U0001103E", "\U0001103F", "\U00011040", "\U00011041", "\U00011042"]
BRAHMI_DIGITS = [chr(c) for c in range(0x11052, 0x11066)]
VIRAMA = "\U00011035"


def build_brahmi_inventory() -> List[str]:
    """Grapheme clusters used for Brahmi line recognition.

    A Brahmi syllable is  consonant [+ vowel sign] | consonant + virama
    | independent vowel | digit.
    """
    inv: List[str] = list(BRAHMI_VOWELS)
    inv += list(BRAHMI_DIGITS)
    for c in BRAHMI_CONSONANTS:
        inv.append(c)                       # inherent /a/
        inv.append(c + VIRAMA)              # consonant cluster half
        for m in BRAHMI_MAATRAS:
            inv.append(c + m)
    inv.append("\U00011044")                # anusvara
    inv.append("\U00011046")                # punctuation
    seen, uniq = set(), []
    for c in inv:                            # AA doubles as independent vowel + sign
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


TAMIL_INVENTORY = build_tamil_inventory()
BRAHMI_INVENTORY = build_brahmi_inventory()

TAMIL_BLANK = "<blank>"     # CTC blank
TAMIL_UNK = "<unk>"         # glyph-CNN "I cannot read this stone" class
BRAHMI_BLANK = "<blank>"
BRAHMI_UNK = "<unk>"


def classes_for(script: str) -> List[str]:
    """Full class list for a script, index 0 reserved for CTC blank."""
    if script == "tamil":
        inv = TAMIL_INVENTORY
    elif script == "brahmi":
        inv = BRAHMI_INVENTORY
    else:
        raise ValueError(f"unknown script {script!r}")
    return [TAMIL_BLANK] + sorted(inv)


def glyph_classes_for(script: str) -> List[str]:
    """Class list for the glyph CNN: real clusters + an explicit 'unreadable'."""
    inv = TAMIL_INVENTORY if script == "tamil" else BRAHMI_INVENTORY
    return sorted(inv) + [TAMIL_UNK]


_TAMIL_MATRA_SET = set(v for v in TAMIL_MAATRAS.values() if v)
_TAMIL_CONS_SET = set(TAMIL_CONSONANTS)
_BRAHMI_CONS_SET = set(BRAHMI_CONSONANTS)
_BRAHMI_SIGN_SET = set(BRAHMI_MAATRAS) | {VIRAMA}


def clusterize(text: str, script: str) -> List[str]:
    """Split a raw Unicode string into grapheme clusters.

    Tamil: consonant + optional maatra/pulli form one unit; independent vowels,
    digits and punctuation stand alone.  Brahmi: consonant + optional vowel
    sign or virama form one unit.  This is the unit the CTC reader predicts and
    the unit the renderer draws with a single pen advance.
    """
    out: List[str] = []
    for ch in text:
        if ch == " ":
            continue
        if script == "tamil":
            attach = ch in _TAMIL_MATRA_SET and out and out[-1][0] in _TAMIL_CONS_SET
        else:
            attach = ch in _BRAHMI_SIGN_SET and out and out[-1][0] in _BRAHMI_CONS_SET
        if attach:
            out[-1] += ch
        else:
            out.append(ch)
    return out


if __name__ == "__main__":  # quick inventory audit
    print(f"tamil grapheme clusters : {len(TAMIL_INVENTORY)}")
    print(f"brahmi grapheme clusters: {len(BRAHMI_INVENTORY)}")
    print("sample tamil :", TAMIL_INVENTORY[:14], "...")
    print("roman of கி க் அ ௩ :", tamil_cluster_roman("கி"), tamil_cluster_roman("க்"),
          tamil_cluster_roman("அ"), tamil_cluster_roman("௩"))
