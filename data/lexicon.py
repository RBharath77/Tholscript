"""Lexicon + inscription-text generator.

The synthetic training corpus is not random letters: it is built from
templates that imitate South-Indian epigraphy (hero stones, temple donation
records, well/pond grants, cave labels).  That matters three times over:

1. the recognizer learns realistic cluster *sequences*, not gibberish;
2. the post-recognition language model has something to score against;
3. the spell-checker has a real vocabulary to snap near-misses back to.

The word lists are ordinary modern-Tamil / epigraphic vocabulary; nothing here
claims to reproduce a specific published record.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
TITLES = ["வேந்தன்", "மன்னன்", "அரசன்", "வேள்", "கோன்", "அதிகன்", "பெருமகன்", "சேரன்", "பாண்டியன்"]
NAMES = ["அழகன்", "முருகன்", "காரி", "ஓரி", "பாரி", "நள்ளி", "ஆய்", "எழினி",
         "வழுதி", "சேரல்", "மாறன்", "சாத்தன்", "பூதன்", "குமரன்", "தேவன்", "நம்பி"]
PLACES = ["கொங்கு", "வஞ்சி", "மதுரை", "உறையூர்", "கருவூர்", "கொடுங்கோளூர்", "தஞ்சாவூர்",
          "காஞ்சி", "சேரநாடு", "பாண்டிநாடு", "சோழநாடு", "வேங்கைநகர்"]
VILLAGES = ["ஊர்", "குடி", "பட்டி", "மங்கலம்", "சேரி", "கோட்டம்", "நாடு", "வளநாடு",
            "பெருங்குடி", "தனிக்குடி", "மேலூர்", "கீழூர்", "அரிட்டாபட்டி", "புலியூர்"]
STRUCTURES = ["கோயில்", "கோவில்", "மடம்", "சத்திரம்", "கிணறு", "ஏரி", "தெப்பம்",
              "மண்டபம்", "கல்வெட்டு", "படித்துறை", "தண்ணீர்ப்பந்தல்", "சாவடி"]
OFFERING = ["கொடை", "தானம்", "வழிபாடு", "பூசை", "நெல்", "நிலம்", "பொன்",
            "நிறை", "கலம்", "கழஞ்சு", "மா", "பணம்", "ஆடு", "பசு"]
VERBS = ["அளித்தான்", "கொடுத்தான்", "செய்தான்", "கட்டினான்", "அமைத்தான்",
         "குடைந்தான்", "எழுதினான்", "பாதுகாத்தான்", "நிறுவினான்", "விடுத்தான்"]
VERBS_PL = ["அளித்தனர்", "கொடுத்தனர்", "செய்தனர்", "கட்டினர்", "அமைத்தனர்", "எழுதினர்"]
HONORIFICS = ["ஆண்டு", "பிறந்த", "வாழ்க", "நின்ற", "இறந்த", "வீரன்", "நடுகல்",
              "படைவீரன்", "யானை", "புலி", "போர்"]
RELATIVES = ["மகன்", "மகள்", "மனைவி", "தம்பி", "அண்ணன்", "தந்தை", "தாய்", "மருமகன்"]
MONTHS = ["தை", "மாசி", "பங்குனி", "சித்திரை", "வைகாசி", "ஆனி", "ஆடி", "ஆவணி",
          "புரட்டாசி", "ஐப்பசி", "கார்த்திகை", "மார்கழி"]
CONNECTORS = ["இந்த", "அந்த", "இதன்", "இவன்", "என்ற", "என்னும்", "ஆக", "இனி", "இங்கு"]
NUMERAL_WORDS = ["ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது", "பத்து"]

ALL_WORDS: List[str] = sorted(
    set(TITLES + NAMES + PLACES + VILLAGES + STRUCTURES + OFFERING + VERBS +
        VERBS_PL + HONORIFICS + RELATIVES + MONTHS + CONNECTORS + NUMERAL_WORDS)
)

# A compact English gloss used for the "modern readable text" column of the app.
GLOSS = {
    "வேந்தன்": "the king", "மன்னன்": "the king", "அரசன்": "the king", "வேள்": "the chief (Vel)",
    "கோன்": "the lord", "அதிகன்": "the lord (Adigan)", "பெருமகன்": "the great one",
    "சேரன்": "the Chera", "பாண்டியன்": "the Pandya",
    "கொங்கு": "Kongu", "வஞ்சி": "Vanchi", "மதுரை": "Madurai", "உறையூர்": "Uraiyur",
    "கருவூர்": "Karur", "கொடுங்கோளூர்": "Kodungallur", "தஞ்சாவூர்": "Thanjavur",
    "காஞ்சி": "Kanchi", "சேரநாடு": "the Chera country", "பாண்டிநாடு": "the Pandya country",
    "சோழநாடு": "the Chola country", "வேங்கைநகர்": "Vengainagar",
    "ஊர்": "village", "குடி": "hamlet", "பட்டி": "pasture settlement", "மங்கலம்": "brahmin settlement",
    "சேரி": "hamlet", "கோட்டம்": "district", "நாடு": "region", "வளநாடு": "sub-region",
    "பெருங்குடி": "Perungudi", "தனிக்குடி": "Thanikkudi", "மேலூர்": "Melur", "கீழூர்": "Keelur",
    "அரிட்டாபட்டி": "Arittapatti", "புலியூர்": "Puliyur",
    "கோயில்": "temple", "கோவில்": "temple", "மடம்": "monastery", "சத்திரம்": "rest house",
    "கிணறு": "well", "ஏரி": "tank", "தெப்பம்": "raft/festival float", "மண்டபம்": "hall",
    "கல்வெட்டு": "inscription", "படித்துறை": "bathing ghat", "தண்ணீர்ப்பந்தல்": "water shed",
    "சாவடி": "village council hall",
    "கொடை": "gift", "தானம்": "donation", "வழிபாடு": "worship", "பூசை": "ritual",
    "நெல்": "paddy", "நிலம்": "land", "பொன்": "gold", "நிறை": "a measure", "கலம்": "kalams of grain",
    "கழஞ்சு": "kalanchu of gold", "மா": "ma (grain measure)", "பணம்": "coins",
    "ஆடு": "sheep", "பசு": "cow",
    "அளித்தான்": "gave", "கொடுத்தான்": "donated", "செய்தான்": "made", "கட்டினான்": "built",
    "அமைத்தான்": "installed", "குடைந்தான்": "excavated", "எழுதினான்": "wrote",
    "பாதுகாத்தான்": "protected", "நிறுவினான்": "erected", "விடுத்தான்": "issued",
    "அளித்தனர்": "gave", "கொடுத்தனர்": "donated", "செய்தனர்": "made", "கட்டினர்": "built",
    "அமைத்தனர்": "installed", "எழுதினர்": "wrote",
    "ஆண்டு": "year", "பிறந்த": "born", "வாழ்க": "long live", "நின்ற": "standing",
    "இறந்த": "who fell", "வீரன்": "hero", "நடுகல்": "hero stone", "படைவீரன்": "soldier",
    "யானை": "elephant", "புலி": "tiger", "போர்": "battle",
    "மகன்": "son", "மகள்": "daughter", "மனைவி": "wife", "தம்பி": "younger brother",
    "அண்ணன்": "elder brother", "தந்தை": "father", "தாய்": "mother", "மருமகன்": "nephew",
    "தை": "Tai (month)", "மாசி": "Masi (month)", "பங்குனி": "Pankuni (month)",
    "சித்திரை": "Chittirai (month)", "வைகாசி": "Vaikasi (month)", "ஆனி": "Ani (month)",
    "ஆடி": "Adi (month)", "ஆவணி": "Avani (month)", "புரட்டாசி": "Purattasi (month)",
    "ஐப்பசி": "Aippasi (month)", "கார்த்திகை": "Karthigai (month)", "மார்கழி": "Margazhi (month)",
    "இந்த": "this", "அந்த": "that", "இதன்": "its", "இவன்": "he", "என்ற": "named",
    "என்னும்": "called", "ஆக": "as", "இனி": "henceforth", "இங்கு": "here",
    "ஒன்று": "one", "இரண்டு": "two", "மூன்று": "three", "நான்கு": "four", "ஐந்து": "five",
    "ஆறு": "six", "ஏழு": "seven", "எட்டு": "eight", "ஒன்பது": "nine", "பத்து": "ten",
}

# --------------------------------------------------------------------------- #
# Sentence templates
# --------------------------------------------------------------------------- #
TEMPLATES: List[str] = [
    "{title} {name} {place} {village} {structure} {offering} {verb}",
    "{name} {rel} {name2} {structure} {verb}",
    "{month} {connector} {numword} {offering} {title} {name} {verb}",
    "{village} {connector} {structure} {title} {name} {verb}",
    "{title} {name} {place} {village} {structure} {offering} {offering} {verb}",
    "{honorific} {name} {rel} {name2} {structure} {verb}",
    "{connector} {village} {offering} {numword} {title} {name} {verb_pl}",
    "{place} {connector} {village} {structure} {title} {name} {verb}",
    "{title} {name} {month} {offering} {numword} {structure} {verb}",
    "{name} {honorific} {rel} {name2} {village} {structure} {verb}",
]

BRAHMI_TEMPLATES: List[List[str]] = [
    # short cave-shelter labels: name + epithet + "gift/cave"
    ["{name}", "{rel}", "{name2}", "{offering}"],
    ["{title}", "{name}", "{place}", "{offering}"],
    ["{name}", "{rel}", "{name2}", "{place}", "{offering}"],
    ["{title}", "{name}", "{offering}", "{numword}"],
    ["{place}", "{village}", "{title}", "{name}", "{offering}"],
]

_FILLERS = {
    "title": TITLES, "name": NAMES, "name2": NAMES, "place": PLACES, "village": VILLAGES,
    "structure": STRUCTURES, "offering": OFFERING, "verb": VERBS, "verb_pl": VERBS_PL,
    "honorific": HONORIFICS, "rel": RELATIVES, "month": MONTHS, "connector": CONNECTORS,
    "numword": NUMERAL_WORDS,
}


def _fill(template: str, rng: random.Random) -> str:
    words = []
    for tok in template.split():
        key = tok.strip("{}")
        if key in _FILLERS:
            words.append(rng.choice(_FILLERS[key]))
        else:
            words.append(tok)
    return " ".join(words)


@dataclass
class LineSample:
    text: str                  # Tamil (or Brahmi) text, the CTC target
    words: List[str]
    script: str


def sample_tamil_line(rng: random.Random) -> LineSample:
    text = _fill(rng.choice(TEMPLATES), rng)
    return LineSample(text=text, words=text.split(), script="tamil")


def sample_brahmi_words(rng: random.Random) -> List[str]:
    """Tamil words re-spelt in Brahmi orthography.

    Tamil-Brahmi inscriptions write the local language with the Brahmi
    inventory, so we transliterate the Tamil lexicon into Brahmi clusters
    (see :mod:`tholscript.translation.brahmi_map`).
    """
    from tholscript.translation.brahmi_map import tamil_to_brahmi

    template = rng.choice(BRAHMI_TEMPLATES)
    out = []
    for tok in template:
        key = tok.strip("{}")
        word = rng.choice(_FILLERS[key]) if key in _FILLERS else tok
        br = tamil_to_brahmi(word)
        if br:
            out.append(br)
    return out


def sample_brahmi_line(rng: random.Random) -> LineSample:
    words = sample_brahmi_words(rng)
    if not words:
        words = sample_brahmi_words(random.Random(0))
    return LineSample(text=" ".join(words), words=words, script="brahmi")


def sample_line(rng: random.Random, script: str = "tamil") -> LineSample:
    return sample_brahmi_line(rng) if script == "brahmi" else sample_tamil_line(rng)


def build_word_lexicon() -> List[str]:
    """Vocabulary used by the spell-checker / language model."""
    return list(ALL_WORDS)


def gloss_sentence(words: Sequence[str]) -> str:
    """Rough word-by-word English gloss (for the 'modern readable text' pane)."""
    parts = []
    for w in words:
        if w in GLOSS:
            parts.append(GLOSS[w])
        elif w.strip().isdigit() or w.isdigit():
            parts.append(w)
        else:
            parts.append(w)
    out = " ".join(parts)
    return out[0].upper() + out[1:] + "." if out else ""


if __name__ == "__main__":
    rng = random.Random(7)
    for _ in range(4):
        s = sample_tamil_line(rng)
        print(s.text)
        print("   ->", gloss_sentence(s.words))
    for _ in range(3):
        print("BRAHMI:", sample_brahmi_line(rng).text)
    print("lexicon size:", len(build_word_lexicon()))
