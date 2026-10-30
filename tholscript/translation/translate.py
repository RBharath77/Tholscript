"""Ancient -> modern readable text.

Two readings of "modern":

* **modern Tamil**: the recognized clusters, spell-corrected against the
  inscription lexicon (and, for Brahmi input, first re-spelt from Prakrit
  orthography into Tamil orthography by :mod:`brahmi_map`);
* **English gloss**: word-by-word meaning pulled from the epigraphic lexicon,
  which turns ``வேள் நள்ளி கொடை அளித்தான்`` into
  ``The chief (Vel) Nalli gave gift``.

Everything is rule/lexicon based and auditable; nothing is a black box.
"""
from __future__ import annotations

from typing import Dict, List

from tholscript.data.lexicon import GLOSS
from tholscript.translation.brahmi_map import brahmi_to_tamil, tamil_to_roman


def gloss_words(words: List[str]) -> List[str]:
    return [GLOSS.get(w, f"[{w}]") for w in words]


def modern_reading(words: List[str], script: str) -> Dict:
    """Return the modern forms of one recognised line (already spell-corrected)."""
    if script == "brahmi":
        words = [brahmi_to_tamil(w) or w for w in words]
    tamil = " ".join(words)
    roman = [tamil_to_roman(w) for w in words]
    gloss = gloss_words(words)
    return dict(tamil=tamil, roman=roman, gloss=gloss,
                gloss_sentence=" ".join(gloss))


def summarise(result_text: str, script: str, corrected_text: str) -> Dict:
    """Top-level convenience used by the web app and the CLI."""
    lines = []
    for raw, cor in zip(result_text.split("\n"), corrected_text.split("\n")):
        words = cor.split()
        m = modern_reading(words, script)
        lines.append(dict(raw=raw, corrected=cor, **m))
    return dict(script=script, lines=lines,
                full_modern=" ".join(l["tamil"] for l in lines),
                full_gloss=" | ".join(l["gloss_sentence"] for l in lines if l["gloss_sentence"]))


if __name__ == "__main__":
    demo = ["வேள் நள்ளி கொடை அளித்தான்", "மன்னன் மகன் முருகன் கோயில் கட்டினான்"]
    for line in demo:
        r = summarise(line, "tamil", line)
        print(r["lines"][0]["tamil"])
        print("   roman:", " ".join(r["lines"][0]["roman"]))
        print("   gloss:", r["lines"][0]["gloss_sentence"])
