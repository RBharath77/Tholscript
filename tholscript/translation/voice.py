"""Voice & Audio module for TholScript.

Provides text-to-speech phonetic breakdown, letter-by-letter epigraphic spelling,
and audio synchronization metadata for modern Tamil, Brahmi, and English glosses.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from tholscript.data.inventory import clusterize, tamil_cluster_roman
from tholscript.translation.brahmi_map import brahmi_to_tamil, tamil_to_roman

# Traditional Tamil phonetic names for pulli consonants (pure consonants)
# Prefixing with 'இ' reflects traditional Tamil grammar (மெய்யெழுத்துக்கள்)
# and ensures browser TTS engines pronounce the isolated stop crisply.
_PULLI_PRONUNCIATION: Dict[str, str] = {
    "க்": "இக்", "ங்": "இங்", "ச்": "இச்", "ஞ்": "இஞ்",
    "ட்": "இட்", "ண்": "இண்", "த்": "இத்", "ந்": "இந்",
    "ப்": "இப்", "ம்": "இம்", "ய்": "இய்", "ர்": "இர்",
    "ல்": "இல்", "வ்": "இவ்", "ழ்": "இழ்", "ள்": "இள்",
    "ற்": "இற்", "ன்": "இன்",
}


def spell_cluster(cluster: str, script: str = "tamil") -> Dict[str, str]:
    """Breakdown a single grapheme cluster for display and audio synthesis."""
    if script == "brahmi":
        # Map Brahmi cluster to Tamil equivalent for audio pronunciation
        tamil_equiv = brahmi_to_tamil(cluster) or cluster
        roman = tamil_to_roman(tamil_equiv)
        speak_ta = _PULLI_PRONUNCIATION.get(tamil_equiv, tamil_equiv)
        return {
            "cluster": cluster,
            "tamil_equiv": tamil_equiv,
            "roman": roman,
            "speak_ta": speak_ta,
            "speak_en": roman,
        }

    # Tamil cluster
    roman = tamil_cluster_roman(cluster)
    speak_ta = _PULLI_PRONUNCIATION.get(cluster, cluster)
    return {
        "cluster": cluster,
        "tamil_equiv": cluster,
        "roman": roman,
        "speak_ta": speak_ta,
        "speak_en": roman,
    }


def spell_text(text: str, script: str = "tamil") -> List[Dict[str, str]]:
    """Convert a line or sentence into a list of spellable tokens."""
    clusters = clusterize(text, script=script)
    out: List[Dict[str, str]] = []
    for c in clusters:
        if c.strip() == "":
            out.append({"cluster": " ", "tamil_equiv": " ", "roman": " ", "speak_ta": "", "speak_en": ""})
        else:
            out.append(spell_cluster(c, script=script))
    return out


def make_voice_payload(reading: Dict) -> Dict:
    """Enrich the translation reading result with audio and spelling metadata."""
    lines_voice = []
    for line in reading.get("lines", []):
        raw_text = line.get("corrected") or line.get("raw") or ""
        script = reading.get("script", "tamil")
        modern_text = line.get("modern") or raw_text
        gloss_text = line.get("gloss") or ""

        # Produce spelling units aligned with the characters
        spelling_tokens = spell_text(raw_text, script=script)
        lines_voice.append({
            "line_index": line.get("line_index", 0),
            "text": raw_text,
            "modern": modern_text,
            "gloss": gloss_text,
            "spelling": spelling_tokens,
        })

    return {
        "script": reading.get("script", "tamil"),
        "full_modern": reading.get("modern", ""),
        "full_gloss": reading.get("gloss", ""),
        "lines": lines_voice,
    }

