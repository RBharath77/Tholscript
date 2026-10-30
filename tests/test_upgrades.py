"""Tests for new modules: Voice/TTS spelling, EpiDoc export, and CTC span decoding."""
from __future__ import annotations

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tholscript.models.crnn import decode_with_spans, beam_search_decode, greedy_decode
from tholscript.translation.voice import spell_cluster, spell_text, make_voice_payload
from tholscript.translation.epidoc import to_epidoc
from tholscript.config import get_device


def test_device_selection():
    dev = get_device()
    assert dev in ("mps", "cuda", "cpu")


def test_decode_with_spans_returns_valid_boundaries():
    torch.manual_seed(42)
    logits = torch.randn(25, 10)
    spans = decode_with_spans(logits)
    assert isinstance(spans, list)
    for cls_idx, conf, t0, t1 in spans:
        assert isinstance(cls_idx, int)
        assert 0.0 <= conf <= 1.0
        assert 0.0 <= t0 <= t1 <= 1.0


def test_beam_search_decode_format():
    torch.manual_seed(42)
    logits = torch.randn(20, 8)
    res = beam_search_decode(logits, beam_width=4)
    assert isinstance(res, list)
    for cls_idx, conf, t0, t1 in res:
        assert isinstance(cls_idx, int)
        assert 0.0 <= conf <= 1.0
        assert 0.0 <= t0 <= t1 <= 1.0


def test_voice_spelling_tamil():
    tokens = spell_text("கோன்", script="tamil")
    assert len(tokens) == 2
    assert tokens[0]["cluster"] == "கோ"
    assert tokens[0]["roman"] == "koo"
    assert tokens[1]["cluster"] == "ன்"
    assert tokens[1]["speak_ta"] == "இன்"  # Traditional pulli pronunciation


def test_voice_spelling_brahmi():
    tokens = spell_text("𑀓", script="brahmi")
    assert len(tokens) >= 1
    assert tokens[0]["cluster"] == "𑀓"


def test_voice_payload_generation():
    dummy = {
        "script": "tamil",
        "modern": "கோன்",
        "gloss": "the lord",
        "lines": [{
            "line_index": 0,
            "raw": "கோன்",
            "corrected": "கோன்",
            "modern": "கோன்",
            "gloss": "the lord",
            "chars": []
        }]
    }
    payload = make_voice_payload(dummy)
    assert payload["script"] == "tamil"
    assert payload["full_modern"] == "கோன்"
    assert len(payload["lines"]) == 1
    assert len(payload["lines"][0]["spelling"]) == 2


def test_epidoc_xml_validity():
    reading = {
        "script": "tamil",
        "modern": "கோன் அதிகன்",
        "gloss": "the lord the chief",
        "lines": [
            {
                "raw": "கோன்",
                "corrected": "கோன்",
                "chars": [
                    {"cluster": "கோ", "doubtful": False},
                    {"cluster": "ன்", "doubtful": True}
                ]
            }
        ]
    }
    xml = to_epidoc(reading, "test-inscr-01")
    assert '<?xml version="1.0" encoding="UTF-8"?>' in xml
    assert '<TEI xmlns="http://www.tei-c.org/ns/1.0"' in xml
    assert '<unclear reason="damage">ன்</unclear>' in xml
    assert '<title>TholScript Inscription Reading: test-inscr-01</title>' in xml


def test_multilingual_translation():
    from tholscript.translation.multilingual import translate_all_languages, translate_epigraphic_text
    sample = "வேந்தன் சேரன் கோயில் கட்டினான் பொன் அளித்தான் வாழ்க"
    res = translate_all_languages(sample)
    assert "te" in res
    assert "kn" in res
    assert "ml" in res
    assert "hi" in res
    assert "sa" in res
    assert "en" in res

    # Telugu contains Raju / Devayam / Nirminchadu
    assert "దేవాలయం" in res["te"]["translated"]
    assert "నిర్మించాడు" in res["te"]["translated"]

    # Kannada contains Devalaya / Kattisidanu
    assert "ದೇವಾಲಯ" in res["kn"]["translated"]
    assert "ಕಟ್ಟಿಸಿದನು" in res["kn"]["translated"]

    # Hindi contains Mandir
    assert "मंदिर" in res["hi"]["translated"]


