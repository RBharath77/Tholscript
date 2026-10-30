"""EpiDoc TEI-XML exporter for TholScript epigraphic readings.

EpiDoc (Epigraphic Documents) is an international standard schema for digital
editions of ancient inscriptions, using the Text Encoding Initiative (TEI) XML.
"""
from __future__ import annotations

import datetime
import html
from typing import Dict, List


def to_epidoc(reading: Dict, inscription_id: str = "tholscript-inscr-001") -> str:
    """Export a reading result into standardized EpiDoc TEI-XML format."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d")
    script = reading.get("script", "tamil")
    lang_code = "ta" if script == "tamil" else "pra"

    lines = reading.get("lines", [])
    edition_lines = []
    for idx, line in enumerate(lines, 1):
        line_num = idx
        raw = line.get("raw", "")
        corrected = line.get("corrected", raw)
        chars = line.get("chars", [])

        # Build line text with <unclear> for doubtful characters
        if chars:
            line_parts = []
            for ch in chars:
                c_text = html.escape(ch.get("cluster", ""))
                if ch.get("doubtful", False):
                    line_parts.append(f"<unclear reason=\"damage\">{c_text}</unclear>")
                else:
                    line_parts.append(c_text)
            encoded_content = "".join(line_parts)
        else:
            encoded_content = html.escape(raw)

        if corrected != raw:
            encoded_line = (
                f'<choice><orig>{encoded_content}</orig>'
                f'<reg>{html.escape(corrected)}</reg></choice>'
            )
        else:
            encoded_line = encoded_content

        edition_lines.append(f'          <lb n="{line_num}"/>{encoded_line}')

    edition_xml = "\n".join(edition_lines)
    full_gloss = html.escape(reading.get("gloss", ""))
    full_modern = html.escape(reading.get("modern", ""))

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:space="preserve" xml:lang="en">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>TholScript Inscription Reading: {inscription_id}</title>
        <editor>TholScript Epigraphic Pipeline</editor>
      </titleStmt>
      <publicationStmt>
        <authority>TholScript Project</authority>
        <date when="{now_str}">{now_str}</date>
        <availability>
          <licence target="https://creativecommons.org/licenses/by/4.0/">CC-BY 4.0</licence>
        </availability>
      </publicationStmt>
      <sourceDesc>
        <p>Digitally transcribed from damaged stone inscription ({script} script).</p>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <langUsage>
        <language ident="{lang_code}">{script.capitalize()}</language>
        <language ident="en">English</language>
      </langUsage>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div type="edition" xml:lang="{lang_code}">
        <ab>
{edition_xml}
        </ab>
      </div>
      <div type="transcription" subtype="modern" xml:lang="ta">
        <p>{full_modern}</p>
      </div>
      <div type="translation" xml:lang="en">
        <p>{full_gloss}</p>
      </div>
    </body>
  </text>
</TEI>"""
    return xml

