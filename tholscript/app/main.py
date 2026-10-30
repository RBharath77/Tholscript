"""TholScript web app: upload a slab photo (or pick a demo) -> readable text.

Run:  python -m tholscript.app.main          (binds 0.0.0.0:8080)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request

from tholscript.app.service import TranslateService, from_b64

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
svc: TranslateService | None = None


def service() -> TranslateService:
    global svc
    if svc is None:
        t0 = time.time()
        svc = TranslateService()
        print(f"[app] models loaded in {time.time() - t0:.1f}s")
    return svc


@app.get("/")
def index():
    return (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    s = service()
    return jsonify(ok=True, scripts=sorted(s.rec.crnns.keys()))


@app.get("/api/demo/<key>")
def demo(key: str):
    return jsonify(service().demo(key))


@app.post("/api/translate")
def translate():
    data = request.get_json(force=True)
    img = from_b64(data["image_b64"])
    script = data.get("script", "auto")
    t0 = time.time()
    result = service().translate_image(img, script=script)
    result["seconds"] = round(time.time() - t0, 2)
    from tholscript.translation.multilingual import translate_all_languages
    result["translations"] = translate_all_languages(result.get("modern", ""))
    return jsonify(result)


@app.route("/api/translate/languages", methods=["GET", "POST"])
def translate_languages():
    from tholscript.translation.multilingual import translate_all_languages, translate_epigraphic_text
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        target = data.get("target", None)
    else:
        text = request.args.get("text", "")
        target = request.args.get("target", None)

    if target:
        return jsonify(translate_epigraphic_text(text, target_lang=target))
    return jsonify(translations=translate_all_languages(text))


@app.route("/api/voice/spelling", methods=["GET", "POST"])
def voice_spelling():
    from tholscript.translation.voice import spell_text
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        script = data.get("script", "tamil")
    else:
        text = request.args.get("text", "")
        script = request.args.get("script", "tamil")
    return jsonify(spelling=spell_text(text, script=script))


@app.post("/api/export/epidoc")
def export_epidoc():
    from tholscript.translation.epidoc import to_epidoc
    data = request.get_json(force=True)
    xml_str = to_epidoc(data)
    from flask import Response
    return Response(xml_str, mimetype="application/xml",
                    headers={"Content-Disposition": "attachment;filename=inscription_epidoc.xml"})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    service()                      # warm the models before the first click
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)
