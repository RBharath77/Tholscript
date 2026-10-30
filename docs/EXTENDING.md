# Fine-tuning on real inscription corpora

The synthetic pipeline exists so the models start from a sane place; real
epigraphy adds palaeographic variation (century-specific letter forms, ligatures,
tool styles) that only real scans teach.  Everything downstream of the datasets
is dataset-agnostic, so wiring a real corpus in touches **two small classes**.

## 1. Line images (CRNN)

Implement the same contract `tholscript/scripts/train_crnn.py::LineDataset` uses:

```python
class RealLineDataset(torch.utils.data.Dataset):
    """One item = (ink tensor 1x64xW, [class ids], text)."""
    def __getitem__(self, i):
        photo  = load_scan(i)                 # uint8 photo of ONE line
        ink    = enhance(photo)               # tholscript.vision.preprocess.enhance
        ink    = cv2.resize(ink, (W, 64))     # any W
        clusters = clusterize(label_text(i), script)   # tholscript.data.inventory
        targets  = [class_index[c] for c in clusters]  # classes_for(script)
        return torch.from_numpy(ink).float().unsqueeze(0), \
               torch.tensor(targets), label_text(i)
```

Then swap it into `train_crnn.main`.  Start from the shipped checkpoint:
`model.load_state_dict(torch.load("artifacts/crnn_tamil.pt")["model"])` and use
a learning rate 10× smaller.

### Candidate corpora

* **Brahmi** – IIT Kanpur *"Brahmi script character dataset"* (isolated characters,
  perfect for the glyph CNN) and the Ashokan edict scans used in several CTC
  papers; also the open *"Millennium* EGD"* epigraphic glyph database.
* **Tamil** – Tamil Nadu Archaeology Department / French Institute of
  Pondicherry inscription scans; several Kaggle *"Tamil heritage OCR"* mirrors
  exist.  Crop lines with `segment_lines` — it is exactly the function the web
  app runs.

## 2. Glyph patches (glyph CNN)

`RealGlyphDataset.__getitem__` returns `(1x56x56 tensor, class_id)`; reuse
`extract_glyph_patch` on the line crops and the character boxes from your
annotator.  Include 10 % of *stone-only* patches as `<unk>` exactly like
`train_glyph.GlyphDataset` does.

## 3. Vocabulary

Add real words to `tholscript/data/lexicon.py::ALL_WORDS` (and `GLOSS` if you want
English meanings) — the spell-checker and LM pick them up automatically.

## 4. New scripts

Add a font + inventory block (`tholscript/data/inventory.py`), a transliteration
table (`tholscript/translation/brahmi_map.py` style) and a `FONTS[script]` entry;
the renderer, trainer and pipeline are parameterised by `script` everywhere.
