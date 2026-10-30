# Design notes — why the pipeline is built this way

## The central decision: train on *physics*, not on clean glyphs

An OCR model trained on clean rendered text scores ~99 % on clean text and
falls apart on a rain-worn hero stone, which is the only text this project
cares about.  So the training distribution is **render → damage → photograph**:

* every sample passes through a stochastic damage engine whose effects are
  named physical processes (chipping = flaking, worn band = erosion, raking
  light = directional illumination, …);
* the damage *level* is drawn uniformly per sample, so one network covers the
  whole clean-to-ruined range and the evaluator can report the accuracy-vs-
  damage curve that tells a user when to trust a reading.

The same `enhance()` function is applied during training and at inference, so
the network only ever sees recovered ink maps — the stone surface, lighting
and camera noise are removed *before* recognition, which is what makes a small
CPU-sized network viable.

## Why CTC over the whole line instead of segment-then-classify

Ruined characters merge, chip and fade.  Any pipeline that must first cut the
line into character boxes inherits a brittle step whose errors are
unrecoverable.  Connectionist Temporal Classification (CTC) over a CRNN reads
the **sequence of grapheme clusters** straight from the line image; no cuts,
no ordering assumptions.  (The classical segment-then-classify path still
exists as the *glyph CNN*, but as a cross-check, not as the spine.)

Two implementation lessons are encoded in `models/crnn.py`:

1. **BatchNorm after every conv.**  Without normalisation the stroke signal
   attenuates through the stack (measured ink/background column-energy ratio
   2.8 → 1.1 over four stages) and CTC collapses to all-blank.
2. **Pool the width 8×.**  At ~40 columns per character the blank posterior
   dominates and gradients to real characters vanish; at ~6-10 frames per
   character training escapes the blank within a few hundred steps.
3. **Warm-start with dense frame supervision.**  Plain CTC needed thousands of
   steps to reach what per-column classification against the *known character
   boxes* of the synthetic renders reaches in a hundred (measured: col_acc 1.0
   at step 100, after which CTC emits correct words within ~150 steps).  The
   renderer's ground-truth boxes make this free — a luxury real corpora do not
   have, which is exactly why the pretrain phase is optional in
   `scripts/train_crnn.py`.

## The reading unit is the grapheme cluster

Tamil is an abugida: `கி` is one visual unit (consonant + vowel sign).  The
vocabulary is therefore 259 Tamil clusters / 477 Brahmi clusters, built in
`data/inventory.py::clusterize`.  CTC predicts clusters; the renderer draws
clusters; the glyph CNN classifies clusters.  One unit everywhere.

## Brahmi → modern text is transliteration + sound laws, not a second OCR

Brahmi inscriptions of Tamil Nadu write early Tamil/Prakrit in the Brahmi
alphabet.  The reader outputs Brahmi clusters; `translation/brahmi_map.py`
re-spells them into Tamil orthography through documented, editable
correspondences (aspiration lost, voiced stops reduced, *s → h*, *-au- →
-ō-*).  Every rule lives in one table; nothing is a black box.

## Recognition ≠ trust: two readers + a lexicon

Each output character carries the agreement of two independent models (CRNN
confidence × glyph-CNN agreement); disagreements are flagged *doubtful* in the
UI instead of being silently emitted.  Whole words then pass a spell-checker
that snaps unknown strings to the nearest epigraphic lexicon entry, re-ranked
by a character-trigram language model — the same trick a human epigraphist
uses when a worn Tholscript glyph leaves two candidate readings.

## What is deliberately *not* here

* no GPU code paths — the models are sized for CPU training (~1 h total);
* no claim of philological authority — the sound laws are conventional and
  auditable;
* no real-scan fine-tuning — but `docs/EXTENDING.md` shows exactly the two
  `Dataset` subclasses a real corpus needs.
