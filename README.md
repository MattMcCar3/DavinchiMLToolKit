# DavinchiMLToolKit

Two things live in this repo:

1. **`davinchi.ml`** — a from-scratch k-NN toolkit (normalization, 5x2 cross-validation, mixed-type distance metrics, k-NN classification/regression, condensed k-NN), originally built for JHU 605.649 coursework. See `REPORT.md` for the full writeup, including a bug that was found and fixed in the condensed k-NN implementation.
2. **`davinchi.video`** + **`davinchi.resolve`** — a footage-processing pipeline for the "Written in Paint" content channel: detects hard cuts and dead air, classifies painting stages from frame color, unifies all three into one timeline, and pushes that timeline into DaVinci Resolve as markers via its Python scripting API.

The `ml` package isn't just kept around for sentiment — the stage classifier in `video/classification/stage_classifier.py` directly reuses `davinchi.ml.models.knn`.

## Install

```bash
pip install -e ".[dev]"
```

## Run the tests

```bash
pytest tests/
```

## Try the video pipeline on synthetic footage

No real footage needed to try this out — generate a synthetic test clip with known ground truth (colored segments as stand-in "stages", hard cuts, and alternating tone/silence audio):

```bash
python scripts/generate_synthetic_test_video.py
python scripts/process_footage.py data/samples/synthetic_test.mp4
python scripts/demo_stage_classifier.py
```

`process_footage.py` detects cuts and silence and writes a `.timeline.json` file. `demo_stage_classifier.py` shows the k-NN stage classifier correctly identifying which "stage" (color segment) a frame belongs to.

## Status of the Resolve integration

`davinchi.resolve` follows Blackmagic's documented scripting API but is **untested against a real Resolve installation** — there's no Resolve app available in the environment this was built in. Before trusting it on real footage:

- Confirm Resolve's scripting modules path on your machine (see the docstring in `resolve/client.py` for likely locations per OS).
- Enable External Scripting in Resolve's Preferences (General > External scripting using = Local).
- Start small: push markers onto a test timeline and confirm they land where expected before trying it on anything real.

The Resolve integration currently adds **markers** (cut points, silence spans, stage labels) to an already-open timeline — it doesn't perform automatic cuts. That's deliberate: markers are reviewable and non-destructive, which matters most on the first real footage this touches.

## Next steps

- Film real test footage and re-tune `detect_cuts`' `threshold` and `detect_silence`'s `rms_threshold` against it — the synthetic test video's hard color/tone transitions are easy mode compared to real painting footage.
- Build a real reference set of labeled stage frames for `StageClassifier` (the synthetic demo uses solid colors as a stand-in).
- Verify the Resolve integration against an actual Resolve install and real project.
