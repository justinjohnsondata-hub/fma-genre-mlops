# Model card — FMA Hip-Hop Detector

## What it is

A binary classifier that predicts whether a music track is **Hip-Hop** from
precomputed audio (DSP) features. Positive class = Hip-Hop; negative = any of the
other seven `fma_small` genres. It is a one-vs-rest *detector*, not a full genre
classifier and not a quality or popularity judgment.

- **Model:** RandomForest (600 trees, max depth 24, balanced class weights),
  selected by positive-class F1 from a five-model sweep.
- **Inputs:** 148 librosa summary features (MFCC, chroma, spectral contrast, RMS,
  etc.; mean and std moments) plus two categoricals — license family and release
  era. Numeric features are median-imputed and standardized; categoricals are
  one-hot encoded. The preprocessor is fit on training data only.
- **Output:** a binary label (and a positive-class probability).

## Performance

Measured on a stratified held-out split of the reference population:

| metric | value |
| --- | --- |
| positive-class F1 | 0.54 |
| ROC-AUC | 0.90 |
| accuracy | 0.90 |
| majority-baseline F1 | 0.00 |

Accuracy is reported only to make the imbalance explicit: a model that always
predicts "not Hip-Hop" scores ~88% accuracy and F1 0.0, which is why F1 on the
positive class is the headline metric.

## Intended use and scope

Intended as a catalog-tagging aid: flag likely Hip-Hop tracks in an untagged
library for review. It is a portfolio/educational artifact demonstrating an MLOps
pipeline, not a production service.

**Honest limitations:**

- **Creative-Commons indie, not mainstream.** FMA's Hip-Hop is Creative-Commons
  indie hip-hop. The model learns *that* distribution; it should not be assumed to
  generalize to mainstream/commercial hip-hop, which differs in production and
  instrumentation.
- **A DSP-features ceiling.** The model sees only precomputed spectral/timbral
  summaries, never the audio or lyrics. Genre cues outside those features (vocal
  delivery, sampling, lyrical content) are invisible to it, which caps achievable
  accuracy — the mid-range F1 is expected, not a bug.
- **The monitored drift is a within-genre popularity shift.** Drift is measured
  between the more-played and less-played halves of each genre. It is a
  **covariate shift** (do popular vs obscure tracks differ spectrally), **not** a
  temporal shift and **not** a change in the Hip-Hop base rate — the split holds
  the genre mix and class balance constant. Measured F1 transfers to the obscure
  tier (see [drift_analysis.md](drift_analysis.md)).

## Data and attribution

Trained on the `fma_small` subset of the Free Music Archive dataset (Defferrard et
al., *"FMA: A Dataset for Music Analysis"*, ISMIR 2017). Metadata licensed
**CC BY 4.0**. See [README.md](README.md) for the full citation and license notes.
