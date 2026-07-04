# Drift analysis — popularity shift in the FMA Hip-Hop detector

The detector is trained on the more-played half of each genre (the **reference**
window) and monitored against the less-played half (the **current** window). The
split is taken at the median *within each genre*, so both windows hold the same
genre mix and the same ~12.5% Hip-Hop base rate. Any drift here is therefore a
clean **covariate shift** — do popular and obscure tracks differ in their audio
features — and not a manufactured change in class balance.

The numbers below come from `python src/monitor_drift.py` (Evidently over all 150
features) and the reference-vs-current transfer measurement in `src/evaluate.py`.

## 1. Is the data drifting?

Marginally. Evidently flags **3 of 150 features** as drifted — a drift share of
**0.02**, far below the 0.5 gate, so the monitor exits clean:

| feature | drift score | threshold | RF importance rank |
| --- | --- | --- | --- |
| `mfcc_mean_06` | 0.120 | 0.10 | 37 / 150 |
| `license_family` | 0.107 | 0.10 | 80 / 150 |
| `chroma_stft_std_06` | 0.104 | 0.10 | 81 / 150 |

All three sit just over the 0.10 distance threshold; none is a large shift. The
overwhelming majority of the spectral profile — 147 of 150 features — is
statistically indistinguishable between popular and obscure tracks.

## 2. What is the likely cause?

Two plausible, production-grounded stories, one per feature type:

- **Spectral features (`mfcc_mean_06`, `chroma_stft_std_06`).** These are timbral
  and tonal summaries. A mild shift between popular and obscure tracks is
  consistent with **production and mastering budget**: better-resourced (and
  therefore, on average, more-played) releases tend to be louder, more compressed,
  and more consistently mastered, which nudges the MFCC and spectral-spread
  moments. The effect is real but small here because FMA is uniformly
  Creative-Commons indie material — there is no major-label mastering tier to open
  a large gap.

- **`license_family`.** The CC license mix differs slightly between the tiers.
  This is a metadata/behavioral correlate of popularity (which license an artist
  picks correlates loosely with how they distribute and promote), not an audio
  property, and it is weakly informative for genre.

## 3. Would the drift affect model performance?

No — and this is the question that matters, answered with a **measured** number
rather than the input-drift flag alone.

**Importance-weighted view.** Treating all drifted features equally would overstate
the risk. Weighting the alert by how much the model actually relies on each feature
collapses it: the three drifted features together hold only **1.6% of the
RandomForest's total importance mass**, and the model's top-10 features
(`rmse_std_01`, `mfcc_std_04`, `mfcc_std_06`, …) show **no drift at all**. The
drift is concentrated exactly where the model is not looking.

**Transfer (the payoff).** Training on reference and scoring once on the held-out
current window:

| metric | reference test | current (held out) |
| --- | --- | --- |
| positive-class F1 | 0.543 | 0.595 |

F1 does not degrade on the obscure tier the model never trained on — it is
**+0.05 higher**. Performance transfers. So the honest conclusion is that popular
and obscure CC indie tracks are near-identical in the spectral features that carry
genre, and the detector generalizes across the popularity axis.

## What this would mean in production

The monitoring recommendation is to **weight drift alerts by feature importance**
rather than alert on raw drifted-feature count. A future shift in `rmse_std_01` or
the `mfcc_std_*` block — the features the model leans on — would be worth a
retrain trigger; a shift in a rank-80 feature like `license_family` is noise. The
gate is set on the overall drift share for a coarse tripwire, but the actionable
signal is importance-weighted, and the transfer metric is the ground truth for
whether any observed drift actually costs accuracy.
