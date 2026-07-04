# FMA Hip-Hop Detector — an MLOps pipeline

**Problem.** Given audio tracks with no reliable genre tags, decide from the sound
alone whether a track is Hip-Hop. This is a real catalog problem: sample libraries
and production-music catalogs hold large numbers of untagged files. The task is
framed as binary one-vs-rest detection — `y = (genre_top == "Hip-Hop")` — on the
balanced 8-genre `fma_small` subset (8,000 tracks, ~12.5% Hip-Hop), so the class
imbalance is real.

**Approach.** Features are precomputed librosa DSP summaries (MFCCs, spectral
contrast, chroma, RMS energy — 148 mean/std columns) plus two engineered
categoricals (license family, release era), with seeded missing values injected to
exercise imputation. Preprocessing (impute → scale numeric, one-hot categoricals)
is fit inside the training fold only. Five runs are compared — logistic regression,
gradient boosting, and three RandomForest settings — ranked by positive-class F1.

**Result.** The winning RandomForest reaches **F1 0.54** on the Hip-Hop class
(ROC-AUC 0.90) against a majority-baseline F1 of **0.0**, the deliberate
demonstration that accuracy (~88% for a model that never predicts Hip-Hop) is the
wrong metric under imbalance. The model is honestly mid-range: good enough that
monitoring it is meaningful. Data drift between popular and obscure tracks is a mild
covariate shift (3 of 150 features), and measured performance transfers to the
unseen tier — see [drift_analysis.md](drift_analysis.md) and
[MODEL_CARD.md](MODEL_CARD.md).

## Pipeline

```
download_data.py → prep_data.py → [DVC-tracked prepared data]
                                        │
                     train.py ──────────┤ (preprocess → fit → evaluate → MLflow)
                     run_sweep.py ───────┘
                     compare_experiments.py → winner
                     monitor_drift.py → Evidently report
```

Where each piece of the stack lives:

| Concern | Location |
| --- | --- |
| **Data versioning (DVC)** | `data/prepared.dvc` tracks the prepared reference/current parquet; pointers committed, blobs in the DVC cache |
| **Experiment tracking (MLflow)** | `src/train.py` logs params, data version, metrics, and the model; `src/compare_experiments.py` ranks runs by F1 |
| **Testing (pytest)** | `tests/unit/`, `tests/data_validation/`, `tests/model_validation/` |
| **CI (GitHub Actions)** | `.github/workflows/ci.yml` — test job, then a train job gated on positive-class F1 |
| **Drift monitoring (Evidently)** | `src/monitor_drift.py` + `drift_analysis.md`, report written to `reports/` |

## Running it

```bash
# 1. Environment (Python 3.13)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Data — download + verify (SHA1) + extract, then build the modeling frames
python src/download_data.py
python src/prep_data.py

# 3. Train one model (reads configs/train.yaml; exits non-zero if F1 < threshold)
python src/train.py

# 4. Run the five-model sweep and rank by F1
python src/run_sweep.py
python src/compare_experiments.py

# 5. Tests
pytest tests/ -v

# 6. Drift check (writes reports/drift_report.html)
python src/monitor_drift.py

# View the tracked runs
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The prepared data is DVC-tracked for provenance. This repo regenerates it from the
pinned source (`download_data.py` + `prep_data.py`), which is also what CI does; a
production deployment would instead `dvc pull` the prepared data from a shared
remote.

## Data and license

Free Music Archive (FMA), Defferrard, Benzi, Vandergheynst, and Bresson,
*"FMA: A Dataset for Music Analysis"*, ISMIR 2017. The `fma_metadata.zip` archive
(metadata + precomputed features, no audio) is fetched from the SWITCH object store
and verified against its published SHA1 before use. FMA metadata is licensed
**CC BY 4.0**; the code in this repository is **MIT**. `src/utils.py` is FMA's own
MIT-licensed loader, vendored to read the multi-level CSV headers.
