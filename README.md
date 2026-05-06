# target_lesion_positive Prediction

Lesion-level binary classifier for `target_lesion_positive`, with patient-aware
cross-validation and leakage-aware feature sets. See [`PLAN.md`](PLAN.md) for the
full plan.

## Layout

```
data/
  final_paired_analysis_ml.csv         # raw, 2208 lesions x 113 columns
  final_paired_analysis_ml_clean.csv   # produced by ml/clean.py
ml/
  features.py    # 3 leakage-aware feature sets (A/B/C)
  clean.py       # conservative outlier cleaning (cells -> NaN, no row drops)
  eda.py         # quick EDA -> reports/eda_*.csv
  train.py       # GroupKFold(5) x {LogReg, RandomForest, HistGB, [LightGBM]}
reports/         # metrics + EDA + cleaning audit
PLAN.md
requirements.txt
```

## Outlier cleaning (`ml/clean.py`)

Conservative, cell-level cleaning: every offending value is set to `NaN` so
downstream tree models still see the row. No rows are dropped (positives are
scarce). Total cells modified: **85** in **54 / 2208** rows.

| Rule | Cells -> NaN |
|---|---|
| `baseline_creatinine == 0`    | 4  |
| `followup_creatinine == 0`    | 7  |
| `baseline_albumin == 0`       | 2  |
| `followup_albumin == 0`       | 1  |
| `baseline_pdw == 0`           | 5  |
| `followup_pdw == 0`           | 10 |
| `baseline_ck_mb < 0` (`-1` sentinel) | 4 |
| `postop_ck_mb < 0`            | 3  |
| `remodeling_index > 10` (real range 0.8–2.5; max raw 1972) | 21 |
| `max/min/mean/std_hu` all `== 0` (compound HU sentinel)    | 28 |

Intentionally **not** modified:
`nt_pro_bnp == 35000` (assay cap), `min_hu == -190` (HU clip),
`minimum_luminal_area_mm3 == 0` (possible total occlusion), and high-but-real
values for creatinine/triglycerides/CRP.

Audit trail in `reports/cleaning_audit.csv` (one row per modified cell).

## Quick start

```bash
pip install -r requirements.txt
python ml/eda.py
python ml/train.py
```

## Cross-validated baselines (5-fold GroupKFold by `accession_id`)

| Feature set | Best model | AUROC | AUPRC | Sens@Spec=0.9 |
|---|---|---|---|---|
| A. baseline-only (79 cols)   | HistGB | 0.634 ± 0.029 | 0.249 ± 0.052 | 0.206 |
| B. + postop biomarkers (83)  | HistGB | 0.633 ± 0.014 | 0.248 ± 0.036 | 0.178 |
| C. all incl. followup (107)  | HistGB | 0.644 ± 0.015 | 0.266 ± 0.060 | 0.206 |

These are honest, leakage-free baselines. Tuning, LightGBM/XGBoost, and
engineered delta features (`followup_* - baseline_*`) are the natural next steps.
