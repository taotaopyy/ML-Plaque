# target_lesion_positive Prediction

Lesion-level binary classifier for `target_lesion_positive`, with patient-aware
cross-validation and leakage-aware feature sets. See [`PLAN.md`](PLAN.md) for the
full plan.

## Layout

```
data/   final_paired_analysis_ml.csv       # 2208 lesions x 113 columns
ml/
  features.py    # 3 leakage-aware feature sets (A/B/C)
  eda.py         # quick EDA -> reports/eda_*.csv
  train.py       # GroupKFold(5) x {LogReg, RandomForest, HistGB, [LightGBM]}
reports/         # metrics + EDA outputs
PLAN.md
requirements.txt
```

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
