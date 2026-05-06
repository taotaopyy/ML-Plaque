"""Train and evaluate models with patient-aware GroupKFold cross-validation.

Runs three feature sets (A/B/C) x several models, writes per-fold and
aggregated metrics to reports/.

Default models use only sklearn so the script works without extra deps.
LightGBM is used automatically if installed.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (
    CATEGORICAL_COLS,
    TARGET_COL,
    build_feature_sets,
    get_target_and_groups,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "final_paired_analysis_ml.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42


@dataclass
class ModelSpec:
    name: str
    builder: Callable[[list[str], list[str]], Pipeline]
    handles_nan: bool


def _logreg(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=2000,
                                                class_weight="balanced",
                                                n_jobs=None))])


def _rf(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", RandomForestClassifier(n_estimators=400,
                                                    class_weight="balanced",
                                                    random_state=RANDOM_STATE,
                                                    n_jobs=-1))])


def _hgb(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"
                                               , sparse_output=False))]), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", HistGradientBoostingClassifier(
                         max_iter=400, learning_rate=0.05,
                         max_depth=None, l2_regularization=1.0,
                         class_weight="balanced",
                         random_state=RANDOM_STATE))])


MODELS: list[ModelSpec] = [
    ModelSpec("logreg", _logreg, handles_nan=False),
    ModelSpec("random_forest", _rf, handles_nan=False),
    ModelSpec("hist_gradient_boosting", _hgb, handles_nan=True),
]

try:
    from lightgbm import LGBMClassifier

    def _lgbm(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
        pre = ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), cat_cols),
        ])
        clf = LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    MODELS.append(ModelSpec("lightgbm", _lgbm, handles_nan=True))
except ImportError:
    pass


def sens_at_spec(y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.9) -> float:
    order = np.argsort(-y_score)
    y_true = np.asarray(y_true)[order]
    y_score = np.asarray(y_score)[order]
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    spec = 1 - fp / neg
    sens = tp / pos
    mask = spec >= target_spec
    return float(sens[mask].max()) if mask.any() else float("nan")


def evaluate_fold(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "sens_at_spec90": sens_at_spec(y_true, y_prob, 0.9),
        "f1_at_0.5": f1_score(y_true, (y_prob >= 0.5).astype(int)),
    }


def split_columns(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    cat = [c for c in feature_cols if c in CATEGORICAL_COLS]
    num = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    return num, cat


def run() -> None:
    df = pd.read_csv(DATA)
    y, groups = get_target_and_groups(df)
    feature_sets = build_feature_sets(df)

    cv = GroupKFold(n_splits=N_SPLITS)
    fold_rows: list[dict] = []

    for fs_name, fs in feature_sets.items():
        num_cols, cat_cols = split_columns(df, fs.columns)
        X = df[num_cols + cat_cols]
        for spec in MODELS:
            print(f"[{fs_name}] training {spec.name} on {len(fs.columns)} features ...")
            for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups)):
                pipe = spec.builder(num_cols, cat_cols)
                pipe.fit(X.iloc[tr], y.iloc[tr])
                prob = pipe.predict_proba(X.iloc[va])[:, 1]
                metrics = evaluate_fold(y.iloc[va].values, prob)
                metrics.update({"fold": fold, "model": spec.name,
                                "feature_set": fs_name,
                                "n_train": len(tr), "n_val": len(va)})
                fold_rows.append(metrics)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "metrics_folds.csv", index=False)

    metric_cols = ["auroc", "auprc", "brier", "sens_at_spec90", "f1_at_0.5"]
    agg = (folds.groupby(["feature_set", "model"])[metric_cols]
                .agg(["mean", "std"]).round(4))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index().sort_values(
        ["feature_set", "auprc_mean"], ascending=[True, False]
    )
    agg.to_csv(OUT / "metrics_summary.csv", index=False)

    print("\n=== Cross-validated metrics (mean ± std over folds) ===")
    for _, r in agg.iterrows():
        print(f"  [{r['feature_set']:>22}] {r['model']:>22} | "
              f"AUROC={r['auroc_mean']:.3f}±{r['auroc_std']:.3f}  "
              f"AUPRC={r['auprc_mean']:.3f}±{r['auprc_std']:.3f}  "
              f"Sens@Spec=0.9={r['sens_at_spec90_mean']:.3f}")

    config = {
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "n_rows": int(len(df)),
        "n_patients": int(groups.nunique()),
        "pos_rate": float(y.mean()),
        "models": [m.name for m in MODELS],
        "feature_sets": {n: len(fs.columns) for n, fs in feature_sets.items()},
    }
    with (OUT / "run_config.json").open("w") as f:
        json.dump(config, f, indent=2)
    print(f"\nWrote: {OUT/'metrics_folds.csv'}, {OUT/'metrics_summary.csv'}")


if __name__ == "__main__":
    run()
