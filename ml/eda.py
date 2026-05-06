"""Quick EDA: summarize the dataset and dump CSV reports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from features import (
    CATEGORICAL_COLS,
    DROP_COLS,
    ID_COLS,
    TARGET_COL,
    build_feature_sets,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "final_paired_analysis_ml.csv"
OUT = Path(__file__).resolve().parent.parent / "reports"
OUT.mkdir(exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA)
    print(f"shape: {df.shape}")
    print(f"target balance: {df[TARGET_COL].value_counts().to_dict()} "
          f"(pos rate={df[TARGET_COL].mean():.3f})")
    print(f"unique patients: {df['accession_id'].nunique()}")

    overview = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": df.isna().mean() * 100,
        "n_unique": df.nunique(dropna=True),
    }).sort_values("pct_missing", ascending=False)
    overview.to_csv(OUT / "eda_columns.csv", index_label="column")

    cat_summary = []
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False)
            cat_summary.append(
                pd.DataFrame({"column": c, "value": vc.index.astype(str), "count": vc.values})
            )
    if cat_summary:
        pd.concat(cat_summary, ignore_index=True).to_csv(
            OUT / "eda_categorical.csv", index=False
        )

    fs = build_feature_sets(df)
    rows = []
    for name, s in fs.items():
        rows.append({"feature_set": name,
                     "n_features": len(s.columns),
                     "description": s.description})
    pd.DataFrame(rows).to_csv(OUT / "eda_feature_sets.csv", index=False)
    for name, s in fs.items():
        pd.Series(s.columns, name="feature").to_csv(
            OUT / f"eda_features_{name}.csv", index=False
        )

    print("\n== feature set sizes ==")
    for name, s in fs.items():
        print(f"  {name}: {len(s.columns)} features  -- {s.description}")

    print("\n== dropped (constant / all-NaN) ==")
    for c in DROP_COLS:
        print(f"  {c}")
    print("\n== id columns (excluded from features) ==")
    for c in ID_COLS:
        print(f"  {c}")

    print(f"\nReports written to: {OUT}")


if __name__ == "__main__":
    main()
