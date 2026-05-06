"""Conservative outlier cleaning.

Rules (each sets the offending cells to NaN; rows are NEVER dropped, since
positives are scarce):

  Physiologically impossible values:
    1. baseline_creatinine == 0          -> NaN
    2. followup_creatinine == 0          -> NaN
    3. baseline_albumin    == 0          -> NaN
    4. followup_albumin    == 0          -> NaN
    5. baseline_pdw        == 0          -> NaN
    6. followup_pdw        == 0          -> NaN
    7. baseline_ck_mb       < 0          -> NaN  (-1 sentinel for below LoD)
    8. postop_ck_mb         < 0          -> NaN

  Extreme statistical outliers (clear computation errors):
    9. remodeling_index    > 10          -> NaN  (max observed ~1972; normal 0.8-2.5)

  Compound sentinel:
   10. rows with max_hu==0 AND min_hu==0 AND mean_hu==0 AND std_hu==0:
       -> set those four HU columns to NaN (contact_volume_mm3 may stay 0).

NOT touched (legitimate values, intentionally preserved):
  - nt_pro_bnp == 35000 (assay upper cap)
  - min_hu == -190       (HU lower clip used for low-attenuation regions)
  - minimum_luminal_area_mm3 == 0  (possible total occlusion)
  - eccentricity_index == 0
  - very high creatinine / triglycerides / CRP (true severe disease)

Outputs:
  data/final_paired_analysis_ml_clean.csv
  reports/cleaning_report.csv  (one row per rule with affected count)
  reports/cleaning_audit.csv   (one row per single (row, column) cell changed)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "final_paired_analysis_ml.csv"
DST = ROOT / "data" / "final_paired_analysis_ml_clean.csv"
REPORT = ROOT / "reports" / "cleaning_report.csv"
AUDIT = ROOT / "reports" / "cleaning_audit.csv"


RULES_SINGLE = [
    ("baseline_creatinine", "== 0",      lambda s: s == 0),
    ("followup_creatinine", "== 0",      lambda s: s == 0),
    ("baseline_albumin",    "== 0",      lambda s: s == 0),
    ("followup_albumin",    "== 0",      lambda s: s == 0),
    ("baseline_pdw",        "== 0",      lambda s: s == 0),
    ("followup_pdw",        "== 0",      lambda s: s == 0),
    ("baseline_ck_mb",      "< 0",       lambda s: s < 0),
    ("postop_ck_mb",        "< 0",       lambda s: s < 0),
    ("remodeling_index",    "> 10",      lambda s: s > 10),
]


def main() -> None:
    df = pd.read_csv(SRC)
    audit_rows: list[dict] = []
    report_rows: list[dict] = []

    for col, rule, fn in RULES_SINGLE:
        if col not in df.columns:
            continue
        mask = fn(df[col]).fillna(False)
        n = int(mask.sum())
        if n:
            for idx in df.index[mask]:
                audit_rows.append({
                    "row_index": int(idx),
                    "accession_id": df.at[idx, "accession_id"],
                    "column": col,
                    "old_value": df.at[idx, col],
                    "rule": rule,
                })
            df.loc[mask, col] = np.nan
        report_rows.append({"rule": f"{col} {rule}", "n_cells_set_to_nan": n})

    hu_cols = ["max_hu", "min_hu", "mean_hu", "std_hu"]
    if all(c in df.columns for c in hu_cols):
        compound_mask = (df["max_hu"] == 0) & (df["min_hu"] == 0) \
                       & (df["mean_hu"] == 0) & (df["std_hu"] == 0)
        n_rows = int(compound_mask.sum())
        for col in hu_cols:
            for idx in df.index[compound_mask]:
                audit_rows.append({
                    "row_index": int(idx),
                    "accession_id": df.at[idx, "accession_id"],
                    "column": col,
                    "old_value": df.at[idx, col],
                    "rule": "compound HU sentinel (all zero)",
                })
            df.loc[compound_mask, col] = np.nan
        report_rows.append({
            "rule": "max/min/mean/std_hu all == 0  (HU sentinel)",
            "n_cells_set_to_nan": n_rows * len(hu_cols),
        })
    else:
        n_rows = 0

    pd.DataFrame(report_rows).to_csv(REPORT, index=False)
    pd.DataFrame(audit_rows).to_csv(AUDIT, index=False)
    df.to_csv(DST, index=False)

    total_cells = sum(r["n_cells_set_to_nan"] for r in report_rows)
    n_rows_touched = pd.DataFrame(audit_rows)["row_index"].nunique() if audit_rows else 0
    print("=== Cleaning summary ===")
    for r in report_rows:
        print(f"  {r['rule']:<55} -> NaN x {r['n_cells_set_to_nan']}")
    print(f"  TOTAL cells modified : {total_cells}")
    print(f"  rows affected (any)  : {n_rows_touched}  / {len(df)}")
    print(f"  rows dropped         : 0")
    print(f"\nWrote: {DST}")
    print(f"Wrote: {REPORT}")
    print(f"Wrote: {AUDIT}")


if __name__ == "__main__":
    main()
