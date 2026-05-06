"""Feature set definitions for the lesion-level ML task.

Three feature sets are defined for leakage-aware modeling:
- A (baseline-only):    pre-procedure information only -> prospective prediction
- B (A + postop):       adds early post-procedure biomarkers
- C (A + B + followup): includes follow-up period; for retrospective association only
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


ID_COLS = ["accession_id", "cta_admission_number"]
TARGET_COL = "target_lesion_positive"

DROP_COLS = [
    "myocardial_bridge_level",
    "current_intervention_flag",
    "prior_revascularization_flag",
]

CATEGORICAL_COLS = [
    "artery_vessel",
    "lesion_segment",
    "contact_segment",
    "plaque_type",
    "lesion_stenosis_grade",
]


def _split_columns_by_prefix(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = [c for c in df.columns
            if c not in ID_COLS + [TARGET_COL] + DROP_COLS]

    postop = [c for c in cols if c.startswith("postop_")]
    followup = [c for c in cols if c.startswith("followup_")]
    baseline = [c for c in cols if c not in postop + followup]
    return {"baseline": baseline, "postop": postop, "followup": followup}


@dataclass
class FeatureSet:
    name: str
    columns: list[str]
    description: str


def build_feature_sets(df: pd.DataFrame) -> dict[str, FeatureSet]:
    parts = _split_columns_by_prefix(df)
    return {
        "A_baseline": FeatureSet(
            name="A_baseline",
            columns=parts["baseline"],
            description="Demographics + history + baseline labs + CTA morphology/plaque/FFRct",
        ),
        "B_baseline_plus_postop": FeatureSet(
            name="B_baseline_plus_postop",
            columns=parts["baseline"] + parts["postop"],
            description="A + postop biomarkers",
        ),
        "C_all": FeatureSet(
            name="C_all",
            columns=parts["baseline"] + parts["postop"] + parts["followup"],
            description="A + B + followup labs/echo (retrospective only)",
        ),
    }


def get_target_and_groups(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return df[TARGET_COL].astype(int), df["accession_id"]
