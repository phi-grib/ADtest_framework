"""
Extends ad_ood_analysis.py with AUC values computed from the
compound-level 'test_combined_outliers.csv' files (Raw and OOF flags,
raw continuous AD metrics, and QSAR Prediction_Error per compound).

Two AUC flavours are computed per endpoint x strategy:
1. 'AUC_flagcount': ROC-AUC of 'Total Metrics Flagged_<raw/oof>' (an
   integer 0-9 severity score) against 'Prediction_Error'. This answers
   "does the number of AD metrics violated rank compounds by how likely
   they are to be mispredicted?" - the same idea as RR/EF, but as a
   threshold-free ranking metric.
2. Per-metric AUC ('AUC_<metric>'): ROC-AUC of each individual continuous
   AD metric (DModX, T2, LOF_Score, ...) against 'Prediction_Error'.
   Metric directionality (e.g. LOF_Score is 'low = outlier') is handled
   by sign-flipping before roc_auc_score, consistent with
   analysis.calculate_metric_power().

Author: analysis helper for Eric March's AD/QSAR Out-of-domain study.
"""

import os
import pandas as pd

from pathlib import Path
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = str(REPO_ROOT / "results" / "final_analysis" / "outlier_tables")
OUT_DIR = str(REPO_ROOT / "results" / "final_analysis" / "outlier_tables")

# Standardized endpoint formatting map (all keys lowercase for case-insensitive lookup)
ENDPOINT_MAP = {
    "ache": "AChE",
    "ames": "AMES",
    "ar": "AR",
    "carcinogen": "Carcinogen",
    "carcinogenicity": "Carcinogen",
    "dat": "DAT",
    "dili": "DILI",
    "er_alpha": "ER alpha",
    "er alpha": "ER alpha",
    "er_beta": "ER beta",
    "er beta": "ER beta",
    "gr": "GR",
    "hdac": "HDAC",
    "herg": "hERG",
    "net": "NET",
    "sert": "SERT",
    "skin_reaction": "Skin Reaction",
    "skin reaction": "Skin Reaction",
    "skin_sensitization": "Skin Reaction",
    "tr_alpha": "TR alpha",
    "tr alpha": "TR alpha",
}

# Endpoints under chemical domain shift
OOD_ENDPOINTS = {"ACHE", "AChE", "DAT", "HDAC", "NET", "SERT"}

# Metrics where lower values indicate higher outlier likelihood / risk
LOW_IS_OUTLIER = {
    "LOF_Score", "LOF",
    "Di_Similarity_Density", "Similarity_Density",
    "KDE_LL", "KDE",
    "IForest_Score", "IForest"
}

# Continuous metrics list (handles spelling variants)
CONTINUOUS_METRICS = [
    "DModX", "T2", "LOF_Score", "kNN_Dist", "KDE_LL",
    "IForest_Score", "AE_Recon_Error", 
    "Di_Similarity_Density", "Ii_Activity_Inconstency", "Ii_Activity_Inconsistency"
]

# Metric clean name mapping for output columns
METRIC_RENAME_MAP = {
    "Ii_Activity_Inconstency": "Ii_Activity_Inconsistency",
    "Di_Similarity_Density": "Di_Similarity_Density"
}

def parse_endpoint_from_filename(fname: str) -> tuple[str, str]:
    stem = fname.split("_CV_")[0].split("_test_")[0].strip()
    endpoint = ENDPOINT_MAP.get(stem.lower(), stem)
    domain = "Out-of-domain" if endpoint in OOD_ENDPOINTS else "In-domain"
    return endpoint, domain

def endpoint_name(filename: str) -> str:
    """Extracts and formats the endpoint name from the standard filename pattern."""
    raw_ep = filename.split("_")[0]
    return ENDPOINT_MAP.get(raw_ep, raw_ep)


def safe_auc(y_true, y_score) -> float:
    """Wraps roc_auc_score to safely handle single-class inputs and NaN values."""
    mask = y_true.notna() & y_score.notna()
    y_true, y_score = y_true[mask], y_score[mask]
    if y_true.nunique() < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def compute_auc_table() -> pd.DataFrame:
    """Computes AUCs for flag-counts and raw continuous AD metrics."""
    rows = []
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith("_test_combined_outliers.csv")]

    for fname in sorted(files):
        ep, domain = parse_endpoint_from_filename(fname)
        # ep = endpoint_name(fname)
        # domain = "Domain shift" if ep in OOD_ENDPOINTS else "In-domain"
        df = pd.read_csv(os.path.join(UPLOAD_DIR, fname))

        if "Prediction_Error" not in df.columns:
            continue

        y_err = df["Prediction_Error"]

        for strategy, suffix in [("Raw", "_raw"), ("OOF", "_oof")]:
            row = {"Endpoint": ep, "Domain": domain, "Strategy": strategy}

            # 1. Overall flag-count AUC
            flag_col = f"Total Metrics Flagged{suffix}"
            if flag_col in df.columns:
                row["AUC_flagcount"] = safe_auc(y_err, df[flag_col])

            # 2. Per-metric AUC (continuous raw values)
            if strategy == "Raw":
                for metric in CONTINUOUS_METRICS:
                    if metric not in df.columns:
                        continue
                    
                    score = -df[metric] if metric in LOW_IS_OUTLIER else df[metric]
                    clean_metric_name = METRIC_RENAME_MAP.get(metric, metric)
                    row[f"AUC_{clean_metric_name}"] = safe_auc(y_err, score)

            rows.append(row)

    auc_df = pd.DataFrame(rows)

    # Forward/backward-fill continuous metric AUCs across strategies for the same endpoint
    metric_cols = [c for c in auc_df.columns if c.startswith("AUC_") and c != "AUC_flagcount"]
    if metric_cols and not auc_df.empty:
        auc_df[metric_cols] = auc_df.groupby("Endpoint")[metric_cols].transform(lambda s: s.ffill().bfill())

    return auc_df


if __name__ == "__main__":
    auc_df = compute_auc_table()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ad_auc_table.csv")
    auc_df.to_csv(out_path, index=False)
    print(auc_df.round(3).to_string(index=False))

    # Merges with the output generated by ad_ood_analysis.py
    prev_path = os.path.join(OUT_DIR, "ad_ood_comparison.csv")
    if os.path.exists(prev_path):
        prev_df = pd.read_csv(prev_path)
        
        # Ensure Endpoint names in prev_df match ENDPOINT_MAP
        if "Endpoint" in prev_df.columns:
            prev_df["Endpoint"] = prev_df["Endpoint"].map(
                lambda x: ENDPOINT_MAP.get(str(x).strip().lower(), str(x).strip())
            )

        merged = prev_df.merge(auc_df, on=["Endpoint", "Domain", "Strategy"], how="left")
        merged_path = os.path.join(OUT_DIR, "ad_full_comparison_with_auc.csv")
        merged.to_csv(merged_path, index=False)
        print(f"\nMerged table saved to {merged_path}")