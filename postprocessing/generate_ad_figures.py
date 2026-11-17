"""
Builds the Out-of-domain vs. In-domain AD comparison table (RR, EF, %Outside,
AUC, F1) from:
1. The per-endpoint YAML reports (aggregated Raw/OOF outlier counts), and
2. The per-compound 'test_combined_outliers.csv' files (needed for AUC which require row-level Prediction_Error + flag data).

Produces:
- ad_full_comparison_with_auc.csv   (one row per endpoint x strategy)
- ad_group_summary_final.csv               (means by Domain x Strategy)
- fig1_group_summary_in_domain.png         (In-domain 3-panel figure, Panel A)
- fig1_group_summary_ood.png               (Out-of-domain 3-panel figure, Panel B)
- fig2_per_metric_AUC_by_domain.png        (Horizontal barplot of individual AD metrics)

Author: analysis helper for Eric March's AD/QSAR Out-of-domain study.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# ---------------------------------------------------------------------
# CONFIGURATION & GLOBAL MAPS
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
YAML_DIR = str(REPO_ROOT / "results" / "final_analysis" / "yaml_dir")
CSV_DIR = str(REPO_ROOT / "results" / "final_analysis" / "outlier_tables")
OUT_DIR = str(REPO_ROOT / "results" / "final_analysis" / "figures")

os.makedirs(OUT_DIR, exist_ok=True)

# Endpoint Formatting Dictionary
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

DOMAIN_SHIFT_ENDPOINTS = {"ACHE", "AChE", "DAT", "HDAC", "NET", "SERT"}

LOW_IS_OUTLIER = {
    "LOF_Score", "LOF", 
    "Di_Similarity_Density", "Similarity_Density", 
    "KDE_LL", "KDE", 
    "IForest_Score", "IForest"
}

CONTINUOUS_METRICS = [
    "DModX", "T2", "LOF_Score", "kNN_Dist", "KDE_LL",
    "IForest_Score", "AE_Recon_Error", "Di_Similarity_Density",
    "Ii_Activity_Inconstency", "Ii_Activity_Inconsistency"
]

METRIC_MAP = {
    "Ii_Activity_Inconstency": r"Activity Inconsistency ($I_i$)",
    "Ii_Activity_Inconsistency": r"Activity Inconsistency ($I_i$)",
    "Activity_Inconsistency": r"Activity Inconsistency ($I_i$)",
    "Ii": r"Activity Inconsistency ($I_i$)",
    "Di_Similarity_Density": r"Similarity Density ($D_i$)",
    "Similarity_Density": r"Similarity Density ($D_i$)",
    "Di": r"Similarity Density ($D_i$)",
    "LOF_Score": "LOF",
    "LOF": "LOF",
    "AE_Recon_Error": "AE Recon. Error",
    "AE": "AE Recon. Error",
    "KDE_LL": "KDE",
    "KDE": "KDE",
    "IForest_Score": "IForest",
    "IForest": "IForest",
    "DModX": "DModX",
    "T2": "T2",
    "kNN_Dist": "kNN"
}

DOMAIN_COLORS = {"In-domain": "#2ca02c", "Out-of-domain": "#d62728"}

def normalize_endpoint(ep: str) -> str:
    key = str(ep).strip().lower().replace(" ", "_")
    return ENDPOINT_MAP.get(key, ep)

def endpoint_name(filename: str) -> str:
    """Extracts and formats the endpoint name from the standard filename pattern."""
    raw_ep = filename.split("_CV_")[0].split("_test_")[0].strip()
    return ENDPOINT_MAP.get(raw_ep.lower(), raw_ep)

# =======================================================================
# STEP 1 - Aggregated counts from YAML (RR, EF, %Outside)
# =======================================================================

def compute_rr_ef(total, inside, outside, err_in_pct, err_out_pct) -> dict:
    """
    Recomputes Relative Risk (RR) and Enrichment Factor (EF) from the
    aggregated AD outlier summary counts stored in the YAML reports.
    """
    errors_in = inside * err_in_pct / 100.0
    errors_out = outside * err_out_pct / 100.0
    baseline_pct = (errors_in + errors_out) / total * 100.0 if total > 0 else np.nan

    rr = err_out_pct / err_in_pct if err_in_pct > 0 else np.nan
    ef_in = err_in_pct / baseline_pct if baseline_pct > 0 else np.nan
    ef_out = err_out_pct / baseline_pct if baseline_pct > 0 else np.nan

    return {"RR": rr, "EF_inside": ef_in, "EF_outside": ef_out,
            "baseline_error_pct": baseline_pct}


def build_rr_ef_table() -> pd.DataFrame:
    """
    Parses every endpoint YAML report and returns a long-format table with
    one row per (Endpoint, Strategy), containing %Outside, %ErrIn, %ErrOut,
    RR and EF.
    """
    rows = []
    strategy_keys = {"Raw": "Raw (Training Boundaries)", "OOF": "OOF (CV Boundaries)"}

    if not os.path.exists(YAML_DIR):
        print(f"[Warning] YAML_DIR not found: {YAML_DIR}")
        return pd.DataFrame()

    for fname in sorted(os.listdir(YAML_DIR)):
        if not fname.endswith(".yaml"):
            continue
        ep = endpoint_name(fname)
        domain = "Out-of-domain" if ep in DOMAIN_SHIFT_ENDPOINTS else "In-domain"

        with open(os.path.join(YAML_DIR, fname)) as fh:
            report = yaml.safe_load(fh)

        summary = report.get("external_test_set_performance", {}).get("test_outlier_summary", {})
        for strategy, key in strategy_keys.items():
            block = summary.get(key)
            if block is None:
                continue
            total, inside, outside = block["Total Test Set"], block["Inside AD"], block["Outside AD"]
            err_in, err_out = block.get("% Errors Inside AD"), block.get("% Errors Outside AD")
            metrics = compute_rr_ef(total, inside, outside, err_in, err_out)
            rows.append({
                "Endpoint": ep, "Domain": domain, "Strategy": strategy,
                "Total": total, "Inside": inside, "Outside": outside,
                "%Outside": block["% Outside"], "%ErrIn": err_in, "%ErrOut": err_out,
                **metrics,
            })
    return pd.DataFrame(rows)


# =======================================================================
# STEP 2 - Compound-level CSVs (AUC)
# =======================================================================

def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """ROC-AUC that returns NaN when y_true has a single class."""
    mask = y_true.notna() & y_score.notna()
    y_true, y_score = y_true[mask], y_score[mask]
    if y_true.nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def build_auc_table() -> pd.DataFrame:
    """Iterates over compound-level CSVs and computes AUC."""
    rows = []
    if not os.path.exists(CSV_DIR):
        print(f"[Warning] CSV_DIR not found: {CSV_DIR}")
        return pd.DataFrame()

    files = [f for f in os.listdir(CSV_DIR) if f.endswith("_test_combined_outliers.csv")]

    for fname in sorted(files):
        ep = endpoint_name(fname)
        domain = "Out-of-domain" if ep in DOMAIN_SHIFT_ENDPOINTS else "In-domain"
        df = pd.read_csv(os.path.join(CSV_DIR, fname))
        
        if "Prediction_Error" not in df.columns:
            continue
            
        y_err = df["Prediction_Error"].fillna(0).astype(int)

        for strategy, suffix in [("Raw", "_raw"), ("OOF", "_oof")]:
            row = {"Endpoint": ep, "Domain": domain, "Strategy": strategy}

            flag_col = f"Total Metrics Flagged{suffix}"
            if flag_col in df.columns:
                row["AUC_flagcount"] = safe_auc(y_err, df[flag_col])

            if strategy == "Raw":
                for metric in CONTINUOUS_METRICS:
                    if metric not in df.columns:
                        continue
                    score = -df[metric] if metric in LOW_IS_OUTLIER else df[metric]
                    row[f"AUC_{metric}"] = safe_auc(y_err, score)

            rows.append(row)

    auc_df = pd.DataFrame(rows)
    if not auc_df.empty:
        metric_cols = [c for c in auc_df.columns if c.startswith("AUC_") and c != "AUC_flagcount"]
        if metric_cols:
            auc_df[metric_cols] = auc_df.groupby("Endpoint")[metric_cols].transform(lambda s: s.ffill().bfill())
    return auc_df


# =======================================================================
# STEP 3 - Merge, Summarise, and Plot
# =======================================================================

def main():
    rr_ef_df = build_rr_ef_table()
    auc_df = build_auc_table()

    if rr_ef_df.empty and not auc_df.empty:
        full = auc_df
    elif not rr_ef_df.empty and auc_df.empty:
        full = rr_ef_df
    elif not rr_ef_df.empty and not auc_df.empty:
        full = rr_ef_df.merge(auc_df, on=["Endpoint", "Domain", "Strategy"], how="left")
    else:
        print("[Error] No data loaded from YAML_DIR or CSV_DIR.")
        return

    full.to_csv(os.path.join(OUT_DIR, "ad_full_comparison_with_auc.csv"), index=False)

    group_summary = full.groupby(["Domain", "Strategy"]).agg(
        pctOut_mean=("%Outside", "mean") if "%Outside" in full else ("Total", "count"),
        pctOut_sem=("%Outside", lambda x: x.std(ddof=1) / len(x) ** 0.5) if "%Outside" in full else ("Total", "count"),
        RR_mean=("RR", "mean") if "RR" in full else ("Strategy", "count"),
        RR_sem=("RR", lambda x: x.std(ddof=1) / len(x) ** 0.5) if "RR" in full else ("Strategy", "count"),
        EF_mean=("EF_outside", "mean") if "EF_outside" in full else ("Strategy", "count"),
        EF_sem=("EF_outside", lambda x: x.std(ddof=1) / len(x) ** 0.5) if "EF_outside" in full else ("Strategy", "count"),
        AUC_mean=("AUC_flagcount", "mean") if "AUC_flagcount" in full else ("Strategy", "count"),
        AUC_sem=("AUC_flagcount", lambda x: x.std(ddof=1) / len(x) ** 0.5) if "AUC_flagcount" in full else ("Strategy", "count"),
    ).reset_index()
    
    group_summary.to_csv(os.path.join(OUT_DIR, "ad_group_summary_final.csv"), index=False)

    # 1. Group summary bars (In-domain Panel A and Out-of-domain Panel B)
    plot_group_summary_bars(
        group_summary, 
        os.path.join(OUT_DIR, "fig2A_group_summary_in_domain.png"), 
        os.path.join(OUT_DIR, "fig2B_group_summary_ood.png")
    )

    # 2. Extract continuous metric AUC means (Individual metrics ONLY, excluding N_flagged)
    cont_cols = [c for c in full.columns if c.startswith("AUC_") and c != "AUC_flagcount"]
    if cont_cols:
        cont_means = (
            full[full.Strategy == "Raw"]
            .groupby("Domain")[cont_cols]
            .mean()
            .T
        )
        # Clean index metric names using METRIC_MAP
        cont_means.index = [
            METRIC_MAP.get(c.replace("AUC_", ""), c.replace("AUC_", "")) 
            for c in cont_means.index
        ]

        # 3. Generate Figure 2 (Individual AD Metrics ROC-AUC)
        plot_per_metric_auc(cont_means, os.path.join(OUT_DIR, "fig1_per_metric_AUC_by_domain.png"))

    print("\nGroup Summary Results:")
    print(group_summary.round(3))


def _plot_single_domain_summary(summary: pd.DataFrame, domain_name: str, out_path: str, panel_letter: str):
    """
    Helper function to generate a 3-panel grouped bar chart (mean +/- SEM)
    comparing Raw vs. OOF thresholds for a specific domain regime with subfigure letter.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    panels = [
        ("pctOut_mean", "pctOut_sem", "% Outside AD", None),
        ("RR_mean", "RR_sem", "Relative Risk", 1.0),
        ("EF_mean", "EF_sem", "Enrichment Factor (Outside)", 1.0),
    ]
    
    strategies = ["Raw", "OOF"]
    colors = ["#1f77b4", "#ff7f0e"]
    x = np.arange(len(strategies))
    bar_width = 0.5

    domain_mask = summary["Domain"] == domain_name

    for idx, (ax, (mcol, scol, title, null_line)) in enumerate(zip(axes, panels)):
        vals = [
            summary[domain_mask & (summary["Strategy"] == strat)][mcol].values[0] 
            if not summary[domain_mask & (summary["Strategy"] == strat)].empty else 0
            for strat in strategies
        ]
        errs = [
            summary[domain_mask & (summary["Strategy"] == strat)][scol].values[0] 
            if not summary[domain_mask & (summary["Strategy"] == strat)].empty else 0
            for strat in strategies
        ]

        ax.bar(
            x, vals, width=bar_width, yerr=errs, capsize=5, 
            color=colors, edgecolor="black", linewidth=0.8, alpha=0.9
        )

        ax.set_xticks(x)
        ax.set_xticklabels(strategies, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        
        if null_line is not None:
            ax.axhline(null_line, color="grey", linestyle="--", linewidth=1.2)
            
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Add Subfigure Panel Letter ('A' or 'B') on first subplot
        if idx == 0:
            ax.text(
                -0.25, 1.15, panel_letter, transform=ax.transAxes,
                fontsize=20, fontweight="bold", va="top", ha="right"
            )

    plt.suptitle(f"{domain_name} Benchmarks (Raw vs. OOF Calibration)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_group_summary_bars(summary: pd.DataFrame, out_path_in_domain: str, out_path_ood: str):
    """
    Generates two 3-panel figures (mean +/- SEM) comparing Raw vs. OOF thresholds
    across % Outside AD, Relative Risk, and Enrichment Factor (Outside).
    """
    _plot_single_domain_summary(summary, "In-domain", out_path_in_domain, panel_letter="A")
    _plot_single_domain_summary(summary, "Out-of-domain", out_path_ood, panel_letter="B")


def plot_per_metric_auc(metric_means: pd.DataFrame, out_path: str):
    """
    Figure 2: Horizontal grouped bar chart of mean ROC-AUC for individual
    continuous AD metrics, grouped by Domain (In-domain vs Out-of-domain).
    Excludes N_flagged and uses LaTeX math formatting for subscripted labels.
    """
    if "In-domain" in metric_means.columns:
        metric_means = metric_means.sort_values("In-domain")
        
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(metric_means))
    h = 0.35

    if "In-domain" in metric_means.columns:
        ax.barh(y - h / 2, metric_means["In-domain"], height=h, label="In-domain", color=DOMAIN_COLORS["In-domain"])
    if "Out-of-domain" in metric_means.columns:
        ax.barh(y + h / 2, metric_means["Out-of-domain"], height=h, label="Out-of-domain", color=DOMAIN_COLORS["Out-of-domain"])
    
    ax.axvline(0.50, color="black", linestyle="--", linewidth=1.2, label="Random chance (0.50)")
    ax.set_yticks(y)
    ax.set_yticklabels(metric_means.index, fontsize=11)
    ax.set_xlabel("Mean ROC-AUC", fontweight="bold", fontsize=11)
    ax.set_title("Individual AD metric discriminative power", fontweight="bold", fontsize=12, pad=12)
    ax.set_xlim(0.40, 0.75)
    ax.legend(title="Domain Regime", loc="lower right", frameon=True)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()