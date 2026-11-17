"""
Aggregates per-endpoint AD YAML reports (output of ADTestPipeline) into a
single comparison table of Raw vs. OOF thresholding, split by whether the
external test set represents an Out-of-domain relative to the training data.

Author: analysis helper for Eric March's AD/QSAR Out-of-domain study.
"""

import os
import yaml
import pandas as pd

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
YAML_DIR = str(REPO_ROOT / "results" / "final_analysis" / "yaml_dir")
OUT_DIR = str(REPO_ROOT / "results" / "final_analysis" / "outlier_tables")


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

# Endpoints whose external test set represents an Out-of-domain
OOD_ENDPOINTS = {"ACHE", "AChE", "DAT", "HDAC", "NET", "SERT"}


def endpoint_name(filename: str) -> str:
    """Extracts and formats the endpoint name from the standard filename pattern."""
    raw_ep = filename.split("_CV_")[0].strip()
    return ENDPOINT_MAP.get(raw_ep.lower(), raw_ep)


def parse_yaml_report(path: str) -> dict:
    """Loads a single AD YAML report and returns the raw dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def compute_rr_ef(total: int, inside: int, outside: int,
                   err_in_pct: float, err_out_pct: float) -> dict:
    """Recomputes Relative Risk (RR) and Enrichment Factor (EF)."""
    errors_in = inside * err_in_pct / 100.0
    errors_out = outside * err_out_pct / 100.0
    baseline_pct = (errors_in + errors_out) / total * 100.0 if total > 0 else float("nan")

    rr = err_out_pct / err_in_pct if err_in_pct > 0 else float("nan")
    ef_in = err_in_pct / baseline_pct if baseline_pct > 0 else float("nan")
    ef_out = err_out_pct / baseline_pct if baseline_pct > 0 else float("nan")

    return {
        "RR": rr, 
        "EF_inside": ef_in, 
        "EF_outside": ef_out,
        "baseline_error_pct": baseline_pct
    }


def build_comparison_table() -> pd.DataFrame:
    """Iterates over YAML reports and builds the long-format comparison table."""
    rows = []
    strategy_keys = {
        "Raw": "Raw (Training Boundaries)",
        "OOF": "OOF (CV Boundaries)",
    }

    files = [f for f in os.listdir(YAML_DIR) if f.endswith(".yaml")]

    for fname in sorted(files):
        ep = endpoint_name(fname)
        domain = "Out-of-domain" if ep in OOD_ENDPOINTS else "In-domain"
        report = parse_yaml_report(os.path.join(YAML_DIR, fname))

        summary = report.get("external_test_set_performance", {}).get("test_outlier_summary", {})
        if not summary:
            continue

        for strategy, key in strategy_keys.items():
            block = summary.get(key)
            if block is None:
                continue

            total = block["Total Test Set"]
            inside = block["Inside AD"]
            outside = block["Outside AD"]
            pct_inside = block.get("% Inside", round((inside / total) * 100.0, 2) if total > 0 else 0.0)
            pct_outside = block["% Outside"]
            err_in = block.get("% Errors Inside AD")
            err_out = block.get("% Errors Outside AD")

            metrics = compute_rr_ef(total, inside, outside, err_in, err_out)

            rows.append({
                "Endpoint": ep,
                "Domain": domain,
                "Strategy": strategy,
                "Total": total,
                "Inside": inside,
                "%Inside": pct_inside,
                "Outside": outside,
                "%Outside": pct_outside,
                "%ErrIn": err_in,
                "%ErrOut": err_out,
                **metrics
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_comparison_table()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "ad_ood_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(df.round(2).to_string(index=False))