import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import mannwhitneyu

# Suppress minor RDKit parsing warnings
warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_YAML_PATH = str(REPO_ROOT / "ad_final_cv_analysis_files.yaml")
OUT_DIR = str(REPO_ROOT / "results" / "final_analysis" / "figures")

# Global endpoint formatting dictionary
ENDPOINT_MAP = {
    "ACHE": "AChE",
    "Ache": "AChE",
    "CARCINOGEN": "Carcinogen",
    "CARCINOGENICITY": "Carcinogen",
    "Carcinogenicity": "Carcinogen",
    "ER_ALPHA": "ER alpha",
    "Er_alpha": "ER alpha",
    "ER_BETA": "ER beta",
    "Er_beta": "ER beta",
    "HERG": "hERG",
    "Herg": "hERG",
    "SKIN_REACTION": "Skin Reaction",
    "SKIN_SENSITIZATION": "Skin Reaction",
    "Skin_sensitization": "Skin Reaction",
    "TR_ALPHA": "TR alpha",
    "Tr_alpha": "TR alpha",
    # Standard Endpoints
    "AMES": "AMES",
    "AR": "AR",
    "DAT": "DAT",
    "DILI": "DILI",
    "GR": "GR",
    "HDAC": "HDAC",
    "NET": "NET",
    "SERT": "SERT",
}

# Out-of-Domain endpoints (5 bioactivity/DNT extrapolation benchmarks)
OOD_ENDPOINTS = {"ACHE", "AChE", "DAT", "HDAC", "NET", "SERT"}

# Candidate column headers for SMILES strings
SMILES_COL_CANDIDATES = [
    "canon_smiles", "SMILES", "smiles", "canonical_smiles", 
    "smiles_str", "Structure", "SMILE", "SMILES_str", "Drug", "Compound", "mol"
]


# =============================================================================
# BULLETPROOF DATA & FILE LOADERS
# =============================================================================
def load_smiles_from_file(file_path: str) -> list:
    """
    Robustly reads SMILES strings from CSV, TSV, or Excel (.xlsx / .xls) files,
    handling arbitrary delimiters, encodings, and missing headers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    df = None

    # Priority 1: Direct Excel check
    if ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(file_path)
        except Exception:
            pass

    # Priority 2: Fallback pipeline across delimiters, encodings, and Excel readers
    if df is None or df.empty:
        readers = [
            lambda: pd.read_csv(file_path),
            lambda: pd.read_csv(file_path, sep="\t"),
            lambda: pd.read_csv(file_path, sep=";"),
            lambda: pd.read_csv(file_path, sep=None, engine="python"),
            lambda: pd.read_csv(file_path, encoding="latin1"),
            lambda: pd.read_csv(file_path, sep="\t", encoding="latin1"),
            lambda: pd.read_csv(file_path, sep=";", encoding="latin1"),
            lambda: pd.read_excel(file_path),  # Handles binary Excel files misnamed as .csv
        ]

        for reader in readers:
            try:
                candidate_df = reader()
                if candidate_df is not None and not candidate_df.empty and len(candidate_df.columns) > 0:
                    df = candidate_df
                    break
            except Exception:
                continue

    if df is None or df.empty:
        raise ValueError(f"Unable to parse table structure from: {file_path}")

    # Identify SMILES column
    smiles_col = None

    # 1. Match known header names
    for col in df.columns:
        col_clean = str(col).strip()
        if col_clean in SMILES_COL_CANDIDATES or "smiles" in col_clean.lower():
            smiles_col = col
            break

    # 2. Fallback: Validate column contents with RDKit
    if smiles_col is None:
        for col in df.columns:
            sample_vals = df[col].dropna().astype(str).head(10).tolist()
            valid_mols = sum(1 for s in sample_vals if Chem.MolFromSmiles(s) is not None)
            if valid_mols >= max(1, len(sample_vals) // 2):
                smiles_col = col
                break

    if smiles_col is None:
        smiles_col = df.columns[0]

    return df[smiles_col].dropna().astype(str).tolist()


def parse_master_yaml(yaml_path: str) -> dict:
    """
    Recursively parses master YAML structure to extract (train_path, test_path) pairs per endpoint.
    Applies ENDPOINT_MAP formatting directly to extracted endpoint keys.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Master YAML file not found at: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    endpoint_paths = {}

    def extract_from_dict(d: dict, fallback_name: str = None):
        train_keys = ["train_path", "train_file", "training_data", "train_data", "train", "X_train"]
        test_keys = ["test_path", "test_file", "external_test_data", "test_data", "test", "X_test"]

        train_p = None
        for k in train_keys:
            if k in d and d[k]:
                train_p = str(d[k])
                break

        test_p = None
        for k in test_keys:
            if k in d and d[k]:
                test_p = str(d[k])
                break

        raw_name = d.get("name") or d.get("endpoint") or d.get("dataset_name") or d.get("dataset") or fallback_name

        if train_p and not os.path.isabs(train_p):
            train_p = str(REPO_ROOT / train_p)
        if test_p and not os.path.isabs(test_p):
            test_p = str(REPO_ROOT / test_p)
            
        return raw_name, train_p, test_p

    def traverse(node, current_key=None):
        if isinstance(node, dict):
            name, train_p, test_p = extract_from_dict(node, fallback_name=current_key)
            if train_p and test_p and name and isinstance(name, str) and not name.lower().startswith("common"):
                ep_raw = name.split("_CV_")[0].upper()
                ep_clean = ENDPOINT_MAP.get(ep_raw, ep_raw)
                endpoint_paths[ep_clean] = (train_p, test_p)
            else:
                for k, v in node.items():
                    traverse(v, current_key=k)
        elif isinstance(node, list):
            for item in node:
                traverse(item, current_key=current_key)

    traverse(data)
    return endpoint_paths


def compute_max_tanimoto(train_smiles: list, test_smiles: list) -> list:
    """Calculates Max Tanimoto Similarity (S_max) to the training set for each test molecule."""
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles if s and isinstance(s, str)]
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles if s and isinstance(s, str)]

    train_mols = [m for m in train_mols if m is not None]
    test_mols = [m for m in test_mols if m is not None]

    if not train_mols or not test_mols:
        return []

    train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in train_mols]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in test_mols]

    max_sims = []
    for t_fp in test_fps:
        sims = DataStructs.BulkTanimotoSimilarity(t_fp, train_fps)
        max_sims.append(max(sims) if sims else 0.0)

    return max_sims


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("--- Starting Max Tanimoto Similarity (S_max) Analysis ---")
    
    yaml_path = MASTER_YAML_PATH
    if not os.path.exists(yaml_path):
        user_input = input(f"Default master YAML not found at '{yaml_path}'. Enter path: ").strip()
        yaml_path = user_input.strip("'\"")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading master configuration from: {yaml_path}")
    endpoints_map = parse_master_yaml(yaml_path)

    if not endpoints_map:
        print("\n[Error] Could not extract any valid (train_path, test_path) pairs from the YAML.")
        sys.exit(1)

    print(f"Successfully extracted {len(endpoints_map)} endpoints: {', '.join(sorted(endpoints_map.keys()))}\n")

    all_records = []

    for endpoint, (train_p, test_p) in sorted(endpoints_map.items()):
        # Determine domain (checks formatted endpoint name)
        domain = "Out-of-Domain" if endpoint in OOD_ENDPOINTS else "In-Domain"
        print(f"Processing {endpoint} ({domain})...")

        try:
            train_smiles = load_smiles_from_file(train_p)
            test_smiles = load_smiles_from_file(test_p)

            s_max_values = compute_max_tanimoto(train_smiles, test_smiles)

            for val in s_max_values:
                all_records.append({
                    "Endpoint": endpoint,
                    "Domain": domain,
                    "S_max": val
                })

            print(f"  -> {len(train_smiles)} train | {len(test_smiles)} test compounds | Mean S_max = {np.mean(s_max_values):.3f}")

        except Exception as e:
            print(f"  [Error] Skipping {endpoint}: {e}")

    if not all_records:
        print("\n[Error] No similarity records were collected. Check dataset file paths.")
        sys.exit(1)

    df_results = pd.DataFrame(all_records)

    # Save raw compound-level results
    raw_csv_path = os.path.join(OUT_DIR, "domain_shift_smax_values.csv")
    df_results.to_csv(raw_csv_path, index=False)
    print(f"\nSaved raw S_max values to: {raw_csv_path}")

    # Statistical Summary & Mann-Whitney U Test
    id_smax = df_results[df_results["Domain"] == "In-Domain"]["S_max"]
    ood_smax = df_results[df_results["Domain"] == "Out-of-Domain"]["S_max"]

    id_mean, id_median = id_smax.mean(), id_smax.median()
    ood_mean, ood_median = ood_smax.mean(), ood_smax.median()

    stat, p_val = mannwhitneyu(id_smax, ood_smax, alternative="greater")

    if p_val < 0.001:
        p_val_result = "p < 0.001"
    else:
        p_val_result = f"p = {p_val:.3f}"
    
    print("\n" + "="*65)
    print("STATISTICAL SUMMARY & DOMAIN SHIFT VERIFICATION")
    print("="*65)
    print(f"In-Domain (ID)      -> Mean S_max: {id_mean:.3f} | Median: {id_median:.3f} | N = {len(id_smax)}")
    print(f"Out-of-Domain (OOD) -> Mean S_max: {ood_mean:.3f} | Median: {ood_median:.3f} | N = {len(ood_smax)}")
    print(f"Mann-Whitney U Test (ID > OOD): U = {stat:.1f}, {p_val_result}")
    print("="*65 + "\n")

    # Generate Publication Figure
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(13, 6))

    palette = {"In-Domain": "#3498db", "Out-of-Domain": "#e74c3c"}

    endpoint_order = sorted(df_results[df_results["Domain"] == "In-Domain"]["Endpoint"].unique()) + \
                     sorted(df_results[df_results["Domain"] == "Out-of-Domain"]["Endpoint"].unique())

    ax = sns.boxplot(
        data=df_results, x="Endpoint", y="S_max", hue="Domain",
        order=endpoint_order, palette=palette, dodge=False, showmeans=True, showfliers=False,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "6"}
    )

    plt.axhline(0.4, color="gray", linestyle="--", linewidth=1.2, label=r"Similarity threshold ($S_{\mathrm{max}} = 0.40$)")

    plt.title(f"Chemical space structural overlap: In-Domain vs. Out-of-Domain benchmarks\n"
              f"(Mean $S_{{max}}$: ID = {id_mean:.2f} vs. OOD = {ood_mean:.2f}, Mann–Whitney U {p_val_result})",
              fontsize=13, fontweight="bold", pad=15)

    plt.xlabel("Endpoints", fontsize=11, fontweight="bold")
    plt.ylabel(r"Max Tanimoto Similarity ($S_{\mathrm{max}}$ to Training Set)", fontsize=11, fontweight="bold")
    plt.ylim(-0.02, 1.05)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.legend(title="Domain Regime", loc="upper right", frameon=True)
    plt.tight_layout()

    fig_out_path = os.path.join(OUT_DIR, "supfig1_domain_shift_smax_boxplot.png")
    plt.savefig(fig_out_path, dpi=300)
    plt.close()

    print(f"Saved publication figure to: {fig_out_path}")
    print("--- Analysis Completed Successfully ---")


if __name__ == "__main__":
    main()