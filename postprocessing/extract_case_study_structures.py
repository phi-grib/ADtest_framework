"""
Extracts representative compound case studies and chemical structures for:
  1. AMES (In-Domain, 6 compounds total):
     - 3 Recovered: Correctly predicted (E=0), T_raw Outside -> T_OOF Inside
     - 3 Purged: Mispredicted (E=1), T_raw Inside -> T_OOF Outside
  2. AChE (Chemical Domain Shift, 6 compounds total):
     - 6 Test compounds shifting T_raw Outside -> T_OOF Inside
     - Nearest training set match (kNN_1) & Tanimoto similarity (S_max)

Outputs summary tables in Excel (.xlsx) format without unnecessary metric clutter.

Author: Analysis helper for AD/QSAR paper.
"""

import os
import re
import pandas as pd
import numpy as np

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = str(REPO_ROOT / "results" / "final_analysis")
OUTLIER_DIR = os.path.join(BASE_DIR, "outlier_tables")
IMG_OUT_DIR = os.path.join(BASE_DIR, "case_study_images")

# AChE exact training set path
ACHE_TRAIN_FILE = str(REPO_ROOT / "DNT_case_study" / "ache" / "AChE_fixed.xlsx")

os.makedirs(IMG_OUT_DIR, exist_ok=True)

def clean_filename(text: str) -> str:
    """Sanitizes compound IDs for safe filename creation."""
    return re.sub(r'[^\w\-_\. ]', '_', str(text)).replace(' ', '_')

def resolve_cols(df: pd.DataFrame, default_smiles: str, default_id: str):
    """Resolves exact SMILES and ID column names flexibly."""
    smiles_col = None
    id_col = None

    for col in df.columns:
        if col.lower() == default_smiles.lower():
            smiles_col = col
        if col.lower() == default_id.lower():
            id_col = col

    if not smiles_col:
        for col in df.columns:
            if "smiles" in col.lower() or col.lower() == "drug":
                smiles_col = col
                break

    if not id_col:
        for col in df.columns:
            if "chembl" in col.lower() or "id" in col.lower() or "drug" in col.lower():
                id_col = col
                break

    return id_col, smiles_col

def draw_and_save_structure(smiles: str, out_path: str, img_size=(300, 300)):
    """Renders a 2D chemical structure image from SMILES."""
    if pd.isna(smiles) or not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol:
        AllChem.Compute2DCoords(mol)
        Draw.MolToFile(mol, out_path, size=img_size)
        return True
    return False


# ==============================================================================
# 1. AMES CASE STUDY EXTRACTION (6 Compounds: 3 Recovered + 3 Purged)
# ==============================================================================
def process_ames():
    print("\n================ Processing AMES Case Study (6 compounds) ================")
    ames_file = [f for f in os.listdir(OUTLIER_DIR) if "AMES" in f and f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(OUTLIER_DIR, ames_file))

    # AMES: SMILES is 'Drug', ID is 'Drug_ID'
    id_col, smiles_col = resolve_cols(df, default_smiles="Drug", default_id="Drug_ID")
    err_col = "Prediction_Error"

    print(f"AMES -> ID Column: '{id_col}' | SMILES Column: '{smiles_col}'")

    # 3 Recovered: Correctly predicted (E=0), Flagged Outside by T_raw, Inside by T_OOF
    group_a = df[
        (df[err_col] == 0) & 
        (df["Total Metrics Flagged_raw"] > 0) & 
        (df["Total Metrics Flagged_oof"] == 0)
    ].head(3)

    # 3 Purged: Mispredicted (E=1), Flagged Inside by T_raw, Outside by T_OOF
    group_b = df[
        (df[err_col] == 1) & 
        (df["Total Metrics Flagged_raw"] == 0) & 
        (df["Total Metrics Flagged_oof"] > 0)
    ].head(3)

    extracted_rows = []

    # Process 3 Recovered Compounds
    for idx, row in group_a.iterrows():
        cid = str(row[id_col])
        smiles = str(row[smiles_col])
        img_filename = f"AMES_Recovered_E0_{clean_filename(cid)}.png"
        img_path = os.path.join(IMG_OUT_DIR, img_filename)
        draw_and_save_structure(smiles, img_path)

        extracted_rows.append({
            "Transition_Type": "Recovered (T_raw Out -> T_OOF In)",
            "Compound_ID": cid,
            "Prediction_Status": "Correct (E = 0)",
            "T_raw_Status": "Outside AD",
            "T_OOF_Status": "Inside AD",
            "SMILES": smiles,
            "Image_Filename": img_filename
        })

    # Process 3 Purged Compounds
    for idx, row in group_b.iterrows():
        cid = str(row[id_col])
        smiles = str(row[smiles_col])
        img_filename = f"AMES_Purged_E1_{clean_filename(cid)}.png"
        img_path = os.path.join(IMG_OUT_DIR, img_filename)
        draw_and_save_structure(smiles, img_path)

        extracted_rows.append({
            "Transition_Type": "Purged (T_raw In -> T_OOF Out)",
            "Compound_ID": cid,
            "Prediction_Status": "Error (E = 1)",
            "T_raw_Status": "Inside AD",
            "T_OOF_Status": "Outside AD",
            "SMILES": smiles,
            "Image_Filename": img_filename
        })

    results_df = pd.DataFrame(extracted_rows)
    excel_out = os.path.join(BASE_DIR, "AMES_case_studies_table_n6.xlsx")
    results_df.to_excel(excel_out, index=False)

    print(f"Saved AMES case study Excel table to: {excel_out}")
    print(f"Saved 2D structure images to: {IMG_OUT_DIR}")
    return results_df


# ==============================================================================
# 2. AChE CASE STUDY EXTRACTION (6 Test Compounds + Nearest Train Matches)
# ==============================================================================
def process_ache():
    print("\n================ Processing AChE Case Study (6 compounds) ================")
    ache_outlier_file = [f for f in os.listdir(OUTLIER_DIR) if "ACHE" in f and f.endswith(".csv")][0]
    test_df = pd.read_csv(os.path.join(OUTLIER_DIR, ache_outlier_file))

    if not os.path.exists(ACHE_TRAIN_FILE):
        raise FileNotFoundError(f"Could not find AChE training file at: {ACHE_TRAIN_FILE}")

    train_df = pd.read_excel(ACHE_TRAIN_FILE)

    test_id_col, test_smiles_col = resolve_cols(test_df, default_smiles="canon_smiles", default_id="Molecule ChEMBL ID")
    train_id_col, train_smiles_col = resolve_cols(train_df, default_smiles="structure_curated", default_id="Molecule ChEMBL ID")

    print(f"AChE Test  -> ID: '{test_id_col}' | SMILES: '{test_smiles_col}'")
    print(f"AChE Train -> ID: '{train_id_col}' | SMILES: '{train_smiles_col}'")

    train_fps = []
    train_mols = []
    for idx, r in train_df.iterrows():
        smi = str(r[train_smiles_col]) if pd.notna(r[train_smiles_col]) else ""
        mol = Chem.MolFromSmiles(smi.strip()) if smi else None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None
        train_mols.append((str(r[train_id_col]), mol, smi))
        train_fps.append(fp)

    shifted_test = test_df[
        (test_df["Total Metrics Flagged_raw"] > 0) & 
        (test_df["Total Metrics Flagged_oof"] == 0)
    ].head(6)

    extracted_rows = []

    for idx, row in shifted_test.iterrows():
        test_id = str(row[test_id_col])
        test_smiles = str(row[test_smiles_col])
        test_mol = Chem.MolFromSmiles(test_smiles.strip()) if test_smiles else None
        test_fp = AllChem.GetMorganFingerprintAsBitVect(test_mol, 2, nBits=2048) if test_mol else None

        best_sim = -1.0
        best_train_id = None
        best_train_smiles = None

        if test_fp:
            for j, tr_fp in enumerate(train_fps):
                if tr_fp:
                    sim = DataStructs.TanimotoSimilarity(test_fp, tr_fp)
                    if sim > best_sim:
                        best_sim = sim
                        best_train_id = train_mols[j][0]
                        best_train_smiles = train_mols[j][2]

        test_img_file = f"AChE_Test_{clean_filename(test_id)}.png"
        train_img_file = f"AChE_Train_Match_{clean_filename(best_train_id)}.png"

        draw_and_save_structure(test_smiles, os.path.join(IMG_OUT_DIR, test_img_file))
        draw_and_save_structure(best_train_smiles, os.path.join(IMG_OUT_DIR, train_img_file))

        err_val = row["Prediction_Error"]
        pred_status = "Error (E = 1)" if err_val == 1 else "Correct (E = 0)"

        extracted_rows.append({
            "Test_Compound_ID": test_id,
            "Prediction_Status": pred_status,
            "T_raw_Status": "Outside AD",
            "T_OOF_Status": "Inside AD",
            "S_max_Tanimoto": round(best_sim, 3),
            "Nearest_Train_Match_ID": best_train_id,
            "Test_SMILES": test_smiles,
            "Train_Match_SMILES": best_train_smiles,
            "Test_Image_File": test_img_file,
            "Train_Match_Image_File": train_img_file
        })

    results_df = pd.DataFrame(extracted_rows)
    excel_out = os.path.join(BASE_DIR, "AChE_case_studies_table_n6.xlsx")
    results_df.to_excel(excel_out, index=False)

    print(f"Saved AChE case study Excel table to: {excel_out}")
    print(f"Saved 2D structure images to: {IMG_OUT_DIR}")
    return results_df


def main():
    process_ames()
    process_ache()

if __name__ == "__main__":
    main()