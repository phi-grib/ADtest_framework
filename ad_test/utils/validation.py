"""
Validation utilities for ensuring the quality and disjointness of cross-validation folds.

Author: Eric March (eric.march@upf.edu)
Created: 2026-04-28
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def precompute_fingerprints(smiles_list, radius=2, n_bits=2048, logger=None):
    """
    Pre-calculates Morgan fingerprints for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings to fingerprint.
    radius : int
        Morgan fingerprint radius (default 2 ≈ ECFP4).
    n_bits : int
        Bit-vector length.
    logger : optional
        If provided, logs the number of successfully computed fingerprints.

    Returns
    -------
    fps : list
        List of RDKit fingerprint bit-vectors (None for failed molecules).
    """
    fps = []
    mols_count = 0
    for s in smiles_list:
        try:
            m = Chem.MolFromSmiles(s)
            if m:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits))
                mols_count += 1
            else:
                fps.append(None)
        except Exception:
            fps.append(None)

    if logger:
        logger.info(f"Pre-calculated fingerprints for {mols_count}/{len(smiles_list)} compounds.")
    return fps

def compute_fold_disjointness(fps, train_idx, val_idx, fold_num):
    """
    Calculates the maximum Tanimoto similarity of each validation compound
    to the training set for a single CV fold.

    Parameters
    ----------
    fps : list
        Pre-computed fingerprints (from ``precompute_fingerprints``).
    train_idx : array-like
        Positional indices of training compounds in this fold.
    val_idx : array-like
        Positional indices of validation compounds in this fold.
    fold_num : int
        Fold number (1-indexed), used for labelling.

    Returns
    -------
    list of dict
        Each dict has keys ``'Fold'`` and ``'Max Similarity'``.
        Empty list if either set has no valid fingerprints.
    """
    train_fps = [fps[j] for j in train_idx if fps[j] is not None]
    val_fps   = [fps[j] for j in val_idx   if fps[j] is not None]

    if not train_fps or not val_fps:
        return []

    results = []
    for f in val_fps:
        sims = DataStructs.BulkTanimotoSimilarity(f, train_fps)
        results.append({
            'Fold': f'Fold {fold_num}',
            'Max Similarity': max(sims)
        })
    return results