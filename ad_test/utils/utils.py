"""
Utility functions for the AD Test framework.

Author: Eric March (eric.march@upf.edu)
Created: 2026-02-09
"""

import pandas as pd

from typing import Tuple, Optional

from .logger import LOG

def _align_data_indices(X_train_main: pd.DataFrame, y_train_main: pd.DataFrame, train_struc: pd.DataFrame, scores_df: pd.DataFrame, 
                        y_train_for_clustering: pd.Series, logger: LOG) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame], 
                        Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Aligns AD descriptor data and Clustering score data by their common indices.
    This is crucial when different descriptor sets (e.g., RDKit vs. FPs)
    result in different sets of valid compounds after NaN removal.
    
    Returns:
        A tuple of (X_train_main_aligned, y_train_main_aligned, 
                    scores_df_aligned, y_train_for_clustering_aligned).
        Returns (None, None, None, None) if alignment fails.
    """
    logger.info("Aligning descriptor sets by finding common indices...")

    ad_scores_index = X_train_main.index
    cluster_scores_index = scores_df.index
    
    common_indices = ad_scores_index.intersection(cluster_scores_index)
    
    if len(common_indices) < len(ad_scores_index) or len(common_indices) < len(cluster_scores_index):
        logger.warning(
            f"Index mismatch detected. "
            f"AD Descriptors had {len(ad_scores_index)} compounds. "
            f"Cluster Descriptors/Scores had {len(cluster_scores_index)} compounds. "
            f"Proceeding with {len(common_indices)} common compounds."
        )
        
    if len(common_indices) == 0:
        logger.error("No common compounds found between AD and Cluster descriptor sets. Cannot proceed.")
        return None, None, None, None

    # Filter all datasets to the common indices
    X_train_main_aligned = X_train_main.loc[common_indices]
    y_train_main_aligned = y_train_main.loc[common_indices]
    
    scores_df_aligned = scores_df.loc[common_indices]
    y_train_for_clustering_aligned = y_train_for_clustering.loc[common_indices]
    
    train_struc_aligned = train_struc.loc[common_indices] # <-- Align train_struc
    
    logger.info(f"All data aligned to {len(common_indices)} common compounds.")
    
    return X_train_main_aligned, y_train_main_aligned, scores_df_aligned, y_train_for_clustering_aligned, train_struc_aligned

def generate_filename_prefix(config: dict) -> str:
    """
    Generates a standardized filename prefix based on config.
    
    Example (Single): "DILI_single_pls_fps"
    Example (CV): "DILI_CV_fps_fps" 
    (Reads 'dataset_name' from config, which is 'name' in command_file.yaml)
    """
    # Use .get() with fallbacks for safety
    # 'dataset_name' should be added to config when processing each dataset
    dataset_name = str(config.get('name', 'dataset'))
    if not dataset_name: dataset_name = 'dataset'
    dataset_name = dataset_name.upper().replace(" ", "_").replace("-", "_")

    is_cv_run = config.get('cluster_analysis', False)

    run_type = "CV" if is_cv_run else "single"
    
    if is_cv_run:
        # CV Run Format: {dataset_name}_{run_type}_{cluster_descriptors}_{cluster_algorithm}_{n_clusters}_{ad_descriptors}
        
        # Get AD descriptors
        ad_descriptors = str(config.get('descriptors', 'ad_descs'))
        if not ad_descriptors: ad_descriptors = 'ad_descs'
        
        # Get Cluster descriptors
        cluster_descriptors = str(config.get('cluster_descriptors', ''))
        if not cluster_descriptors: 
            cluster_descriptors = ad_descriptors # Fallback to AD descriptors
        
        # Get clustering algorithm
        cluster_algorithm = str(config.get('cluster_algorithm', ''))

        # Get number of clusters
        n_clusters = str(config.get('n_clusters', ''))

        # Sanitize
        ad_descriptors = ad_descriptors.lower().replace(" ", "")
        cluster_descriptors = cluster_descriptors.lower().replace(" ", "")
        cluster_algorithm = cluster_algorithm.lower().replace(" ", "")
        n_clusters = n_clusters.lower().replace(" ", "")

        return f"{dataset_name}_{run_type}_{cluster_descriptors}_{cluster_algorithm}_{n_clusters}_{ad_descriptors}"
        
    else:
        # Single Run Format: {dataset_name}_{run_type}_{model}_{descriptors}
        model = str(config.get('model', 'model'))
        descriptors = str(config.get('descriptors', 'descs'))
        
        # Clean up potential None or empty strings
        if not model: model = 'model'
        if not descriptors: descriptors = 'descs'

        model = model.lower().replace(" ", "")
        descriptors = descriptors.lower().replace(" ", "")
        
        return f"{dataset_name}_{run_type}_{model}_{descriptors}"