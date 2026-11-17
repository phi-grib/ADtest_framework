"""
Helper functions for data input and output.

Author: Eric March (eric.march@upf.edu)
Created: 2026-04-28
"""

import os
import pandas as pd

from typing import Tuple, Optional

from .preprocess_datasets import Preprocess

def _open_input_files(dataset_path: str) -> pd.DataFrame:
    """
    Opens and reads the input dataset file and returns a DataFrame.
    Checks for file type (CSV, TSV, XLSX) and uses appropriate pandas reader.
    
    :param dataset_path: Path to the dataset file.
    :return df: Processed dataframe.
    """

    file_ext = os.path.splitext(dataset_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(dataset_path, sep='\t')
    elif file_ext in ['.tsv', '.tab']:
        df = pd.read_csv(dataset_path, sep='\t')
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported file extension '{file_ext}' for dataset: {dataset_path}")
    
    return df

def _handle_input_files(config: dict, logger) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Opens and reads the input dataset files (train and test) based on config.
    
    :param config: Configuration dictionary for the dataset/run.
    :param logger: Logger instance for logging messages.
    :return: Tuple of (train_df, test_df) DataFrames.
    """
    try:
        logger.info(f"Loading training data from: {config['train_path']}")
        train_df = _open_input_files(config['train_path'])
        
        if config.get('test_path'):
            logger.info(f"Loading testing data from: {config['test_path']}")
            test_df = _open_input_files(config['test_path'])
        else:
            test_df = None
            
        return train_df, test_df
    
    except FileNotFoundError as e:
        logger.critical(f"Data file not found: {e}. Aborting for this dataset.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        raise

def _load_and_preprocess_data(config: dict, logger):
    """
    Loads raw data and runs the full preprocessing pipeline.
    
    NOTE: This function assumes a 'Preprocess' class that is compatible
    with the 'descriptors' argument in __init__ and 'execute_pipeline' method.
    """
    logger.info(f"--- Starting processing for dataset: {config['name']} ---")
    
    # Decide which descriptors to load based on the analysis type
    if config.get('cluster_analysis', False):
        # If CV run, use cluster_descriptors (or fallback to main)
        main_desc = config['descriptors']
        cluster_desc = config.get('cluster_descriptors') or main_desc
        descriptors_to_use = cluster_desc
        
        if cluster_desc != main_desc:
            logger.info(f"Cluster analysis detected. Loading initial descriptors using 'cluster_descriptors': {descriptors_to_use}")
        else:
            logger.info(f"Cluster analysis detected. Using main 'descriptors' for clustering: {descriptors_to_use}")
    else:
        # If single run, use main descriptors
        descriptors_to_use = config['descriptors']
        logger.info(f"Single analysis detected. Using 'descriptors' for preprocessing: {descriptors_to_use}")

    # --- 1. Load Data ---
    train_df, test_df = _handle_input_files(config, logger)

    # --- 2. Preprocessing ---
    logger.info("Starting data preprocessing...")
    try:
        
        # NEW: Retrieve prediction_column from config (defaults to None if missing/False)
        pred_col = config.get('prediction_column')
        # Handle the case where YAML might parse 'False' boolean as False, or it might be None
        if pred_col is False: 
            pred_col = None

        preprocessor = Preprocess(
            train_series=train_df,
            test_series=test_df,
            id_col=config['id_col'],
            struct_col=config['smiles_col'], # Assumes Preprocess uses 'struct_col'
            activity_col=config['activity_col'],
            descriptors=descriptors_to_use,
            prediction_col=pred_col,
            logger=logger
        )

        X_train_full, y_train_full, train_struc, X_test_full, y_test_full, test_struc = preprocessor.execute_pipeline()
        logger.info("Data preprocessing complete.")
        
        return X_train_full, y_train_full, train_struc, X_test_full, y_test_full, test_struc
    
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}", exc_info=True)
        raise

def _prepare_main_descriptors(config, logger, X_train_full, y_train_full, train_struc, X_test_full, y_test_full, test_struc):
    """
    Ensures that if clustering uses different descriptors, we still have the 
    correct 'main' descriptors for the actual AD analysis.
    """
    main_desc = config['descriptors']
    cluster_desc = config.get('cluster_descriptors') or main_desc
    
    # Fast path: If descriptors match, just return input data
    if cluster_desc == main_desc:
        return X_train_full, y_train_full, train_struc, X_test_full, y_test_full, test_struc

    logger.info(f"Main descriptors ('{main_desc}') differ from cluster descriptors ('{cluster_desc}'). Reloading...")
    try:
        # Re-load raw data (Assumes _handle_input_files is available in scope)
        raw_train_df, raw_test_df = _handle_input_files(config, logger)

        main_preprocessor = Preprocess(
            train_series=raw_train_df, 
            test_series=raw_test_df, 
            id_col=config['id_col'], 
            struct_col=config['smiles_col'], 
            activity_col=config['activity_col'], 
            descriptors=main_desc,
            prediction_col=config.get('prediction_column'),
            logger=logger
        )
        
        # Overwrite with new main descriptors
        return main_preprocessor.execute_pipeline()
        
    except Exception as e:
        logger.error(f"Failed to generate *main* descriptors: {e}", exc_info=True)
        return None