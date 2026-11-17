"""
Defines the QSARModel class to encapsulate QSAR classification model training,
prediction, and evaluation functionalities.

This module provides:
- Initialization with customizable model hyperparameters.
- Training on a given dataset.
- Prediction of labels for a test set.
- Evaluation of classification performance with standard metrics.

Author: Eric March Vila (eric.march@upf.edu)
Created: 2025-11-17
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

class QSARModel:
    """
    A class to encapsulate QSAR classification modeling.

    Attributes:
        model_params (dict): Parameters used to initialize the classifier.
        model (sklearn classifier): The classifier instance after initialization.
        is_trained (bool): Flag indicating if the model has been trained.
    """

    def __init__(self, model_type='random_forest', model_params=None):
        """
        Initializes the QSARModel with specified hyperparameters.

        Parameters:
            model_params (dict, optional): Dictionary of model parameters.
                Defaults to a random forest with 100 trees and fixed random seed.
        """

        self.model_type = model_type
        if model_params is None:
            model_params = {}

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**model_params)
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.is_trained = False

    def train(self, X_train, y_train):
        """
        Trains the QSAR classification model on training data.

        Parameters:
            X_train (pd.DataFrame or np.ndarray): Training feature matrix.
            y_train (pd.Series or np.ndarray): Training target labels.
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test):
        """
        Predicts labels for the test set using the trained model.

        Parameters:
            X_test (pd.DataFrame or np.ndarray): Test feature matrix.

        Returns:
            np.ndarray: Predicted labels.

        Raises:
            RuntimeError: If called before model is trained.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X_test)

    def evaluate_classification_metrics(self, y_true, y_pred):
        """
        Compute key classification metrics commonly used for QSAR model evaluation.
        Handles Conformal Prediction output (-1 for uncertain) by calculating metrics
        only on the 'covered' (confident) subset.

        Args:
            y_true (np.ndarray or pd.Series): True binary labels (0 or 1).
            y_pred (np.ndarray or pd.Series): Predicted labels (0, 1, or -1 for uncertain).

        Returns:
            dict: Dictionary with metrics (Accuracy, Precision, Sensitivity, Specificity, F1, MCC)
                  plus 'coverage' and confusion matrix counts.
        """
        # Ensure inputs are numpy arrays for boolean indexing
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        # --- Conformal Prediction Handling ---
        # 1. Calculate Coverage (How many are NOT -1)
        total_samples = len(y_pred)
        mask = y_pred != -1
        coverage = float(np.sum(mask) / total_samples) if total_samples > 0 else 0.0

        # 2. Filter data to keep only confident predictions
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        # Safety Check: If NO predictions were confident (or empty input), return 0s
        if len(y_true_filtered) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'sensitivity': 0.0,
                'specificity': 0.0,
                'f1_score': 0.0,
                'mcc': 0.0,
                'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
                'coverage': coverage
            }

        # --- Standard Metric Calculation (on filtered data) ---
        # Note: We use labels=[0, 1] to ensure confusion_matrix always returns 4 values 
        # even if one class is missing in the filtered subset.
        tn, fp, fn, tp = confusion_matrix(y_true_filtered, y_pred_filtered, labels=[0, 1]).ravel()

        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        precision = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
        sensitivity = recall_score(y_true_filtered, y_pred_filtered, zero_division=0) # Recall (Pos)
        specificity = recall_score(y_true_filtered, y_pred_filtered, pos_label=0, zero_division=0) # Recall (Neg)
        f1 = f1_score(y_true_filtered, y_pred_filtered, zero_division=0)
        mcc = matthews_corrcoef(y_true_filtered, y_pred_filtered)

        return {
            'accuracy': accuracy,
            'precision': precision, 
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'mcc': mcc,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'coverage': coverage # Added coverage metric
        }

    @staticmethod
    def get_predictions(logger, config, test_struc, X_train, y_train, X_test, y_test, is_cv=False):
        """
        Strict prediction retrieval:
        1. If prediction_column is provided, use it. Crash if missing.
        2. If use_qsar is True (or we are inside CV), train the model.
        3. Never silently fall back to training if the user didn't ask for it.
        """

        pred_col = config.get('prediction_column', False)
        use_qsar = config.get('use_qsar', False)

        # 1. Try to use pre-calculated column FIRST
        if pred_col and not is_cv and not use_qsar:
            if pred_col in test_struc.columns:
                logger.info(f"Using pre-calculated predictions from column '{pred_col}'.")
                return test_struc[pred_col]
            else:
                # HARD STOP: If they specified a column and it's missing, crash.
                raise ValueError(f"CRITICAL: prediction_column '{pred_col}' not found in dataset! "
                                 f"Available columns: {list(test_struc.columns)}")

        # 2. If no prediction column, check if we are allowed to train
        if use_qsar or is_cv:
            qsar_config = config.get('qsar_model', {})
            model_type = qsar_config.get('type', 'random_forest')
            
            logger.info(f"Training internal {model_type} QSAR model...")
            try:
                model_instance = QSARModel(model_type=model_type, model_params=qsar_config.get('params', {}))
                model_instance.train(X_train, y_train)
                y_pred = model_instance.predict(X_test)
                return pd.Series(y_pred, index=y_test.index)
            except Exception as e:
                logger.error(f"Failed to train QSAR model: {e}", exc_info=True)
                raise RuntimeError(f"QSAR Training failed: {e}")
        
        # 3. If we reach here, no column was provided and training is disabled.
        raise ValueError("No prediction_column provided and use_qsar is False. Cannot get predictions!")