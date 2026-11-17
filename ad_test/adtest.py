"""
This module defines the ADTest class for robust Applicability Domain (AD) analysis
in chemoinformatics and machine learning.
[MODIFIED: 2025-11-13 for CV-based validation workflow]

It provides methods for:
1. Fitting AD models (PCA, PLS, LOF, kNN, KDE) on a training set.
2. Calculating raw AD metrics for new query compounds against the fitted models.
3. Calculating empirical thresholds based on the training data distribution.

Author: Eric March (eric.march@upf.edu)
Original: 2025-09-01
Refactored: 2025-11-13
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from scipy.spatial.distance import cdist, pdist, squareform
from typing import Dict, Any, Tuple, Optional

from .plots.plots import Plotter
from .utils.logger import LOG

class ADTest:
    def __init__(self, X_train: np.ndarray, y_train: pd.Series, logger=None, n_components: Optional[int] = None,
                 descriptors: str = 'rdkit', report_base_path: str = None, random_state: int = 42,
                 train_struc: pd.DataFrame = None,
                 id_col: str = None,
                 smiles_col: str = None,
                 k_knn: int = 5,
                 alpha_percentile: float = 0.05,
                 alpha_exponent_adsal: float = 5.0,
                 kde_bandwidth: Optional[float] = None,
                 model_type: str = 'pca'):
        """
        Initializes the ADTest object by fitting all necessary models on the TRAINING data.
        This object is now a "fitted" AD engine, ready to be queried.
        """

        # --- 1. Store Data and Configs ---
        self.X_train = X_train
        self.y_train = y_train
        self.descriptor_type = descriptors
        self.model_type = model_type
        self.n_components = n_components
        self.k_knn = k_knn
        self.alpha_percentile = alpha_percentile
        self.alpha_exponent_adsal = alpha_exponent_adsal
        self.kde_bandwidth = kde_bandwidth
        self.random_state = random_state

        self.train_struc = train_struc
        self.id_col = id_col
        self.smiles_col = smiles_col

        self.logger = logger if logger else LOG(name="adtest_default", log_file_path="adtest.log")
        self.plot = Plotter(report_base_path=report_base_path, logger=self.logger)

        if self.X_train is None:
            self.logger.error("X_train must be provided.")
            raise ValueError("X_train must be provided.")

        self.logger.info(f"Initializing ADTest engine with {self.X_train.shape[0]} training samples.")

        # --- 2. Store Data Dimensions and Names (ADDED) ---
        self.n_samples, self.n_features = self.X_train.shape
        # Store feature names
        self.feature_names = self.X_train.columns if isinstance(self.X_train, pd.DataFrame) else [f'Feature {i+1}' for i in range(self.n_features)]
        if len(self.feature_names) != self.n_features:
            self.logger.error(f"Feature names count ({len(self.feature_names)}) does not match X_train features ({self.n_features}).")
            raise ValueError("Feature names/X_train dimension mismatch.")
        
        # --- 3. Fit All Models on Training Data ---
        
        # A. Fit Scaler (if needed) - Use .values to get NumPy array
        X_train_values = self.X_train.values

        if self.descriptor_type == 'rdkit':
            self.logger.info("RDKit descriptors detected. Using StandardScaler.")
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(X_train_values) # Fit on array
        else: # 'fps'
            self.logger.info("FPS descriptors detected. Skipping scaling.")
            self.scaler = None
            # --- FIX: Use 0/1 integers, not booleans ---
            self.X_train_scaled = X_train_values 
            # --- END FIX ---
            
        # B. Fit PLS/PCA Model (always uses the processed array)
        self.logger.info(f"Fitting {self.model_type.upper()} model...")
        if self.model_type == 'pls':
            if self.n_components is None:
                self.n_components = min(self.n_samples, self.n_features, 10)
            self.pls = PLSRegression(n_components=self.n_components, scale=False)

            if self.y_train.ndim == 1:
                y_train_2d = self.y_train.values.reshape(-1, 1)
            else:
                y_train_2d = self.y_train.values

            self.pls.fit(self.X_train_scaled, y_train_2d) # Fit on array AND 2D y
            
            self.train_scores = self.pls.transform(self.X_train_scaled)
            self.model_ = self.pls
        
        elif self.model_type == 'pca':
            if self.n_components is None:
                self.n_components = min(self.n_samples, self.n_features, 10)
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
            self.pca.fit(self.X_train_scaled) # Fit on array
            self.train_scores = self.pca.transform(self.X_train_scaled)
            self.model_ = self.pca
        
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Must be 'pls' or 'pca'.")

        # C. Calculate and Store Baseline Variances & Covariances
        self.logger.info("Calculating baseline residual variances for DModX...")
        self.train_residual_variances_ = self._calculate_residual_variances(
            data=self.X_train_scaled,
            model=self.model_,
            scores=self.train_scores
        )
        self.logger.info("Calculating baseline inverse covariance matrix for T²...")
        self.train_inv_cov_ = self._calculate_inversed_covariance_matrix(self.train_scores)
        self.train_scores_mean_ = np.mean(self.train_scores, axis=0) # Store training centroid

         # D. Fit k-NN Model (on scores)
        self.knn_model = NearestNeighbors(n_neighbors=self.k_knn, metric='euclidean')
        self.knn_model.fit(self.train_scores) # Fit on scores

        # E. Fit LOF Model (on scores)
        self.lof_model = LocalOutlierFactor(
            n_neighbors=self.k_knn, 
            novelty=True, 
            contamination=self.alpha_percentile,
            metric='euclidean' # Always euclidean for scores
        )
        self.lof_model.fit(self.train_scores) # Fit on scores

        # F. Fit KDE Model (on scores)
        # We use the scores because they are lower dimensional and continuous.
        # If bandwidth is not provided, use a heuristic (Scott's Rule approximation)
        if self.kde_bandwidth is None:
            # Scott's rule: n**(-1./(d+4))
            d = self.train_scores.shape[1]
            n = self.train_scores.shape[0]
            self.kde_bandwidth = n**(-1./(d+4))
        
        self.kde_model = KernelDensity(bandwidth=self.kde_bandwidth, kernel='gaussian')
        self.kde_model.fit(self.train_scores)

        # G. Fit Isolation Forest (NEW)
        # We fit this on Scores to find "model-space" anomalies
        self.iforest_model = IsolationForest(
            contamination=self.alpha_percentile,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.iforest_model.fit(self.train_scores)

        # --- Feature Space Models (Autoencoder) ---
        self.logger.info("Fitting Feature-Space AD models (Autoencoder)...")
        
        # H. Fit Autoencoder (Reconstruction Error) (NEW)
        # We fit this on scaled features to find "feature-space" anomalies
        # Simple bottleneck architecture: Input -> [n_components] -> Output
        # This forces non-linear compression similar to PCA but with ReLU

        # Solver choice based on data size. If dataset is big, use 'adam' for scalability. Else 'lbfgs'.
        if len(self.X_train_scaled) >= 1000:
            solver = 'adam'
        else:
            solver = 'lbfgs'

        bottleneck_size = max(1, self.n_components)
        self.ae_model = MLPRegressor(
            hidden_layer_sizes=(bottleneck_size,),
            activation='relu',
            solver=solver,
            alpha=0.0001,
            max_iter=2000,
            random_state=self.random_state
        )
        # Fit X to X
        self.ae_model.fit(self.X_train_scaled, self.X_train_scaled)

        self.logger.info(f"All AD models fitted.")
        self.logger.info(f"Params: Alpha={self.alpha_percentile}, KDE_BW={self.kde_bandwidth:.4f}, AE_Hidden={bottleneck_size}")

        # --- 4. Calculate and Store Raw Metrics for the Training Set ---
        self.logger.info("Calculating raw metrics for internal training set...")
        self.train_metrics_raw_df = self.get_raw_metrics(self.X_train, self.y_train, is_internal=True)
        self.logger.info("ADTest engine initialized and ready.")
    
    def get_raw_metrics(self, X_query: np.ndarray, y_query: pd.Series, is_internal: bool = False) -> pd.DataFrame:
        """
        Calculates all 9 raw AD metrics for a new query set (X_query, y_query)
        by calling individual helper methods.
        
        This is the "query engine" or "conductor" of the class.
        """
        if X_query is None or X_query.shape[0] == 0:
            self.logger.warning("get_raw_metrics received empty X_query. Returning empty DataFrame.")
            return pd.DataFrame(index=y_query.index)

        self.logger.debug(f"Calculating raw metrics for {X_query.shape[0]} query compounds...")
        
        # Capture the index for the final output
        if hasattr(y_query, 'index'):
            final_index = y_query.index
        elif hasattr(X_query, 'index'):
            final_index = X_query.index
        else:
            final_index = pd.RangeIndex(len(y_query))
        
        # --- 1. Pre-process Query Data (Scale or Convert Type) ---
        X_query_values = X_query.values if isinstance(X_query, pd.DataFrame) else X_query
        y_query_values = y_query.values if hasattr(y_query, 'values') else y_query
        y_query_values = y_query_values.ravel() # Ensure 1D array

        if self.scaler is not None: # 'rdkit'
            X_query_processed = self.scaler.transform(X_query_values)
        else: # 'fps'
            X_query_processed = X_query_values
        
        if self.model_type == 'pls':
            query_scores = self.pls.transform(X_query_processed)
        else: # pca
            query_scores = self.pca.transform(X_query_processed)

        # --- Metrics on Scores ---
        raw_dmodx = self._calculate_dmodx_raw(query_scores, X_query_processed)
        raw_t2 = self._calculate_t2_raw(query_scores, self.train_scores_mean_, self.train_inv_cov_)
        raw_lof = self._calculate_lof_raw(query_scores)
        raw_knn = self._calculate_knn_raw(query_scores, is_internal)
        raw_kde = self._calculate_kde_raw(query_scores)
        raw_iforest = self._calculate_isolation_forest_raw(query_scores) # NEW

        # --- Metrics on Features ---
        raw_ae_error = self._calculate_reconstruction_error_raw(X_query_processed) # NEW

        # ADSal
        X_query_adsal = X_query_processed
        X_ref_adsal = self.X_train_scaled

        adsal_metrics_df = self._calculate_adsal(
            X_query=X_query_adsal, 
            y_query=y_query_values, 
            X_ref=X_ref_adsal, 
            y_ref=self.y_train, 
            k=self.k_knn, 
            alpha_exponent=self.alpha_exponent_adsal, 
            is_internal=is_internal
        )

        # --- Combine All Metrics ---
        raw_metrics_df = pd.DataFrame(
            {
                'DModX': raw_dmodx,
                'T2': raw_t2,
                'LOF_Score': raw_lof,
                'kNN_Dist': raw_knn,
                'KDE_LL': raw_kde,
                'IForest_Score': raw_iforest,
                'AE_Recon_Error': raw_ae_error,
            },
            index=final_index
        )
        
        # Make sure adsal_metrics_df has the same index
        adsal_metrics_df.index = final_index

        raw_metrics_df = raw_metrics_df.join(adsal_metrics_df)
        
        return raw_metrics_df
    
    def get_final_thresholds(self) -> Dict[str, float]:
        """
        Calculates the final AD thresholds based on the specified alpha percentile
        using the distribution of the internal training set metrics.
        
        This is the "rule setter" of the class.
        """
        if self.train_metrics_raw_df is None or self.train_metrics_raw_df.empty:
            self.logger.error("Training set metrics are not calculated. Cannot get thresholds.")
            self.train_metrics_raw_df = self.get_raw_metrics(self.X_train, self.y_train, is_internal=True)
            if self.train_metrics_raw_df.empty:
                 return {}

        if not (0 < self.alpha_percentile < 1):
            self.logger.warning(f"alpha_percentile ({self.alpha_percentile}) is not between 0 and 1. Defaulting to 0.05.")
            self.alpha_percentile = 0.05
            
        low_percentile = self.alpha_percentile * 100.0
        high_percentile = 100.0 - low_percentile

        self.logger.info(f"Calculating thresholds using {low_percentile}th/{high_percentile}th percentiles.")

        thresholds = {}

        # "High is bad" metrics: Use high percentile
        for metric in ['DModX', 'T2', 'kNN_Dist', 'Ii_Activity_Inconsistency', 'AE_Recon_Error']:
            thresholds[f'{metric}_thresh'] = np.percentile(self.train_metrics_raw_df[metric].dropna(), high_percentile)

        # "Low is bad" metrics: Use low percentile
        for metric in ['LOF_Score', 'Di_Similarity_Density', 'KDE_LL','IForest_Score']:
            if metric == 'LOF_Score':
                thresholds[f'{metric}_thresh'] = self.lof_model.offset_
            else:
                thresholds[f'{metric}_thresh'] = np.percentile(self.train_metrics_raw_df[metric].dropna(), low_percentile)
        
        return thresholds

    #
    # --- CORE METRIC & HELPER METHODS ---
    #

    # --- Isolation Forest Helper ---

    def _calculate_isolation_forest_raw(self, query_scores: np.ndarray) -> np.ndarray:
        """
        Calculates Isolation Forest scores.
        
        Standard iForest: Lower is more anomalous (negative).
        We return: decision_function.
        So: Low Value = More Anomalous (Bad).
        """

        # decision_function returns anomaly score. Lower = more abnormal.

        scores = self.iforest_model.decision_function(query_scores)

        return scores

    # --- Autoencoder Helper ---

    def _calculate_reconstruction_error_raw(self, X_query_processed: np.ndarray) -> np.ndarray:
        """
        Calculates Reconstruction Error (MSE) using the Autoencoder.
        
        High MSE = Poor reconstruction = Outlier (Bad).
        """

        X_pred = self.ae_model.predict(X_query_processed)
        # Calculate Mean Squared Error per sample
        mse = np.mean((X_query_processed - X_pred)**2, axis=1)

        return mse
    
    # --- KDE Helper ---

    def _calculate_kde_raw(self, query_scores: np.ndarray) -> np.ndarray:
        """
        Calculates the Log-Likelihood of the query samples.
        
        We return Log-Likelihood ( log(density) ) so that:
        - Low values = Low density = Outlier (Bad)
        - High values = High density = Inlier (Good)
        
        This aligns with LOF and Di_Similarity_Density.
        """

        # score_samples returns log(density)
        log_density = self.kde_model.score_samples(query_scores)

        return log_density
    
    # --- T² (Hotelling's) Helpers ---

    def _calculate_t2_raw(self, query_scores: np.ndarray, train_centroid: np.ndarray, train_inv_cov: np.ndarray) -> np.ndarray:
        """
        Private helper to calculate raw T² (squared Mahalanobis) values.
        """
        # T² is squared Mahalanobis distance from the center.
        raw_t2_scores = self._calculate_mahalanobis_from_centroid(
            scores=query_scores,
            centroid=train_centroid,
            inv_cov=train_inv_cov)**2 # T² is the squared distance
        
        return raw_t2_scores

    # --- DModX Helpers ---
    
    def _calculate_residual_variances(self, data, model, scores):
        """
        Calculates the residual variances from the training data.
        This should ONLY be called with training data.
        """
        if isinstance(model, PCA):
            X_reconstructed = model.inverse_transform(scores)
            residual = data - X_reconstructed
        elif isinstance(model, PLSRegression):
            X_reconstructed = np.dot(scores, model.x_loadings_.T)
            residual = data - X_reconstructed
        else:
            self.logger.error("Model must be a PCA or PLSRegression instance.")
            raise TypeError("Model must be a PCA or PLSRegression instance.")
        
        residual_variances = np.var(residual, axis=0, ddof=1)
        
        epsilon = 1e-6
        residual_variances[residual_variances < epsilon] += epsilon

        return residual_variances

    def _calculate_dmodx_scores_with_variances(self, data, model, scores, residual_variances):
        """
        Calculates the DModX scores for a given dataset using pre-computed residual variances.
        """
        if isinstance(model, PCA):
            X_reconstructed = model.inverse_transform(scores)
            residual = data - X_reconstructed
        elif isinstance(model, PLSRegression):
            X_reconstructed = np.dot(scores, model.x_loadings_.T)
            residual = data - X_reconstructed
        else:
            self.logger.error("Model must be a PCA or PLSRegression instance.")
            raise TypeError("Model must be a PCA or PLSRegression instance.")

        dmodx_scores = np.sqrt(np.sum(residual**2 / residual_variances, axis=1))

        return dmodx_scores

    def _calculate_dmodx_raw(self, query_scores: np.ndarray, X_query_scaled: np.ndarray) -> np.ndarray:
        """
        Private helper to calculate weighted DModX values.
        """
        raw_dmodx = self._calculate_dmodx_scores_with_variances(
            data=X_query_scaled,
            model=self.model_,
            scores=query_scores,
            residual_variances=self.train_residual_variances_
        )
        return raw_dmodx
    
    # --- LOF Helper ---
    
    def _calculate_lof_raw(self, query_scores: np.ndarray) -> np.ndarray:
        """
        Private helper to calculate raw LOF scores for scaled query data.
        """
        return self.lof_model.score_samples(query_scores)

    # --- kNN Helper ---
    
    def _calculate_knn_raw(self, query_scores: np.ndarray, is_internal: bool = False) -> np.ndarray:
        """
        Private helper to calculate raw kNN distances for scaled query data.
        """

        # 1. Decide how many neighbors to ask for
        k_to_query = self.k_knn + 1 if is_internal else self.k_knn
        
        self.knn_model.set_params(n_neighbors=k_to_query)
        
        distances, _ = self.knn_model.kneighbors(query_scores)
        
        # 2. Calculate the mean distance
        if is_internal:
            # If internal (training), skip the first column (dist=0)
            raw_knn_dist = np.mean(distances[:, 1:], axis=1) 
        else:
            # If external (test), use all k neighbors
            raw_knn_dist = np.mean(distances, axis=1)
            
        return raw_knn_dist
    
    # --- ADSal Helper ---
    
    def _calculate_adsal(self, X_query: np.ndarray, y_query: pd.Series, X_ref: np.ndarray, y_ref: pd.Series, k: int, alpha_exponent: float, is_internal: bool = False) -> pd.DataFrame:
        """
        Private method to calculate ADSal metrics (Di, Ii).
        """

        if self.descriptor_type == 'rdkit':
            dist_matrix = cdist(X_query, X_ref, metric='euclidean')
            similarity_matrix = 1.0 / (1.0 + dist_matrix + 1e-6) # Add epsilon
        elif self.descriptor_type == 'fps':
            dist_matrix = cdist(X_query.astype(bool), X_ref.astype(bool), metric='jaccard')
            similarity_matrix = 1.0 - dist_matrix
        else:
            self.logger.error(f"Unsupported descriptor type '{self.descriptor_type}'")
            raise ValueError(f"Unsupported descriptor type '{self.descriptor_type}'")

        y_ref_values = y_ref.values if hasattr(y_ref, 'values') else y_ref

        Di_list, Ii_list = [], []

        for i in range(X_query.shape[0]):
            similarities_i = similarity_matrix[i, :]
            
            if is_internal:
                neighbor_indices = np.argsort(similarities_i)[- (k + 1):-1]
            else:
                neighbor_indices = np.argsort(similarities_i)[-k:]
            
            if len(neighbor_indices) == 0:
                Di_list.append(np.nan)
                Ii_list.append(np.nan)
                continue
                
            neighbor_sims = similarities_i[neighbor_indices]
            neighbor_activities = y_ref_values[neighbor_indices]
            
            weights = np.power(neighbor_sims, alpha_exponent) 
            
            sum_weights = np.sum(weights)
            if sum_weights == 0:
                Di_list.append(0)
                Ii_list.append(0)
                continue
            
            # 1. Calculate Di (Structural Density) - Always valid
            di = np.sum(weights * neighbor_sims) / sum_weights
            Di_list.append(di)
            
            # 2. Calculate Ii (Activity Inconsistency) - Handle -1 (Uncertainty)
            current_y = y_query[i]
            
            # Check for -1 (Uncertainty). We use isclose for float safety.
            if np.isclose(current_y, -1):
                # If prediction is uncertain, Inconsistency is undefined.
                # We return -1 so it isn't flagged as an 'Inconsistency Outlier',
                # but it will still be caught as a 'Prediction Error' downstream.
                Ii_list.append(-1)
            else:
                # Normal calculation for 0 and 1
                activity_diffs = np.abs(current_y - neighbor_activities)
                ii = np.sum(weights * activity_diffs) / sum_weights
                Ii_list.append(ii)

        adsal_df = pd.DataFrame({
            'Di_Similarity_Density': Di_list,
            'Ii_Activity_Inconsistency': Ii_list})

        return adsal_df
    
    # --- HELPER METHODS (SAL, MAHALANOBIS) ---

    def _calculate_sal(self):
        """
        Calculates the Structure-Activity Landscape (SAL) for the training set.
        (Original function provided by user, uses class attributes)
        """
        if self.y_train is None:
            self.logger.error("Cannot calculate SAL without a target variable (y_train).")
            return None, None
        if len(self.y_train.shape) > 1 and self.y_train.shape[1] > 1:
            self.logger.error("SAL calculation is only supported for a single target variable (y_train).")
            return None, None

        # Use scaled data for distance
        structural_dist_matrix = squareform(pdist(self.X_train_scaled, metric='euclidean'))
        
        y_train_2d = self.y_train.values.reshape(-1, 1)
        activity_dist_matrix = squareform(pdist(y_train_2d, metric='cityblock'))
        
        upper_triangle_indices = np.triu_indices(self.n_samples, k=1)
        
        threshold = np.quantile(structural_dist_matrix[upper_triangle_indices], 0.1)

        sal_df = pd.DataFrame({
            'Structural_Distance': structural_dist_matrix[upper_triangle_indices],
            'Activity_Distance': activity_dist_matrix[upper_triangle_indices]
        })
        
        return sal_df, threshold
    
    def _calculate_train_test_sal(self, X_test: np.ndarray, y_test: pd.Series):
        """
        Calculates the SAL between the training and test sets.
        (Refactored to accept X_test and y_test as arguments)
        """
        if self.y_train is None or y_test is None:
            self.logger.error("Cannot calculate train-test SAL without both y_train and y_test.")
            return None, None
        
        # Use scaled data for distances
        if self.scaler is not None: # 'rdkit'
            X_test_scaled = self.scaler.transform(X_test)
        else: # 'fps'
            X_test_scaled = X_test
        
        structural_dist_matrix = cdist(self.X_train_scaled, X_test_scaled, metric='euclidean')
        
        y_train_2d = self.y_train.values.reshape(-1, 1)
        y_test_2d = y_test.values.reshape(-1, 1)
        activity_dist_matrix = cdist(y_train_2d, y_test_2d, metric='cityblock')

        # Threshold is based on internal training set distances
        train_structural_dist_matrix = squareform(pdist(self.X_train_scaled, metric='euclidean')) # Bug fix: use scaled
        upper_triangle_indices = np.triu_indices(self.n_samples, k=1)
        train_structural_distances = train_structural_dist_matrix[upper_triangle_indices]
        
        threshold = np.quantile(train_structural_distances, 0.1)

        sal_df = pd.DataFrame({
             'Structural_Distance': structural_dist_matrix.flatten(),
            'Activity_Distance': activity_dist_matrix.flatten()
        })
        
        return sal_df, threshold

    def _calculate_inversed_covariance_matrix(self, data):
        """
        Helper to calculate the inversed covariance matrix.
        (Needed by _calculate_mahalanobis_from_class_centroids)
        """
        cov_matrix = np.cov(data, rowvar=False)
        # Use pseudo-inverse for stability in case of singular matrix
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

        return inv_cov_matrix
        
    def _calculate_mahalanobis_from_centroid(self, scores, centroid, inv_cov):
        """
        Helper function to calculate Mahalanobis distance from a centroid.
        (Original function provided by user)
        """

        distances = cdist(scores, centroid.reshape(1, -1), 'mahalanobis', VI=inv_cov).flatten()

        return distances

    def plot_pca(self):
        """
        Plots the first two principal components of the training data.
        """
        if self.model_type != 'pca':
            self.logger.warning("plot_pca called, but model_type is not 'pca'. Skipping.")
            return

        scores = self.train_scores
        avg_distances = self.train_metrics_raw_df['kNN_Dist']
        thresholds = self.get_final_thresholds()
        t2_crit = thresholds.get('T2_thresh', 0)
        
        self.plot.plot_pca(
            pca_scores=scores, 
            knn_density_scores=avg_distances,
            explained_variance_ratio=self.pca.explained_variance_ratio_, 
            hotelling_t2_crit_val=t2_crit
        )

    def plot_pca_train_test(self, X_test: np.ndarray, y_test: pd.Series):
        """
        Plots the first two principal components of the training and test data.
        """
        if self.model_type != 'pca':
            self.logger.warning("plot_pca_train_test called, but model_type is not 'pca'. Skipping.")
            return

        # --- Train Data (from class attributes) ---
        train_scores = self.train_scores
        train_avg_dist = self.train_metrics_raw_df['kNN_Dist']
        thresholds = self.get_final_thresholds()
        t2_crit = thresholds.get('T2_thresh', 0)
        
        # --- Test Data (calculated on the fly) ---
        test_metrics_raw_df = self.get_raw_metrics(X_test, y_test, is_internal=False)
        test_scores = self.pca.transform(self.scaler.transform(X_test)) # Get scores
        test_avg_dist = test_metrics_raw_df['kNN_Dist']

        self.plot.plot_pca_train_test(
            pca_train_scores=train_scores, y_train=self.y_train,
            pca_test_scores=test_scores, y_test=y_test,
            knn_density_scores_train=train_avg_dist, 
            knn_density_scores_test=test_avg_dist,
            explained_variance_ratio=self.pca.explained_variance_ratio_,
            hotelling_t2_crit_val=t2_crit
        )
    
    def plot_pls_tu(self, n_components: Optional[int] = None):
        """
        Plots PLS T-U plots for each component pair.
        """
        if self.model_type != 'pls':
            self.logger.warning("plot_pls_tu called, but model_type is not 'pls'. Skipping.")
            return
        
        components_to_use = self.n_components
        if n_components is not None:
            if n_components > self.n_components:
                self.logger.error(f"Selected components {n_components} > fitted components {self.n_components}.")
                return
            components_to_use = n_components
        
        t_scores = self.train_scores
        u_scores = self.pls.y_scores_ # This is stored by scikit-learn PLS

        for i in range(components_to_use):
            self.logger.info(f"Generating PLS T-U plot for Latent Variable {i+1}...")
            self.plot.plot_pls_tu(
                t_scores=t_scores,
                u_scores=u_scores,
                comp_x=i, 
                comp_y=i  # Original code plotted comp vs. itself
            )

    def plot_feature_importance_pca(self, top_n: Optional[int] = None):
        """
        Plots the feature importance based on PCA loadings.
        Now uses self.feature_names stored during __init__.
        """
        if self.model_type != 'pca':
            self.logger.warning("plot_feature_importance_pca called, but model_type is not 'pca'. Skipping.")
            return
        if self.pca is None:
            self.logger.error("PCA model not fitted. Cannot plot feature importance.")
            return

        loadings = self.pca.components_[:self.n_components, :].T
        loading_df = pd.DataFrame(loadings, index=self.feature_names, columns=[f'PC{i+1}' for i in range(self.n_components)])
        
        loading_df['Abs_Loading'] = loading_df.abs().sum(axis=1)
        sorted_loadings = loading_df.sort_values(by='Abs_Loading', ascending=False)

        if top_n is not None and isinstance(top_n, int):
            if top_n > len(sorted_loadings):
                self.logger.warning(f"Warning: `top_n` value ({top_n}) is greater than the number of features ({len(sorted_loadings)}). Plotting all features.")
            else:
                sorted_loadings = sorted_loadings.head(top_n)

        self.plot.plot_feature_importance_pca(sorted_loadings=sorted_loadings, n_components=self.n_components, top_n=top_n)

    def plot_feature_importance_pls(self, top_n: Optional[int] = None):
        """
        Plots the feature importance based on PLS VIP scores.
        Now uses self.feature_names stored during __init__.
        """
        if self.model_type != 'pls':
            self.logger.warning("plot_feature_importance_pls called, but model_type is not 'pls'. Skipping.")
            return
        if self.pls is None:
            self.logger.error("PLS model not fitted. Cannot plot VIP scores.")
            return

        T = self.train_scores        # (n_samples, n_components)
        W = self.pls.x_weights_     # (n_features, n_components)
        Q = self.pls.y_loadings_    # (n_targets, n_components)
        p, h = W.shape
        
        ssy_h = np.sum(T**2, axis=0) * np.sum(Q**2, axis=0) # Assumes 1 target, (n_components,)
        
        if np.sum(ssy_h) == 0:
            self.logger.error("Total sum of squares of Y is zero. Cannot calculate VIP.")
            return

        vip = np.sqrt(p * np.sum(W**2 * ssy_h, axis=1) / np.sum(ssy_h))
            
        vip_df = pd.DataFrame({'VIP': vip, 'Feature': self.feature_names}).sort_values(by='VIP', ascending=False)
        
        if top_n is not None and isinstance(top_n, int):
            if top_n > len(vip_df):
                self.logger.warning(f"Warning: `top_n` value ({top_n}) is greater than the number of features ({len(vip_df)}). Plotting all features.")
            else:
                vip_df = vip_df.head(top_n)

        self.plot.plot_feature_importance_pls_regression(vip_df=vip_df, top_n=top_n)

    def plot_sal(self):
        """
        Generates a Structure-Activity Landscape (SAL) plot for the training set.
        """
        sal_data, threshold = self._calculate_sal()
        if sal_data is None:
            return
    
        self.plot.plot_sal(sal_data=sal_data, threshold=threshold)

    def plot_sal_train_test(self, X_test: np.ndarray, y_test: pd.Series):
        """
        Generates a Structure-Activity Landscape (SAL) plot for training and test data.
        """
        sal_data, threshold = self._calculate_train_test_sal(X_test, y_test)
        if sal_data is None:
            return

        self.plot.plot_train_test_sal(sal_data=sal_data, threshold=threshold)
        
    def plot_sal_violin(self, X_test: np.ndarray, y_test: pd.Series):
        """
        Generates a Structure-Activity Landscape (SAL) plot with violin plots.
        """
        sal_data, threshold = self._calculate_train_test_sal(X_test, y_test)
        if sal_data is None:
            return
    
        self.plot.plot_sal_violin(sal_data=sal_data, threshold=threshold)

    def plot_adsal(self):
        """
        Generates and displays the ADSal plot for the TRAINING data.
        """
        metrics_df = self.train_metrics_raw_df
        if metrics_df is None or metrics_df.empty:
            self.logger.error("Training metrics not found. Cannot plot ADSal.")
            return
            
        thresholds = self.get_final_thresholds()

        Di = metrics_df['Di_Similarity_Density']
        Ii = metrics_df['Ii_Activity_Inconsistency']
        Di_thresh = thresholds.get('Di_Similarity_Density_thresh', 0)
        Ii_thresh = thresholds.get('Ii_Activity_Inconsistency_thresh', 0)

        is_outside = (Di < Di_thresh) | (Ii > Ii_thresh)
        
        report = {
            'x_data': Di,
            'y_data': Ii,
            'is_outside': is_outside,
            'x_label': 'Weighted Similarity Density ($D_i$)',
            'y_label': 'Weighted Activity Inconsistency ($I_i$)',
            'title': 'ADSal Train Set',
            'x_threshold': Di_thresh,
            'y_threshold': Ii_thresh,
            'ad_region_x': 'high', 
            'ad_region_y': 'low'
        }
        self.plot.plot_ad_scatter(report)
    
    def plot_adsal_test(self, test_df: pd.DataFrame, y_pred_qsar: np.ndarray):
        """
        Generates and displays the ADSal plot for the TRAINING data.
        """

        metrics_df = self.get_raw_metrics(test_df, y_pred_qsar, is_internal=False)
        if metrics_df is None or metrics_df.empty:
            self.logger.error("Test metrics not found. Cannot plot ADSal.")
            return
            
        thresholds = self.get_final_thresholds()

        Di = metrics_df['Di_Similarity_Density']
        Ii = metrics_df['Ii_Activity_Inconsistency']
        Di_thresh = thresholds.get('Di_Similarity_Density_thresh', 0)
        Ii_thresh = thresholds.get('Ii_Activity_Inconsistency_thresh', 0)

        is_outside = (Di < Di_thresh) | (Ii > Ii_thresh)
        
        report = {
            'x_data': Di,
            'y_data': Ii,
            'is_outside': is_outside,
            'x_label': 'Weighted Similarity Density ($D_i$)',
            'y_label': 'Weighted Activity Inconsistency ($I_i$)',
            'title': 'ADSal Test Set',
            'x_threshold': Di_thresh,
            'y_threshold': Ii_thresh,
            'ad_region_x': 'high', 
            'ad_region_y': 'low'
        }
        self.plot.plot_ad_scatter(report)

    def plot_distplot_metrics_with_thresholds(self, X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None):
        """
        Plots the distribution of various AD metrics with their thresholds.
        """
        # 1. Get train data and thresholds
        train_df = self.train_metrics_raw_df
        threshold_map = self.get_final_thresholds()
        plot_columns = self.train_metrics_raw_df.columns.tolist()
        
        # 2. Get test data (if provided)
        test_df = None
        if X_test is not None and y_test is not None:
            test_df = self.get_raw_metrics(X_test, y_test, is_internal=False)

        # 3. Call plotter
        if test_df is None:
            self.plot.generate_distplots_with_thresholds(
                all_results=train_df, 
                plot_columns=plot_columns, 
                threshold_map=threshold_map
            )
        else:
            self.plot.generate_distplots_with_thresholds_train_test(
                train_results=train_df, 
                test_results=test_df, 
                plot_columns=plot_columns, 
                threshold_map=threshold_map
            )
    
    #
    # --- HIGH-LEVEL WORKFLOW & REPORTING METHODS ---
    #
    
    def run_all_tests_external(self, X_test: np.ndarray, y_test: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates raw metrics for the internal train set and a provided external test set.
        To properly check for an error model, y_test must be the predicted activity values on the test series.
        """
        self.logger.info("run_all_tests_external() called: Calculating raw metrics for external test set...")
        
        # Train metrics are already calculated and stored in __init__
        train_results = self.train_metrics_raw_df
        
        # Calculate for the external test set
        test_results = self.get_raw_metrics(X_test, y_test, is_internal=False)
        
        self.logger.info("Raw metrics calculated for train and test sets.")
        
        return train_results, test_results

    ###### Reporting functions

    def _generate_unique_outlier_table(self, 
                                   results_df: pd.DataFrame, 
                                   y_test: pd.Series, 
                                   thresholds: Dict[str, float], 
                                   struc_df: pd.DataFrame,
                                   y_pred_qsar: Optional[pd.Series] = None) -> pd.DataFrame:
        """Generates a comprehensive outlier table with real activity, QSAR predictions, 
        and prediction errors using raw, unscaled metrics and thresholds."""

        if results_df is None or results_df.empty:
            return pd.DataFrame()

        # 1. Start with structural information
        outlier_table = struc_df[[self.id_col, self.smiles_col]].copy()

        # 2. Add real activity (Index-aligned assignment)
        if isinstance(y_test, pd.Series):
            outlier_table['Real_Activity'] = y_test.reindex(outlier_table.index)
        else:
            outlier_table['Real_Activity'] = y_test

        # 3. Add raw AD metrics
        outlier_table = outlier_table.join(results_df)

        # 4. Generate AD outlier flags based on raw metric directionality
        for col_name in results_df.columns:
            thresh_key = f"{col_name}_thresh"
            if thresh_key in thresholds:
                threshold = thresholds[thresh_key]
                if any(k in col_name for k in ['LOF_Score', 'Di_Similarity_Density', 'KDE_LL', 'IForest_Score']):
                    outlier_table[f"{col_name}_outlier"] = (results_df[col_name] < threshold).astype(int)
                else:
                    outlier_table[f"{col_name}_outlier"] = (results_df[col_name] > threshold).astype(int)

        # 5. Sum boolean flags
        outlier_cols = [col for col in outlier_table.columns if col.endswith('_outlier')]
        outlier_table['Total Metrics Flagged'] = outlier_table[outlier_cols].sum(axis=1)

        # 6. Add QSAR predictions and prediction errors (Index-aligned)
        if y_pred_qsar is not None:
            if isinstance(y_pred_qsar, pd.Series):
                outlier_table['QSAR_Prediction'] = y_pred_qsar.reindex(outlier_table.index)
            else:
                outlier_table['QSAR_Prediction'] = y_pred_qsar
                
            outlier_table['Prediction_Error'] = (
                outlier_table['Real_Activity'] != outlier_table['QSAR_Prediction']
            ).astype(int)
        else:
            outlier_table['QSAR_Prediction'] = np.nan
            outlier_table['Prediction_Error'] = np.nan

        return outlier_table

    def _calculate_outlier_summary_stats(self, outlier_df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calculates summary statistics from the 'Total Metrics Flagged' column.
        (This function is compatible with the new structure)
        """
        if outlier_df is None or outlier_df.empty or 'Total Metrics Flagged' not in outlier_df.columns:
            return None

        total_compounds = len(outlier_df)
        total_unique_outliers = (outlier_df['Total Metrics Flagged'] > 0).sum()
        percent_outliers = round(total_unique_outliers / total_compounds * 100, 2) if total_compounds > 0 else 0.0

        stats = {
            'Total Compounds': total_compounds,
            'Total Unique Outliers': total_unique_outliers,
            'Percent Unique Outliers': percent_outliers
        }

        flag_counts = outlier_df['Total Metrics Flagged'].value_counts()
        
        if total_unique_outliers > 0:
            max_flags = int(flag_counts.index.max()) # Ensure it's an int for range
            for i in range(1, max_flags + 1):
                 count = flag_counts.get(i, 0)
                 if count > 0:
                    stats[f'Flagged by {i} metric(s)'] = count

        return pd.Series(stats)

    def _calculate_exclusive_metric_flags(self, outlier_df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Identifies, for each AD metric, how many compounds it flags EXCLUSIVELY,
        i.e. compounds where 'Total Metrics Flagged' == 1 and this specific
        metric is the one that fired.

        This is complementary to '_calculate_outlier_summary_stats': that method
        reports HOW MANY metrics fire per compound (the multiplicity distribution),
        while this reports WHICH metric is responsible when only one fires
        (single-metric detection power / uniqueness).

        Args:
            outlier_df (pd.DataFrame): Outlier table with '<metric>_outlier' binary
                                        columns and a 'Total Metrics Flagged' column,
                                        as produced by '_generate_unique_outlier_table'.

        Returns:
            Optional[pd.Series]: Index = metric name, value = count of compounds
            exclusively flagged by that metric. Sorted descending. None if the
            required columns are absent.
        """
        if outlier_df is None or outlier_df.empty or 'Total Metrics Flagged' not in outlier_df.columns:
            return None

        outlier_cols = [c for c in outlier_df.columns if c.endswith('_outlier')]
        if not outlier_cols:
            return None

        # Restrict to compounds flagged by exactly one metric
        exclusive_subset = outlier_df.loc[outlier_df['Total Metrics Flagged'] == 1, outlier_cols]

        exclusive_counts = {
            col.replace('_outlier', ''): int(exclusive_subset[col].sum())
            for col in outlier_cols
            if exclusive_subset[col].sum() > 0
        }

        if not exclusive_counts:
            return pd.Series(dtype=int, name='Exclusive_Flags')

        return pd.Series(exclusive_counts, name='Exclusive_Flags').sort_values(ascending=False)

    # Building report dataframes

    def build_model_report(self):
        """
        Create the model report section containing model type, number of components,
        and key metric (explained variance for PCA or R-squared for PLS).

        Returns:
            pd.DataFrame: DataFrame indexed by parameter names with corresponding values.
        """
        model_name = self.model_type.upper()
        n_components = self.n_components
        if self.model_type == 'pca':
            explained_variance = round(np.cumsum(self.pca.explained_variance_ratio_)[n_components - 1] * 100, 2)
            model_report = pd.DataFrame({
                'Parameter': ['Model Type', 'Number of Components', 'Total Explained Variance'],
                'Value': [model_name, n_components, f"{explained_variance}%"]
            }).set_index('Parameter')
        else:  # 'pls'
            r2 = round(self.pls.score(self.X_train_scaled, self.y_train), 2)
            model_report = pd.DataFrame({
                'Parameter': ['Model Type', 'Number of Components', 'R-squared (Training)'],
                'Value': [model_name, n_components, r2]
            }).set_index('Parameter')
        return model_report

    def build_dataset_summary(self, X_test, y_test):
        """
        Generate a summary of training and test dataset sizes and class distributions.

        Args:
            X_test (Optional[pd.DataFrame]): Test feature data.
            y_test (Optional[pd.Series]): True target labels for test data.

        Returns:
            pd.DataFrame: Summary table indexed by 'Train' and 'Test' datasets with counts and percentages.
        """
        train_size = self.n_samples
        test_size = len(X_test) if X_test is not None else 0
        train_counts = self.y_train.value_counts()
        train_positives = train_counts.get(1, 0)
        train_negatives = train_counts.get(0, 0)
        positive_percentage_train = round(train_positives / train_size * 100, 2) if train_size > 0 else 0.0
        negative_percentage_train = round(train_negatives / train_size * 100, 2) if train_size > 0 else 0.0

        test_positives, test_negatives = 0, 0
        positive_percentage_test, negative_percentage_test = 0.0, 0.0
        if y_test is not None and not y_test.empty:
            test_counts = y_test.value_counts()
            test_positives = test_counts.get(1, 0)
            test_negatives = test_counts.get(0, 0)
            positive_percentage_test = round(test_positives / test_size * 100, 2) if test_size > 0 else 0.0
            negative_percentage_test = round(test_negatives / test_size * 100, 2) if test_size > 0 else 0.0

        dataset_summary = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'Total Compounds': [train_size, test_size],
            'Positive Class': [train_positives, test_positives],
            'Positive Class %': [positive_percentage_train, positive_percentage_test],
            'Negative Class': [train_negatives, test_negatives],
            'Negative Class %': [negative_percentage_train, negative_percentage_test]
        }).set_index('Dataset')
        return dataset_summary

    def build_metrics_report(self, thresholds):
        """
        Build a report DataFrame that describes the thresholds for AD metrics.

        Args:
            thresholds (dict): Dictionary mapping metric names to threshold values.

        Returns:
            pd.DataFrame: DataFrame indexed by metric with threshold info and operator.
        """
        metric_report_rows = []
        for col_name, thresh_val in thresholds.items():
            metric_name = col_name.replace('_thresh', '').replace('_', ' ')
            op = '<' if 'LOF' in metric_name or 'Di Similarity' in metric_name or 'KDE LL' in metric_name  or 'IForest Score' in metric_name else '>'
            metric_report_rows.append({
                'Metric': metric_name,
                'Operator': op,
                'Threshold': round(thresh_val, 3)
            })
        model_metrics_report = pd.DataFrame(metric_report_rows).set_index('Metric')
        return model_metrics_report

    def build_outlier_tables(self, X_test, y_test, test_struc, thresholds, y_pred_qsar) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate full outlier table and outlier summary for the test set.

        Args:
            X_test (pd.DataFrame): Test feature set.
            y_test (pd.Series): True target values for test set.
            test_struc (pd.DataFrame): Structural metadata (e.g., IDs, SMILES).
            thresholds (dict): Metric threshold values.
            y_pred_qsar (Optional[pd.Series]): Optional QSAR model predictions.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Full outlier DataFrame and summary DataFrame.
        """
        if X_test is None or y_test is None or test_struc is None:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        if y_pred_qsar is not None:
            test_results_df = self.get_raw_metrics(X_test, y_pred_qsar, is_internal=False)
        else:
            test_results_df = self.get_raw_metrics(X_test, y_test, is_internal=False)

        test_outlier_table_full = self._generate_unique_outlier_table(
            test_results_df, y_test, thresholds, test_struc, y_pred_qsar=y_pred_qsar)
        
        test_outlier_summary = self._calculate_outlier_summary_stats(test_outlier_table_full)

        test_outlier_summary_df = test_outlier_summary.to_frame(name='Value') if test_outlier_summary is not None else pd.DataFrame()

        return test_outlier_table_full, test_outlier_summary_df, test_results_df

    def _build_report_dict(self, X_test=None, y_test=None, test_struc=None, y_pred_qsar=None):
        """
        Assemble the complete Applicability Domain report dictionary by aggregating
        model details, dataset summaries, metric reports, and outlier tables.

        Args:
            X_test (Optional[np.ndarray]): Test descriptors/features.
            y_test (Optional[pd.Series]): True target labels for the test set.
            test_struc (Optional[pd.DataFrame]): Structural data, e.g., compound IDs and SMILES.
            y_pred_qsar (Optional[pd.Series]): Optional QSAR model predictions for test set.

        Returns:
            dict: Dictionary containing all report sections as pandas DataFrames.
        """
        self.logger.info(f"Building report data structure for {self.model_type}...")
        
        # 1. Get raw thresholds
        thresholds = self.get_final_thresholds()
        
        # 2. Build the standard outlier tables (Raw Metrics)
        test_outlier_table_full, test_outlier_summary_df, test_results_df = self.build_outlier_tables(
            X_test, y_test, test_struc, thresholds, y_pred_qsar
        )

        return {
            'model_report': self.build_model_report(),
            'dataset_summary': self.build_dataset_summary(X_test, y_test),
            'metrics_report': self.build_metrics_report(thresholds),
            'test_outlier_table_full': test_outlier_table_full,
            'test_outlier_summary': test_outlier_summary_df,
            'smiles_col': self.smiles_col
        }

    def generate_ad_report(self, X_test: np.ndarray, y_test: pd.Series, test_struc: pd.DataFrame, y_pred_qsar: Optional[pd.Series] = None):
        """
        Generates a summary report of the AD analysis in pandas DataFrames.
        [REFACTORED to use new class structure]
        """
        self.logger.info(f"Generating main report data structure for model_type: {self.model_type}...")
        
        # Call the new helper with the results from this run
        return self._build_report_dict(
            X_test=X_test,
            y_test=y_test,
            test_struc=test_struc,
            y_pred_qsar=y_pred_qsar)
    
    # Helper function to print a title and its corresponding DataFrame
    def print_section(self, title: str, df: pd.DataFrame):
        """
        Prints a title and its corresponding DataFrame to the console in a formatted manner.
        """
        if df is None:
            print(f"\n{title}\n{'-'*len(title)}\n (No data)\n{'-'*len(title)}")
            return
            
        print(f"\n{title}")
        print("-" * len(title))
        
        if df.index.names and any(name is not None for name in df.index.names):
            print(df.reset_index().to_markdown(index=False, numalign="left", stralign="left"))
        else:
            print(df.to_markdown(numalign="left", stralign="left"))
        print("-" * len(title))

    def print_ad_report_to_console(self, report: Dict[str, Any]) -> None:
        """
        Formats and prints the Applicability Domain (AD) analysis report from 
        a dictionary containing pandas DataFrames directly to the console.
        [REFACTORED for new report structure]
        """
        
        required_keys = ['model_report', 'dataset_summary', 'metrics_report']
        if not all(key in report for key in required_keys):
            print("Error: The report dictionary is missing required keys: 'model_report', 'dataset_summary', or 'metrics_report'.")
            return

        print("\n" + "="*60)
        print(f"  APPLICABILITY DOMAIN (AD) ANALYSIS REPORT ({self.model_type.upper()})")
        print("="*60)
        
        self.print_section("AD Model Parameters", report['model_report'])
        self.print_section("Dataset Summary", report['dataset_summary'])
        self.print_section(f"AD Metrics Thresholds ({self.model_type.upper()})", report['metrics_report'])
        self.print_section(f"Test Set Outlier Summary", report.get('test_outlier_summary'))
        
        test_outlier_full = report.get('test_outlier_table_full')
        if test_outlier_full is not None:
             self.logger.info(f"Full test outlier table ({len(test_outlier_full)} rows) generated (not shown in console).")
        
        print("\nReport printing complete.")