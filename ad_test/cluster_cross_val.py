"""
This module implements a robust cross-validation strategy that combines
clustering and stratification to ensure chemically diverse and balanced folds.

Author: Eric March (eric.march@upf.edu)
Created: 2025-09-17
"""

import numpy as np
import pandas as pd

from kneed import KneeLocator
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from scipy.cluster.hierarchy import linkage
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering, DBSCAN
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import silhouette_score, pairwise_distances
from typing import Optional, Union, Generator, Tuple, List

from .plots.plots import Plotter
from .utils.logger import LOG

class ClusteringCrossValidator:
    """
    A robust cross-validator that generates folds based on clustering
    and ensures stratification by class label.

    This class first performs clustering to define groups of
    chemically similar compounds and then uses scikit-learn's
    StratifiedGroupKFold to generate the folds. It supports various
    clustering methods, including structural grouping by Murcko scaffolds
    and dissimilarity-based selection using MaxMin.
    """

    def __init__(self, n_splits: int = 5, n_clusters: Optional[int] = None, method: str = 'kmeans',
                 random_state: int = 42, 
                 df: Optional[pd.DataFrame] = None, smiles_column: Optional[str] = None,
                 logger: Optional[LOG] = None):
        """
        Initializes the cross-validator.
        
        Args for all methods:
            method (str): The clustering algorithm to use. Options are 'kmeans',
                          'hierarchical', 'dbscan', or 'scaffolds'. Defaults to 'kmeans'.
            n_clusters (int, optional): The number of clusters to create. If None,
                                        the 'auto_split' method will attempt to find the optimal k.
            n_splits (int): The number of splits (folds) for cross-validation.
            random_state (int, optional): Seed for reproducibility. Defaults to 42.
            
         Args specifically for 'scaffolds' or 'maxmin':
            df (pd.DataFrame, optional): The DataFrame containing SMILES strings.
                                         Required if method='scaffolds' or 'maxmin'.
            smiles_column (str, optional): The column name with SMILES strings in the DataFrame.
                                           Required if method='scaffolds' or 'maxmin'.
        """
        
        if logger:
            # If a logger is provided (by main.py), use it.
            self.logger = logger
        else:
            # If no logger is provided, create a default one for interactive use.
            # This makes the class usable in a Jupyter notebook without extra setup.
            self.logger = LOG(name=self.__class__.__name__, log_file_path='cluster_cv_interactive.log')
            self.logger.info("No logger provided; created default logger for interactive session.")

        self.logger.info(f"Initializing ClusteringCrossValidator with method: {method}.")

        # Validate inputs
        method = method.lower()
        
        if method in ['scaffolds', 'maxmin'] and (df is None or smiles_column is None):
            self.logger.error(f"For '{method}' clustering, 'df' and 'smiles_column' must be provided during initialization.")
            raise ValueError(f"For '{method}' clustering, 'df' and 'smiles_column' must be provided during initialization.")
        
        self.method = method
        self.n_clusters = n_clusters
        self.n_splits = n_splits
        self.random_state = random_state
        self.labels_ = None
        
        # StratifiedGroupKFold requires n_splits validation to happen after n_clusters is finalized
        self.sgkf = StratifiedGroupKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        self.clusterer = None
        
        # Store df and smiles_column internally for 'scaffolds' method
        self.df = df
        self.smiles_column = smiles_column

        self.plot = Plotter()

### Clustering methods

    def _scaffold_clustering(self):
        """
        Generates Murcko scaffold SMILES strings.
        
        If n_clusters is None:
            Assigns a unique integer label to each unique scaffold. 
            This allows StratifiedGroupKFold to handle the balancing (Bin Packing).
        If n_clusters is set:
            Clusters the scaffolds by structural similarity (fingerprints + KMeans).
        """
        
        df = self.df
        smiles_column = self.smiles_column

        if isinstance(df, pd.DataFrame):
            self.logger.info(f"Generating Murcko scaffolds for all compounds from column '{smiles_column}'...")
            smiles_list = df[smiles_column]
        elif isinstance(df, pd.Series):
            self.logger.info("Generating Murcko scaffolds for all compounds from input Series...")
            smiles_list = df
        else:
            self.logger.error("Internal data storage for 'scaffolds' is invalid.")
            raise TypeError("Internal data storage for 'scaffolds' is invalid.")
        
        scaffolds = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
                    scaffolds.append(scaffold_smiles)
                else:
                    scaffolds.append("INVALID")
            except Exception:
                scaffolds.append("INVALID")
        
        scaffold_series = pd.Series(scaffolds)

        # --- MODIFIED LOGIC START ---
        if self.n_clusters is None:
            # OPTION A: Granular Scaffolds (Best for Balanced Disjoint Folds)
            self.logger.info("n_clusters is None. Using raw scaffolds as group labels.")
            self.logger.info(f"Found {scaffold_series.nunique()} unique scaffolds.")
            
            # Map each unique scaffold string to a unique integer ID
            # This creates hundreds of small 'groups' for StratifiedGroupKFold to bin-pack
            self.labels_ = scaffold_series.factorize()[0]
        else:
            # OPTION B: Cluster the scaffolds (Old behavior)
            self._cluster_scaffolds_by_similarity(scaffold_series)
        # --- MODIFIED LOGIC END ---

    def _cluster_scaffolds_by_similarity(self, scaffold_series: pd.Series):
        """
        Clusters unique scaffolds into 'Super-Families' to ensure structural disjointness.
        Uses Agglomerative Clustering (Average Linkage) to force distinct chemical spaces apart.
        """
        if self.n_clusters is None:
            # If user didn't specify, default to 3x the number of splits.
            # This gives the splitter enough 'large chunks' to balance sizes,
            # but keeps the chunks large enough to prevent leakage.
            suggested_k = self.n_splits * 10
            self.logger.info(f"n_clusters not set. Defaulting to {suggested_k} scaffold super-families.")
            self.n_clusters = suggested_k

        unique_scaffolds = scaffold_series.unique()
        self.logger.info(f"Grouping {len(unique_scaffolds)} unique scaffolds into {self.n_clusters} chemical super-families...")
        
        # 1. Generate Fingerprints for Scaffolds
        # We use a smaller bit vector (1024) for scaffolds as they are simpler structures
        scaffold_fps = []
        valid_scaffolds = []
        for smi in unique_scaffolds:
            if smi != "INVALID":
                m = Chem.MolFromSmiles(smi)
                if m:
                    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
                    scaffold_fps.append(fp)
                    valid_scaffolds.append(smi)
        
        # 2. Compute Distance Matrix (Jaccard/Tanimoto)
        # Convert to numpy for sklearn        
        X_scaff = []
        for fp in scaffold_fps:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            X_scaff.append(arr)
        X_scaff = np.array(X_scaff)
        
        # Distance = 1 - Similarity
        dist_matrix = pairwise_distances(X_scaff, metric='jaccard')
        
        n_valid_scaffolds = len(valid_scaffolds)
        if n_valid_scaffolds < self.n_clusters:
            self.logger.warning(
                f"⚠️ Requested {self.n_clusters} clusters, but only found {n_valid_scaffolds} valid unique scaffolds. "
                f"Capping n_clusters to {n_valid_scaffolds}."
            )
            self.n_clusters = n_valid_scaffolds
        
        # 3. Hierarchical Clustering
        # Linkage='average' maximizes the mean distance between clusters.
        # This is better than 'ward' for chemical distinctness.
        clusterer = AgglomerativeClustering(
            n_clusters=self.n_clusters, 
            metric='precomputed', 
            linkage='average' 
        )
        labels = clusterer.fit_predict(dist_matrix)
        
        # 4. Map back to original dataframe
        scaffold_to_label = dict(zip(valid_scaffolds, labels))
        
        # Any invalid scaffolds get label -1
        self.labels_ = scaffold_series.apply(lambda x: scaffold_to_label.get(x, -1)).values
        
        self.logger.info(f"Scaffold grouping complete. Created {self.n_clusters} distinct scaffold families.")

    def _report_cluster_sizes(self, y: Optional[np.ndarray] = None):
        """
        Calculates and prints a detailed summary of the cluster sizes, including
        a breakdown by class label and percentages if y is provided.
        """
        if self.labels_ is None:
            return

        total_samples = len(self.labels_)
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        
        # Create a summary DataFrame for easier handling
        summary_df = pd.DataFrame({'label': unique_labels, 'total_count': counts})
        
        # If y is provided, calculate class distribution within each cluster
        class_breakdown = None
        if y is not None:
            class_breakdown = pd.DataFrame({'label': self.labels_, 'class': y}).groupby('label')['class'].value_counts().unstack(fill_value=0)
            summary_df = summary_df.merge(class_breakdown, on='label', how='left').fillna(0)

        # Determine the number of actual clusters (excluding noise for DBSCAN/invalid SMILES)
        # --- MODIFIED --- Also check for -1 from maxmin/scaffolds
        if self.method in ['dbscan', 'maxmin', 'scaffolds'] or 'MISSING' in unique_labels:
            non_noise_labels = [l for l in unique_labels if l not in [-1, 'MISSING', 'INVALID']]
            num_clusters = len(non_noise_labels)
        else:
            # For object/string labels (ATC), nunique is safer
            num_clusters = pd.Series(self.labels_).nunique()

        self.logger.info(f"{self.method.upper()} clustering complete. Found {num_clusters} clusters.")
        self.logger.info("Compound count per cluster:")

        # Sort summary for consistent output
        try:
            summary_df = summary_df.sort_values(by='label')
        except TypeError:
            summary_df['label_str'] = summary_df['label'].astype(str)
            summary_df = summary_df.sort_values(by='label_str').drop(columns='label_str')

        for _, row in summary_df.iterrows():
            label = row['label']
            total_count = int(row['total_count'])
            
            # --- MODIFIED --- Handle -1 label for multiple methods
            if label == -1:
                label_str = "Noise/Invalid"
            elif label == "MISSING":
                 label_str = "MISSING ATC"
            else:
                 label_str = f"Cluster {label}"
            
            # Build the output string with total percentage
            percentage_total = (total_count / total_samples) * 100
            output_str = f"  - {label_str}: {total_count} compounds ({percentage_total:.1f}%)"
            
            if y is not None and class_breakdown is not None:
                class_counts_str = []
                for c in sorted(class_breakdown.columns):
                    class_count = int(row.get(c, 0))
                    # Calculate percentage within the cluster
                    percentage_class = (class_count / total_count) * 100 if total_count > 0 else 0
                    class_counts_str.append(f"Class {int(c)}: {class_count} ({percentage_class:.1f}%)")
                
                if class_counts_str:
                    output_str += f" ({', '.join(class_counts_str)})"
            
            # self.logger.info(output_str)
    
    def fit_clustering(self, scores: Optional[np.ndarray] = None, y=None, **kwargs):
        """
        Performs the actual clustering based on the specified method and parameters.
        Sets self.labels_.
        """
        
        if self.method == 'scaffolds': 
            self._scaffold_clustering()

        self._report_cluster_sizes(y)

    def split(self, X: np.ndarray, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates indices for the training and test sets for each fold.
        """

        if y is None:
            self.logger.error("Class labels 'y' must be provided for stratification.")
            raise ValueError("Class labels 'y' must be provided for stratification.")
        if self.labels_ is None:
            self.logger.error("Clustering must be performed (using fit_clustering or auto_split) before calling split().")
            raise RuntimeError("Clustering must be performed (using fit_clustering or auto_split) before calling split().")
        
        # Re-check n_splits vs unique groups just before splitting
        # Robustly determine the number of unique groups before checking against n_splits
        if self.labels_.dtype == object:
            # For string labels (ATC/Scaffolds), count all unique strings
            unique_groups_set = set(np.unique(self.labels_))
            # Remove any 'missing' or 'invalid' type labels from the count
            unique_groups_set.discard("MISSING")
            unique_groups_set.discard("INVALID")
            unique_groups = len(unique_groups_set)
        else:
            # For numerical labels (KMeans/Hierarchical/DBSCAN/MaxMin), filter out noise points (-1)
            unique_groups = len(np.unique(self.labels_[self.labels_ >= 0]))
        
        if unique_groups < self.n_splits:
            self.logger.warning(f"Warning: Number of unique non-noise groups ({unique_groups}) is less than n_splits ({self.n_splits}).")
            self.logger.info("Reducing n_splits to the number of unique groups.")
            self.n_splits = unique_groups
            # Reinitialize the splitter with the new, lower n_splits
            self.sgkf = StratifiedGroupKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
            
        yield from self.sgkf.split(X, y, groups=self.labels_)

    def auto_split(self, scores: np.ndarray, y: np.ndarray, df: Optional[pd.DataFrame] = None, smiles_column: Optional[str] = None, **kwargs) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Orchestrates parameter finding (if needed), clustering, and split generation
        in a single call. This replaces the multi-step notebook process.

        1. Performs clustering to generate group labels.
        2. Generates and yields the StratifiedGroupKFold splits.
        """
        self.logger.info(f"\n--- Starting Auto-Split Workflow for method: {self.method.upper()} ---")

        if self.method in ['scaffolds']:
            if df is not None:
                self.df = df
            if smiles_column is not None:
                self.smiles_column = smiles_column
            
            if self.df is None or self.smiles_column is None:
                self.logger.error(f"For '{self.method}' method, the 'df' and 'smiles_column' must be provided either during initialization or via the 'auto_split' call.")
                raise ValueError(f"For '{self.method}' method, the 'df' and 'smiles_column' must be provided either during initialization or via the 'auto_split' call.")

        if self.labels_ is None:
            self.fit_clustering(scores=scores, y=y, **kwargs)
        else:
            self._report_cluster_sizes(y)

        # --- Validate n_splits vs n_clusters ---
        # This check is crucial *after* n_clusters has been determined (either by init or auto-finding).
        if self.method in ['scaffolds'] and self.n_clusters is not None and self.n_splits > self.n_clusters:
            self.logger.warning(f"Warning: n_splits ({self.n_splits}) cannot be greater than n_clusters ({self.n_clusters}). Setting n_splits = {self.n_clusters}.")
            self.n_splits = self.n_clusters
            self.sgkf = StratifiedGroupKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)

        # --- 3. Generate Splits ---
        self.logger.info(f"\nGenerating {self.n_splits} stratified folds based on cluster groups...")
        yield from self.split(scores, y)

    def get_cluster_summary(self, X: pd.DataFrame, y: pd.Series) -> Optional[pd.DataFrame]:
        """
        Performs clustering and returns a DataFrame with a summary of each cluster.
        """
        self.logger.info(f"--- Generating Cluster Summary for method: {self.method} ---")
        
        # Ensure clustering has been performed
        if self.labels_ is None:
            self.fit_clustering(X, y)
        
        if self.labels_ is None:
            self.logger.error("Clustering failed, cannot generate summary.")
            return None

        summary_data = []
        total_compounds = len(y)
        
        # Create a temporary DataFrame for easier analysis
        temp_df = pd.DataFrame({'cluster': self.labels_, 'activity': y})
        
        # Get unique cluster labels, handling potential noise cluster (-1) from DBSCAN
        unique_labels = sorted(temp_df['cluster'].unique())

        for label in unique_labels:
            # --- MODIFIED --- Handle -1 and string labels
            if label == -1:
                cluster_name = "Noise/Invalid"
            elif isinstance(label, str):
                cluster_name = label
            else:
                cluster_name = f"Cluster {label}"
            
            cluster_df = temp_df[temp_df['cluster'] == label]
            
            n_compounds = len(cluster_df)
            percent_total = round(n_compounds / total_compounds * 100, 2)
            
            class_counts = cluster_df['activity'].value_counts().to_dict()
            class_0_count = class_counts.get(0, 0)
            class_1_count = class_counts.get(1, 0)
            
            percent_class_0 = round(class_0_count / n_compounds * 100, 2) if n_compounds > 0 else 0.0
            percent_class_1 = round(class_1_count / n_compounds * 100, 2) if n_compounds > 0 else 0.0

            summary_data.append({
                'Cluster': cluster_name,
                'Num Compounds': n_compounds,
                '% Total': percent_total,
                'Negative Class': class_0_count,
                'Negative Class %': percent_class_0,
                'Positive Class': class_1_count,
                'Positive Class %': percent_class_1
            })
            
        summary_df = pd.DataFrame(summary_data).set_index('Cluster')
        self.logger.info("Cluster summary generated successfully.")
        
        return summary_df