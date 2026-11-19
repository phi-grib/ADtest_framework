"""
This module defines the Plotter class for visualizing results.

Author: Eric March Vila (eric.march@upf.edu)
Created: 2025-09-01
Refactored: 2026-08-11
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from typing import Optional, Any, Tuple

from ..utils.logger import LOG

class Plotter(object):
    def __init__(self, report_base_path: Optional[str] = None, logger = None):
        """
        Initializes the Plotter class for visualizing dimensionality reduction and outlier detection results.
        """

        self.logger = logger if logger else LOG(name=self.__class__.__name__, log_file_path='plots.log')
        
        # Suppress RDKit logging output during SVG generation
        warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')

        self.report_base_path = report_base_path

    def _handle_save_or_show(self, plot_name: str):
        """
        Private helper to either save the current matplotlib plot or show it.
        """
        if self.report_base_path:
            save_path = f"{self.report_base_path}_{plot_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            # We need to return the path for the main script to collect
            return save_path
        else:
            plt.show()
            plt.close()
            return None
        
    def plot_pca(self, pca_scores, knn_density_scores, explained_variance_ratio, hotelling_t2_crit_val):
            """
            Plots the first two principal components of the data.
            
            Args:
                pca_scores (np.ndarray): The PCA scores of the data.
                knn_density_scores (np.ndarray): The kNN density scores for each data point.
                explained_variance_ratio (list or np.ndarray): The explained variance ratio for each principal component.
                hotelling_t2_crit_val (float): The critical value for Hotelling's T
            """
            
            # Plot the data using seaborn
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 8))
            
            # Plot the scatter plot with colors for density scores and styles for outliers
            # FIX: Set legend=False to prevent seaborn from trying to create its own legend
            scatter = sns.scatterplot(
                x=pca_scores[:, 0],
                y=pca_scores[:, 1],
                c=knn_density_scores,
                s=60,
                markers=['o', 'X'],
                alpha=0.8,
                legend=False # This is the key fix
            )
            
            # Add a color bar to show the scale of the density scores
            cbar = plt.colorbar(scatter.get_children()[0], ax=plt.gca())
            cbar.set_label('Average Distance to Neighbors (Density Score)', rotation=270, labelpad=15)
            
            # Plot the T² ellipse
            self._plot_t2_ellipse(pca_scores, hotelling_t2_crit_val)

            plt.title('PCA Plot of First Two Principal Components', fontsize=16)
            plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]*100:.2f}% Variance)', fontsize=12)
            plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]*100:.2f}% Variance)', fontsize=12)
            plt.grid(True)
            plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)

            # Determine the plot name from the title for saving
            plot_name = "PCA_train"
            return self._handle_save_or_show(plot_name)
    
    def plot_pca_train_test(self, pca_train_scores, y_train, pca_test_scores, y_test, knn_density_scores_train, knn_density_scores_test, 
                            explained_variance_ratio, hotelling_t2_crit_val):
        """
        Plots the PCA scores for both training and testing datasets on a single plot.
        
        This function visualizes the relationship between the two datasets in the
        principal component space, showing Hotelling's T² boundary and
        outliers identified by the density neighborhood test.
        
        Args:
            pca_train_scores (np.ndarray): The PCA scores for the training data.
            pca_test_scores (np.ndarray): The PCA scores for the testing data.
            knn_density_scores_train (np.ndarray): The KNN density scores for the train set.
            knn_density_scores_test (np.ndarray): The KNN density scores for the test set.
            explained_variance_ratio (np.ndarray): The explained variance ratio for each component.
            hotelling_t2_crit_val (float): The critical value for Hotelling's T².
        """

        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 10))

        # Create a single DataFrame for all data to leverage Seaborn's power
        df_train = pd.DataFrame(pca_train_scores[:,:2], columns=['PC1', 'PC2'])
        df_train['Activity'] = y_train.values  # Add the activity data
        df_train['density_score'] = knn_density_scores_train
        df_train['dataset'] = 'Train'
        
        df_test = pd.DataFrame(pca_test_scores[:,:2], columns=['PC1', 'PC2'])
        df_test['Activity'] = y_test.values  # Add the activity data
        df_test['density_score'] = knn_density_scores_test
        df_test['dataset'] = 'Test'
        
        df_combined = pd.concat([df_train, df_test])

        # Plot the training data in a single color
        sns.scatterplot(
            data=df_combined,
            x='PC1',
            y='PC2',
            hue='Activity',
            s=80,
            marker='o',
            alpha=0.7,
            style='dataset'
        )

        # Plot the T² ellipse
        self._plot_t2_ellipse(pca_train_scores, hotelling_t2_crit_val, color='red')
        
        # Add labels and title
        plt.title('PCA Train vs. Test Scores', fontsize=18)
        plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]*100:.2f}% Variance)', fontsize=14)
        plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]*100:.2f}% Variance)', fontsize=14)

        plt.grid(True)
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
        
        # Determine the plot name from the title for saving
        plot_name = "PCA_train_test"
        return self._handle_save_or_show(plot_name)

    def _plot_t2_ellipse(self, scores, t2_crit_val, confidence_level=0.95, color='red', linestyle='--'):
        """
        Plots the Hotelling's T^2 ellipse on the current matplotlib axes.
        
        Args:
            scores (np.ndarray): The PCA or PLS scores.
            confidence_level (float): The desired confidence level for the ellipse.
            color (str): The color of the ellipse.
            linestyle (str): The linestyle of the ellipse.
        """
        
        # The eigenvalues are the variances of the scores in each dimension
        eigenvalues = np.var(scores, axis=0)
        
        # The axes of the ellipse are sqrt(eigenvalue * T^2_crit)
        a = np.sqrt(eigenvalues[0] * t2_crit_val)
        b = np.sqrt(eigenvalues[1] * t2_crit_val)

        theta = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = a * np.cos(theta)
        y_ellipse = b * np.sin(theta)

        plt.plot(x_ellipse, y_ellipse, color=color, linestyle=linestyle, linewidth=2, label=f'Hotelling\'s T² ({confidence_level*100:.0f}% confidence)')
        plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))

    def plot_feature_importance_pca(self, sorted_loadings, n_components, top_n=None):
        """
        Plots the feature importance (loadings) for the first N principal components.

        Args:
            n_components (int): The number of components to plot.
            sorted_loadings (pd.DataFrame): DataFrame containing feature loadings sorted by importance.
        """
        
        plt.figure(figsize=(12, sorted_loadings.shape[0] * 0.5 + 2)) # Dynamic height
        sns.heatmap(sorted_loadings[[f'PC{i+1}' for i in range(n_components)]], cmap='viridis', annot=True, fmt=".2f")
        
        title_text = f'PCA Feature Loadings for First {n_components} Components'
        if top_n is not None:
            title_text += f' (Top {sorted_loadings.shape[0]} Features)'

        plt.title(title_text, fontsize=16)
        plt.xlabel('Principal Components', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        # Determine the plot name from the title for saving
        plot_name = "PCA_feature_importance"
        return self._handle_save_or_show(plot_name)
    
    def plot_feature_importance_pls_regression(self, vip_df, top_n=None):
        """
        Calculates and plots the Variable Importance in Projection (VIP) scores
        for the PLS Regression model.
        Args:
            vip_df (pd.DataFrame): DataFrame containing VIP scores with columns 'Feature' and 'VIP'.
        """
        
        plt.figure(figsize=(12, vip_df.shape[0] * 0.5 + 2)) # Dynamic height
        sns.barplot(x='VIP', y='Feature', data=vip_df, hue='Feature', palette='crest')
        plt.axvline(x=1, color='red', linestyle='--', label='VIP > 1.0 Threshold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))

        title_text = 'PLS Regression Variable Importance in Projection (VIP)'
        if top_n is not None:
            title_text += f' (Top {vip_df.shape[0]} Features)'

        plt.title(title_text, fontsize=16)
        plt.xlabel('VIP Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        # Determine the plot name from the title for saving
        plot_name = "PLS_VIP_scores"
        return self._handle_save_or_show(plot_name)
    
    def plot_sal(self, sal_data, threshold, jitter_strength=0.1):
        """
        Generates a Structure-Activity Landscape (SAL) plot with optimized thermal agitation
        to prevent overlapping points.
        
        Args:
            sal_data (pd.DataFrame): DataFrame containing 'Structural_Distance' 
                                    and 'Activity_Distance' columns.
            threshold (float): The structural similarity threshold for the activity cliff.
            jitter_strength (float): The amount of random noise to add to the data.
                                    A smaller value means less jitter.
        """

        # Create a copy of the DataFrame to avoid modifying the original data
        jittered_data = sal_data.copy()
        
        # Apply thermal agitation (jitter) to the data. This is a very fast operation.
        jittered_data['Structural_Distance'] += np.random.uniform(
            low=-jitter_strength, high=jitter_strength, size=len(sal_data)
        )
        jittered_data['Activity_Distance'] += np.random.uniform(
            low=-jitter_strength, high=jitter_strength, size=len(sal_data)
        )

        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 8))
        
        # Create the scatter plot using the jittered data
        sns.scatterplot(
            x='Structural_Distance', 
            y='Activity_Distance', 
            data=jittered_data, 
            alpha=0.7, 
            s=60
        )
        
        # Add a vertical line to highlight the "activity cliff" region
        plt.axvline(x=threshold, color='red', linestyle='--', label='Structural Similarity Threshold')

        plt.title('Structure-Activity Landscape (SAL) Plot', fontsize=16)
        plt.xlabel('Structural Distance', fontsize=12)
        plt.ylabel('Activity Distance', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        
        # Determine the plot name from the title for saving
        plot_name = "SAL_train"
        return self._handle_save_or_show(plot_name)

    def plot_train_test_sal(self, sal_data, threshold, jitter_strength=0.1):
        """
        Generates a Structure-Activity Landscape (SAL) plot for the train-test comparison.
        Args:
            sal_data (pd.DataFrame): DataFrame with train-test distance pairs.
            X_train_original (np.ndarray): The original, unscaled training data for calculating the cliff threshold.
        """

        # Create a copy of the DataFrame to avoid modifying the original data
        jittered_data = sal_data.copy()
        
        # Apply thermal agitation (jitter) to the data. This is a very fast operation.
        jittered_data['Structural_Distance'] += np.random.uniform(
            low=-jitter_strength, high=jitter_strength, size=len(sal_data)
        )
        jittered_data['Activity_Distance'] += np.random.uniform(
            low=-jitter_strength, high=jitter_strength, size=len(sal_data)
        )

        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 8))
        
        # Plot the train-test comparison data
        sns.scatterplot(x='Structural_Distance', y='Activity_Distance', data=jittered_data, 
                        alpha=0.7, s=60, color='purple', label='Train vs Test Pairs')
        
        plt.axvline(x=threshold, color='red', linestyle='--', label='Structural Similarity Threshold (10th percentile)')
        
        plt.title('Train vs Test Structure-Activity Landscape (SAL) Plot', fontsize=16)
        plt.xlabel('Structural Distance (between Train and Test samples)', fontsize=12)
        plt.ylabel('Activity Distance (between Train and Test samples)', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        
        # Determine the plot name from the title for saving
        plot_name = "SAL_train_test"
        return self._handle_save_or_show(plot_name)

    def plot_sal_violin(self, sal_data, threshold):
        """
        Generates a violin plot for each category (0 and 1) in 'Activity_distance'
        with 'Structural_Distance' on the y-axis, highlighting only the outliers
        that are above the specified threshold.

        Args:
            sal_data (pd.DataFrame): The input DataFrame with 'Structural_Distance'
                                    and 'Activity_Distance' columns.
            threshold (float): A pre-calculated threshold value. Points with an
                            'Activity_Distance' greater than this threshold will
                            be highlighted.
        """
        # Define the columns based on the user's request
        x_col = 'Structural_Distance'
        y_col = 'Activity_Distance'
        
        # Check if the required columns exist
        if x_col not in sal_data.columns or y_col not in sal_data.columns:
            print(f"Error: The DataFrame must contain '{x_col}' and '{y_col}' columns.")
            return

        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Create the violin plot to show the overall distribution shape
        sns.violinplot(
            x=x_col, 
            y=y_col, 
            data=sal_data, 
            inner=None,
            hue=y_col,
            palette="pastel",
            orient='h'
        )
        
        plt.axvline(x=threshold, color='red', linestyle='--', label='Structural Similarity Threshold (10th percentile)')
        
        plt.title('Violin Plot with Activity Cliffs Highlighted', fontsize=16)
        plt.xlabel('Structure Distance', fontsize=12)
        plt.ylabel('Activity Distance', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
        
        # Determine the plot name from the title for saving
        plot_name = "SAL_violin_plot"
        return self._handle_save_or_show(plot_name)

    def plot_ad_scatter(self, report: dict):
        """
        A generic AD scatter plot generator.

        Args:
            report (dict): A dictionary containing all necessary data and labels for the plot.
        """
        plt.figure(figsize=(10, 8))
        
        inside_ad = ~report['is_outside']
        
        # Plot points inside the AD
        sns.scatterplot(
            x=report['x_data'][inside_ad], 
            y=report['y_data'][inside_ad], 
            color='royalblue', 
            alpha=0.7, 
            label='Inside AD'
        )
        
        # Plot points outside the AD
        sns.scatterplot(
            x=report['x_data'][report['is_outside']], 
            y=report['y_data'][report['is_outside']], 
            color='red', 
            alpha=0.7, 
            label='Outside AD'
        )

        # Draw threshold lines
        if report.get('x_threshold') is not None:
            x_label_parts = ' '.join(report['x_label'].split(' ')[1:3])
            x_label_text = f"{x_label_parts} Threshold: {report['x_threshold']:.3f}"
            plt.axvline(x=report['x_threshold'], color='black', linestyle='--', label=x_label_text)
        if report.get('y_threshold') is not None:
            y_label_parts = ' '.join(report['y_label'].split(' ')[1:3])
            y_label_text = f"{y_label_parts} Threshold: {report['y_threshold']:.3f}"
            plt.axhline(y=report['y_threshold'], color='black', linestyle=':', label=y_label_text)

        # Shade the reliable AD region
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()

        x_fill = [report['x_threshold'], xlim[1]] if report.get('ad_region_x') == 'high' else [xlim[0], report['x_threshold']]
        y_fill = [ylim[0], report['y_threshold']] if report.get('ad_region_y') == 'low' else [report['y_threshold'], ylim[1]]
        
        plt.fill_between(x_fill, y_fill[0], y_fill[1], color='lightgreen', alpha=0.2, label='Reliable AD Region')

        plt.title(report['title'], fontsize=16)
        plt.xlabel(report['x_label'], fontsize=12)
        plt.ylabel(report['y_label'], fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        
        plot_name = report['title'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('²', '2')
        return self._handle_save_or_show(plot_name)
    
    def generate_distplots_with_thresholds(self, all_results, plot_columns, threshold_map):
        """
        Plots the distribution of AD metrics for a single dataset.
        
        Returns:
            list: A list of file paths for the saved plots.
        """
        saved_paths = []
        for column in plot_columns:
            if column not in all_results.columns:
                print(f"Warning: Skipping {column}. Column or its threshold not found.")
                continue

            # threshold_value = all_results[threshold_map[column]].iloc[0]
            threshold_value = threshold_map[f'{column}_thresh']

            plt.figure(figsize=(10, 6))
            sns.set_style("whitegrid")
            
            sns.histplot(data=all_results, x=column, color="skyblue", label="Data", kde=True, stat="density", bins=50, linewidth=0)
            
            plt.axvline(threshold_value, color='red', linestyle='--', label=f'Threshold: {threshold_value:.2f}')
            
            plot_title = f'Distribution of {column}'
            plt.title(plot_title, fontsize=16)
            plt.xlabel(column, fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
            
            plot_name = f"dist_{column.replace(' ','_')}"
            saved_path = self._handle_save_or_show(plot_name)
            if saved_path:
                saved_paths.append(saved_path)
                
        return saved_paths
    
    def generate_distplots_with_thresholds_train_test(self, train_results, test_results, plot_columns, threshold_map):
        """
        Plots the distribution of AD metrics for both training and test data.
        
        Returns:
            list: A list of file paths for the saved plots.
        """
        saved_paths = []
        for column in plot_columns:
            if column not in train_results.columns or column not in test_results.columns:
                print(f"Warning: Skipping {column}. Column or its threshold not found.")
                continue

            threshold_value = threshold_map[f'{column}_thresh']
            
            plt.figure(figsize=(10, 6))
            sns.set_style("whitegrid")
            
            sns.histplot(data=train_results, x=column, color="skyblue", label="Train Data", kde=True, stat="density", bins=50, linewidth=0)
            sns.histplot(data=test_results, x=column, color="salmon", label="Test Data", kde=True, stat="density", bins=50, linewidth=0)
            
            plt.axvline(threshold_value, color='red', linestyle='--', label=f'Threshold: {threshold_value:.2f}')
            
            plot_title = f'Distribution of {column}'
            plt.title(plot_title, fontsize=16)
            plt.xlabel(column, fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
            
            plot_name = f"dist_{column.replace(' ','_')}"
            saved_path = self._handle_save_or_show(plot_name)
            if saved_path:
                saved_paths.append(saved_path)
        
        return saved_paths

    def generate_distplots_by_class(self, train_results, test_results, y_train, y_test, plot_columns, threshold_map):
        """
        Plots the distribution of various AD metrics by class for training and test data.
        
        Args:
            train_results (pd.DataFrame): DataFrame with training data metrics.
            test_results (pd.DataFrame): DataFrame with test data metrics.
            y_train (pd.Series): Training data labels.
            y_test (pd.Series): Test data labels.
            plot_columns (list): A list of metric column names to be plotted.
            threshold_map (dict): A dictionary mapping metric columns to their threshold columns.
        """
        print("Generating class-specific distribution plots...")
        
        # Add labels to the dataframes for easier filtering
        train_results['class'] = y_train
        test_results['class'] = y_test
        saved_paths = []
        for column in plot_columns:
            
            # Check for required columns
            if column not in train_results.columns or column not in test_results.columns:
                print(f"Warning: Skipping {column}. Required columns not found.")
                continue

            threshold_value = threshold_map[f'{column}_thresh']
            
            plt.figure(figsize=(12, 8))
            
            # Plot distributions for the training set
            sns.kdeplot(data=train_results[train_results['class'] == 1], x=column, color='firebrick', linestyle='-', label='Train: Positive Class', fill=False)
            sns.kdeplot(data=train_results[train_results['class'] == 0], x=column, color='darkblue', linestyle='-', label='Train: Negative Class', fill=False)
            
            # Plot distributions for the test set
            sns.kdeplot(data=test_results[test_results['class'] == 1], x=column, color='indianred', linestyle='--', label='Test: Positive Class', fill=False)
            sns.kdeplot(data=test_results[test_results['class'] == 0], x=column, color='cornflowerblue', linestyle='--', label='Test: Negative Class', fill=False)
            
            # Add the vertical threshold line
            plt.axvline(threshold_value, color='black', linestyle='--', label=f'Threshold: {threshold_value:.2f}')
            
            plot_name = f'Distribution of {column} by Class'
            plt.title(plot_name, fontsize=16)
            plt.xlabel(column, fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
            
            plot_name = f"dist_class{column.replace(' ','_')}"
            saved_path = self._handle_save_or_show(plot_name)
            if saved_path:
                saved_paths.append(saved_path)

        return saved_paths
    
    ### Structure plotting functions ###
    
    def plot_fold_disjointness(self, disjointness_df: pd.DataFrame):
        """
        Visualizes the chemical disjointness (Max Similarity to Train) per CV fold.
        """
        if disjointness_df is None or disjointness_df.empty:
            self.logger.warning("No disjointness data to plot.")
            return None

        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Use a nice palette
        sns.boxplot(x='Fold', y='Max Similarity', data=disjointness_df, palette='viridis', hue='Fold', showfliers=False)
        
        # Add a horizontal line at 0.4 (Distinct Threshold)
        plt.axhline(y=0.4, color='r', linestyle='--', label='Distinct Threshold (~0.4)')
        
        plt.title("Chemical Disjointness Verification per Fold", fontsize=16)
        plt.ylabel("Max Tanimoto Similarity to Training Set", fontsize=12)
        plt.xlabel("Cross-Validation Fold", fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
        
        plot_name = "CV_fold_disjointness"
        return self._handle_save_or_show(plot_name)
    
    def plot_ad_validation_analysis(self, ad_test_obj: Any, X_test: Any, y_test: Any, y_pred: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates a comprehensive analysis of AD metrics vs Prediction Outcomes (Traffic Light analysis).
        
        1. Score Plot colored by Outcome (Correct Prediction, Incorrect Prediction, Uncertain).
        * Training Data is plotted as a DENSITY (Applicability Domain).
        2. Boxplots of AD Metrics split by Outcome.

        Args:
            ad_test_obj: The fitted ADTest object (contains model, scaler, training scores).
            X_test: Test features (unscaled).
            y_test: True test labels.
            y_pred: Predicted test labels (-1 indicates uncertain).

        Returns:
            Tuple[str, str]: Paths to the two generated plots (Scores, Boxplots).
        """
        
        # --- 1. Prepare Data & Categories ---
        # Create a DataFrame for plotting
        df = pd.DataFrame()
        df['y_test'] = y_test.values if hasattr(y_test, 'values') else y_test
        df['y_pred'] = y_pred.values if hasattr(y_pred, 'values') else y_pred
        
        # Define Categories
        conditions = [
            (df['y_pred'] == -1),  # Uncertain
            (df['y_pred'] != -1) & (df['y_pred'] != df['y_test']), # Confident Error
            (df['y_pred'] == df['y_test']) # Correct
        ]
        choices = ['Uncertain', 'Incorrect Prediction', 'Correct Prediction']
        df['Outcome'] = np.select(conditions, choices, default='Unknown')
        
        # Define Palette (Traffic Light Logic)
        palette = {
            'Correct Prediction': '#2ca02c',          # Green
            'Uncertain': '#ff7f0e',        # Orange
            'Incorrect Prediction': '#d62728'   # Red
        }
        
        # --- 2. Calculate Scores (Latent Space) ---
        # We need to project X_test into the same space as X_train
        if hasattr(ad_test_obj, 'scaler') and ad_test_obj.scaler is not None:
            # Handle pandas DataFrame vs numpy array
            X_test_vals = X_test.values if hasattr(X_test, 'values') else X_test
            X_test_scaled = ad_test_obj.scaler.transform(X_test_vals)
        else:
            X_test_scaled = X_test.values if hasattr(X_test, 'values') else X_test

        # Get scores (t1, t2)
        scores = ad_test_obj.model_.transform(X_test_scaled)
        df['t1'] = scores[:, 0]
        df['t2'] = scores[:, 1]
        
        # Get Raw AD Metrics for X_test
        metrics_df = ad_test_obj.get_raw_metrics(X_test, y_test)
        df = pd.concat([df, metrics_df.reset_index(drop=True)], axis=1)

        # --- PLOT A: The "Traffic Light" Latent Space ---
        plt.figure(figsize=(10, 8))
        
        # -------------------------------------------------------------
        # 1. Plot Training Data Background (Density / Convex Hull representation)
        # -------------------------------------------------------------
        train_t1 = ad_test_obj.train_scores[:, 0]
        train_t2 = ad_test_obj.train_scores[:, 1]

        # Filled Density "Cloud" (The Safe Zone)
        # thresh=0.05 hides the very lowest density areas to keep the plot clean
        sns.kdeplot(x=train_t1, y=train_t2, 
                fill=True, color='slategrey', alpha=0.3, 
                levels=5, thresh=0.05)
        
        # Boundary Line (Approximating the Convex Hull limit)
        sns.kdeplot(x=train_t1, y=train_t2, 
                fill=False, color='black', linestyles='--', 
                levels=[0.05], linewidths=0.8, alpha=0.5)

        # Optional: Plot faint scatter of training points if density is too abstract
        # plt.scatter(train_t1, train_t2, c='grey', s=10, alpha=0.1)
        
        # -------------------------------------------------------------
        # 2. Plot Test Data Colored by Outcome
        # -------------------------------------------------------------
        # Added zorder=10 to ensure test points sit ON TOP of the density cloud
        ax = sns.scatterplot(data=df, x='t1', y='t2', hue='Outcome', style='Outcome',
                    palette=palette, s=80, alpha=0.9, edgecolor='black', zorder=10)
        
        # Get the handles and labels generated by the scatterplot (The Traffic Light)
        handles, labels = ax.get_legend_handles_labels()
        
        # Create a "Proxy Artist" - a gray patch representing the KDE density
        density_patch = mpatches.Patch(color='slategrey', alpha=0.3, label='Training Series')
        
        # Append this patch to the list of handles
        handles.append(density_patch)
        labels.append('Training Series')

        # Re-draw the legend with the combined list
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Safe title handling based on model type
        model_type = getattr(ad_test_obj, 'model_type', 'Model').upper()
        if hasattr(ad_test_obj.model_, 'explained_variance_ratio_'):
                # PCA-like
                var_1 = ad_test_obj.model_.explained_variance_ratio_[0]
                var_2 = ad_test_obj.model_.explained_variance_ratio_[1]
                xlab = f'Component 1 ({var_1:.1%} var)'
                ylab = f'Component 2 ({var_2:.1%} var)'
        else:
            # PLS or other
            xlab = 'Latent Variable 1'
            ylab = 'Latent Variable 2'

        plt.title(f'{model_type} Score Plot: Prediction Outcomes', fontsize=14)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # SAVE PLOT A
        plot_name_a = "traffic_light_scores"
        path_a = self._handle_save_or_show(plot_name_a)

        # --- PLOT B: Validation Boxplots ---
        metrics_to_plot = ['DModX', 'T2', 'LOF_Score', 'kNN_Dist', 'KDE_LL', 'IForest_Score', 'AE_Recon_Error', 'Di_Similarity_Density', 'Ii_Activity_Inconsistency'] 
        valid_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        path_b = None
        if valid_metrics:
            fig, axes = plt.subplots(1, len(valid_metrics), figsize=(5 * len(valid_metrics), 6))
            if len(valid_metrics) == 1: axes = [axes]
            
            for ax, metric in zip(axes, valid_metrics):
                # Boxplot
                sns.boxplot(data=df, 
                            x='Outcome', 
                            y=metric, 
                            hue='Outcome',
                            legend=False,
                            order=['Correct Prediction', 'Incorrect Prediction', 'Uncertain'],
                            palette=palette, 
                            ax=ax, 
                            showfliers=False, 
                            medianprops={'color': 'black', 'linewidth': 2.5}
                            ) 
                
                # Strip plot: this is used to plot the outliers as well
                # sns.stripplot(data=df, x='Outcome', y=metric, order=['Correct Prediction', 'Incorrect Prediction', 'Uncertain'],
                #                 color='black', alpha=0.3, size=3, ax=ax, jitter=True)
                
                # Add Threshold Line if available
                thresh_key = f"{metric}_thresh"
                thresholds = ad_test_obj.get_final_thresholds()
                if thresh_key in thresholds:
                    ax.axhline(thresholds[thresh_key], color='red', linestyle='--', label='AD Threshold')
                
                ax.set_title(f"{metric} Distribution")
                ax.set_xlabel("")
                ax.tick_params(axis='x', rotation=30)
                
            plt.suptitle("Validation of AD Metrics against Prediction Outcomes", fontsize=16)
            plt.tight_layout()
            
            # SAVE PLOT B
            plot_name_b = "ad_validation_boxplots"
            path_b = self._handle_save_or_show(plot_name_b)

        return path_a, path_b
