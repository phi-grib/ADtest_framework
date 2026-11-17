"""
Author: Eric March (eric.march@upf.edu)
Created: 2026-04-28
Refactored: 2026-08-11
"""

import gc
import os
import numpy as np
import pandas as pd
import matplotlib
import sys

matplotlib.use('Agg')

from typing import Optional, List

from .adtest import ADTest
from .cluster_cross_val import ClusteringCrossValidator
from .utils import data_io as d
from .utils import utils as u
from .utils.qsar_model import QSARModel
from .utils.validation import precompute_fingerprints, compute_fold_disjointness
from .utils.reporting import generate_report

class ADTestPipeline:
    """
    Orchestrator class for the ADTest pipeline.
    """
    def __init__(self, config: dict, logger, filename_prefix: str, report_base_path: str):
        self.config = config
        self.logger = logger
        self.filename_prefix = filename_prefix
        self.report_base_path = report_base_path
        
        self.X_train_full = None
        self.y_train_full = None
        self.train_struc = None
        self.X_test_full = None
        self.y_test_full = None
        self.test_struc = None

    def run(self):
        """
        Runs the data loading, cross-validation fold generation, OOF metric extraction,
        and external test set transformation.
        """
        # --- 1. Load and Preprocess Data ---
        self.X_train_full, self.y_train_full, self.train_struc, \
        self.X_test_full, self.y_test_full, self.test_struc = d._load_and_preprocess_data(
            config=self.config, logger=self.logger
        )
        
        # --- 2. Main Dataset Summary ---
        train_size = len(self.X_train_full)
        test_size = len(self.X_test_full) if self.X_test_full is not None else 0
        train_positives = int((self.y_train_full == 1).sum())
        train_negatives = int((self.y_train_full == 0).sum())
        test_positives = int((self.y_test_full == 1).sum()) if self.X_test_full is not None else 0
        test_negatives = int((self.y_test_full == 0).sum()) if self.X_test_full is not None else 0

        pos_class_percent_train = round(train_positives / train_size * 100, 2) if train_size > 0 else 0.0
        pos_class_percent_test = round(test_positives / test_size * 100, 2) if test_size > 0 else 0.0
        neg_class_percent_train = round(train_negatives / train_size * 100, 2) if train_size > 0 else 0.0
        neg_class_percent_test = round(test_negatives / test_size * 100, 2) if test_size > 0 else 0.0

        main_dataset_summary = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'Total Compounds': [train_size, test_size],
            'Positive Class': [train_positives, test_positives],
            'Positive Class %': [pos_class_percent_train, pos_class_percent_test],
            'Negative Class': [train_negatives, test_negatives],
            'Negative Class %': [neg_class_percent_train, neg_class_percent_test]
        }).set_index('Dataset')

        # --- 3. Run OOF CV Analysis ---
        results = self._run_cv_analysis()

        # Save preliminary state outputs and generate final report
        if results is not None and results.get('master_df') is not None:
            self._save_analysis_outputs(
                master_df=results['master_df'],
                oof_df=results.get('test_outlier_table_oof')
            )

            self.logger.info("Generating Final YAML and DOCX Reports...")
            
            # 1. Calculate QSAR metrics for Raw Thresholds
            raw_report_data = {}
            if self.config.get('use_qsar', False):
                self._add_qsar_performance_metrics(raw_report_data, results['test_outlier_table_full'])
            
            # 2. Calculate QSAR metrics for OOF Thresholds
            oof_report_data = {}
            if self.config.get('use_qsar', False):
                self._add_qsar_performance_metrics(oof_report_data, results['test_outlier_table_oof'])

            def _summarize_outliers(df, name):
                """
                Builds a one-row AD coverage + error-enrichment summary for a given
                outlier table (Raw or OOF thresholds).

                Besides Inside/Outside AD compound counts, this reports the percentage
                of compounds WITHIN each subset (inside vs outside AD) that carry a QSAR
                prediction error ('Prediction_Error' == 1). This is the core evidence used
                to argue that compounds flagged outside the AD are enriched in prediction
                errors relative to compounds inside it.

                Args:
                    df (pd.DataFrame): Outlier table (must contain 'Total Metrics Flagged';
                                        'Prediction_Error' is optional, e.g. absent if no
                                        QSAR predictions were supplied).
                    name (str): Label for the AD strategy (e.g. 'Raw (Training Boundaries)').

                Returns:
                    pd.DataFrame: Single-row summary table.
                """
                total = len(df)
                has_flags = 'Total Metrics Flagged' in df.columns
                outside_mask = (df['Total Metrics Flagged'] >= 1) if has_flags else pd.Series(False, index=df.index)
                inside_mask = ~outside_mask

                outside = int(outside_mask.sum())
                inside = total - outside

                # Column name has varied historically ('Prediction_Error' vs 'Prediction Error')
                error_col = next((c for c in ['Prediction_Error', 'Prediction Error'] if c in df.columns), None)

                pct_err_inside, pct_err_outside = 0.0, 0.0
                if error_col is not None:
                    inside_errors = df.loc[inside_mask, error_col].dropna()
                    outside_errors = df.loc[outside_mask, error_col].dropna()
                    pct_err_inside = round(inside_errors.mean() * 100, 2) if len(inside_errors) > 0 else 0.0
                    pct_err_outside = round(outside_errors.mean() * 100, 2) if len(outside_errors) > 0 else 0.0

                return pd.DataFrame({
                    'AD Strategy': [name],
                    'Total Test Set': [total],
                    'Inside AD': [inside],
                    'Outside AD': [outside],
                    '% Outside': [round((outside / total) * 100, 2) if total > 0 else 0.0],
                    '% Errors Inside AD': [pct_err_inside],
                    '% Errors Outside AD': [pct_err_outside]
                })

            summary_raw = _summarize_outliers(results['test_outlier_table_full'], 'Raw (Training Boundaries)')
            summary_oof = _summarize_outliers(results['test_outlier_table_oof'], 'OOF (CV Boundaries)')
            combined_summary_df = pd.concat([summary_raw, summary_oof], ignore_index=True)

            # 2b. Metric-level flagging diagnostics (Raw vs OOF), reusing ADTest helpers
            #     on the master AD model. These are per-run, not per-fold, since they
            #     summarize the FINAL external test set outlier table.
            master_ad = results['master_ad']

            flag_multiplicity_raw = master_ad._calculate_outlier_summary_stats(results['test_outlier_table_full'])
            flag_multiplicity_oof = master_ad._calculate_outlier_summary_stats(results['test_outlier_table_oof'])

            exclusive_flags_raw = master_ad._calculate_exclusive_metric_flags(results['test_outlier_table_full'])
            exclusive_flags_oof = master_ad._calculate_exclusive_metric_flags(results['test_outlier_table_oof'])

            # 3. Compile the external test report dictionary
            external_test_cv_report = {
                'test_outlier_summary': combined_summary_df, 
                'qsar_metrics_raw': raw_report_data.get('qsar_metrics', {}),
                'qsar_metrics_oof': oof_report_data.get('qsar_metrics', {}),
                'flag_multiplicity_raw': flag_multiplicity_raw.to_dict() if flag_multiplicity_raw is not None else {},
                'flag_multiplicity_oof': flag_multiplicity_oof.to_dict() if flag_multiplicity_oof is not None else {},
                'exclusive_flags_raw': exclusive_flags_raw.to_dict() if exclusive_flags_raw is not None else {},
                'exclusive_flags_oof': exclusive_flags_oof.to_dict() if exclusive_flags_oof is not None else {},
            }

            # 4. Determine output directory from report_base_path
            output_dir = os.path.dirname(self.report_base_path) if os.path.dirname(self.report_base_path) else "."

            # 5. Call the refactored ReportGenerator
            generate_report(
                report_data=None, 
                output_dir=output_dir,
                prefix=self.filename_prefix,
                config=self.config,
                main_dataset_summary=main_dataset_summary,
                external_test_cv_report=external_test_cv_report,
                raw_thresholds=results.get('raw_thresholds'),
                oof_thresholds=results.get('oof_thresholds')
            )

        self.logger.info(f"--- Finished OOF processing for: {self.config['name']} ---")

        matplotlib.pyplot.close('all')
        gc.collect()

        return results

    def _generate_ad_plots(self, ad_analyzer, y_pred_qsar, X_test, y_test):
        self.logger.info("Generating and saving analysis plots...")
        model_type = self.config.get('model', 'pca').lower()
        feature_importance = self.config.get('feature_importance', 10)
        plot_paths = []

        try:
            if model_type == 'pca':
                plot_paths.append(ad_analyzer.plot_pca())
                plot_paths.append(ad_analyzer.plot_pca_train_test(X_test, y_test))
                plot_paths.append(ad_analyzer.plot_feature_importance_pca(top_n=feature_importance))
            elif model_type == 'pls':
                self.logger.error(f"This run only works with pca as model type, not pls. Please check your configuration.")
                sys.exit(1)
            
            plot_paths.append(ad_analyzer.plot_distplot_metrics_with_thresholds(X_test, y_test))
            plot_paths.append(ad_analyzer.plot_sal())
            plot_paths.append(ad_analyzer.plot_sal_train_test(X_test, y_test))
            plot_paths.append(ad_analyzer.plot_sal_violin(X_test, y_test))
            plot_paths.append(ad_analyzer.plot_adsal())
            plot_paths.append(ad_analyzer.plot_adsal_test(X_test, y_pred_qsar))
            
            tl_score_path, tl_box_path = ad_analyzer.plot.plot_ad_validation_analysis(
                ad_test_obj=ad_analyzer,
                X_test=X_test,
                y_test=y_test,
                y_pred=y_pred_qsar
            )
            if tl_score_path: plot_paths.append(tl_score_path)
            if tl_box_path: plot_paths.append(tl_box_path)
                
            plot_paths = [p for p in plot_paths if p is not None]

        except Exception as e:
            self.logger.error(f"Could not generate or save plots: {e}", exc_info=True)

        self.logger.info(f"Saved {len(plot_paths)} plots.")

        matplotlib.pyplot.close('all')  # Frees the figure buffer immediately after saving

        return plot_paths

    def _add_qsar_performance_metrics(self, report_data, outlier_df):
        def eval_subset(mask):
            if mask.sum() == 0: return {}
            return QSARModel().evaluate_classification_metrics(
                outlier_df.loc[mask, 'Real_Activity'],
                outlier_df.loc[mask, 'QSAR_Prediction']
            )

        metrics_global = QSARModel().evaluate_classification_metrics(
            outlier_df['Real_Activity'], 
            outlier_df['QSAR_Prediction']
        )
        
        inside_mask = (outlier_df['Total Metrics Flagged'] == 0)
        outside_mask = (outlier_df['Total Metrics Flagged'] >= 1)
        
        report_data['qsar_metrics'] = {
            'global': metrics_global,
            'inside_ad': eval_subset(inside_mask),
            'outside_ad': eval_subset(outside_mask)
        }
        
        report_data['confusion_matrix_global'] = {k: metrics_global.get(k, 0) for k in ['TP', 'TN', 'FP', 'FN']}
        
        m_in = report_data['qsar_metrics']['inside_ad']
        report_data['confusion_matrix_inside_AD'] = {k: m_in.get(k, 0) for k in ['TP', 'TN', 'FP', 'FN']}
        
        m_out = report_data['qsar_metrics']['outside_ad']
        report_data['confusion_matrix_outside_AD'] = {k: m_out.get(k, 0) for k in ['TP', 'TN', 'FP', 'FN']}

    def _get_clustering_scores(self, X_train, y_train):
        cluster_algo = self.config.get('cluster_algorithm', 'kmeans')
        cluster_desc = self.config.get('cluster_descriptors') or self.config['descriptors']
        cluster_model = self.config.get('cluster_model') or self.config.get('model', 'pca').lower()
        
        if cluster_algo not in ['kmeans', 'hierarchical', 'dbscan', 'atc']:
            self.logger.info(f"Using structure-based clustering ('{cluster_algo}').")
            return pd.DataFrame(index=X_train.index)

        self.logger.info(f"Calculating scores for clustering using '{cluster_desc}'/'{cluster_model}'...")
        
        ad_analyzer = self._adtest_call(
            X_train=X_train, y_train=y_train, 
            descriptors=cluster_desc
        )

        if cluster_model == 'pca':
            return pd.DataFrame(ad_analyzer.train_scores, index=X_train.index)
        elif cluster_model == 'pls':
            return pd.DataFrame(ad_analyzer.train_scores, index=X_train.index)
        else:
            self.logger.error(f"Unsupported cluster_model: {cluster_model}")
            return None

    def _adtest_call(self, X_train, y_train, descriptors, train_struc=None):
        ad_analyzer = ADTest(
            X_train=X_train,
            y_train=y_train,
            logger=self.logger,
            n_components=self.config.get('n_components'),
            descriptors=descriptors,
            report_base_path=self.report_base_path,
            train_struc=train_struc,
            id_col=self.config.get('id_col'),
            smiles_col='canon_smiles',
            model_type=self.config.get('model', 'pca').lower(),
            alpha_percentile=self.config.get('alpha', 0.05)
        )
        return ad_analyzer

    def _generate_cv_plots(
        self,
        oof_metrics_raw_df: pd.DataFrame,
        y_train_oof: pd.Series,
        y_pred_oof: Optional[pd.Series] = None,
        fold_labels: Optional[np.ndarray] = None,
        ad_analyzer: Optional[ADTest] = None,
        plotter: Optional[object] = None, 
        save: bool = False) -> List[str]:

        """
        Generates evaluation plots based on Cross-Validation (Out-Of-Fold - OOF).
        Returns a list of saved plot file paths.
        """
        generated_paths = []
        model_type = self.config.get('model', 'pca').lower()

        if oof_metrics_raw_df is None or oof_metrics_raw_df.empty:
            self.logger.error("oof_metrics_raw_df is empty. Cannot generate CV plots.")
            return generated_paths

        self.logger.info("Generating Cross-Validation (OOF) diagnostic plots...")

        #Comparison of OOF distributions vs Training Thresholds
        threshold_map = ad_analyzer.get_final_thresholds()
        plot_columns = oof_metrics_raw_df.columns.tolist()

        path3 = plotter.generate_distplots_with_thresholds_train_test(
            train_results=ad_analyzer.train_metrics_raw_df,
            test_results=oof_metrics_raw_df,
            plot_columns=plot_columns,
            threshold_map=threshold_map
        )
        if path3: generated_paths.append(path3)

        # 5. Sample distribution and class balance per Fold
        if fold_labels is not None:
            path5 = self._plot_cv_fold_balance(y_train_oof, fold_labels)
            if path5: generated_paths.append(path5)

        self.logger.info("Cross-Validation plot generation completed.")
        return generated_paths

    def _plot_cv_fold_balance(self, y: pd.Series, fold_labels: np.ndarray) -> None:
        """
        Helper method to visually verify stratification equity across folds.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            df_folds = pd.DataFrame({'Target': y, 'Fold': fold_labels})
            
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df_folds, x='Fold', hue='Target')
            plt.title('Class Distribution per Fold in CV')
            plt.xlabel('Fold ID')
            plt.ylabel('Number of Compounds')
            plt.legend(title='Class')
            plt.tight_layout()
            
            if hasattr(self.plot, 'report_base_path') and self.plot.report_base_path:
                plt.savefig(f"{self.plot.report_base_path}/cv_fold_balance.png", dpi=300)
            plt.show()
            plt.close()
        except Exception as e:
            self.logger.warning(f"Could not generate fold balance plot: {e}")

    def _get_qsar_predictions(self, test_struc, X_train, y_train, X_test, y_test, is_cv=False):
        if self.config.get('prediction_column') and not is_cv:
            y_pred = test_struc[self.config['prediction_column']].values
        else:
            y_pred = QSARModel.get_predictions(
                self.logger, self.config, test_struc, X_train, y_train, X_test, y_test, is_cv=is_cv
            )

        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred, index=y_test.index)
        else:
            y_pred = y_pred.reindex(y_test.index)
        return y_pred

    def _save_analysis_outputs(self, master_df: pd.DataFrame, oof_df: pd.DataFrame = None):
        try:
            # 1. Save individual Raw CSV
            raw_csv_path = f"{self.report_base_path}_test_raw_outliers.csv"
            master_df.to_csv(raw_csv_path, index=False)

            if oof_df is None:
                self.logger.info(f"Saved output table to {raw_csv_path}")
                return

            # 2. Save individual OOF CSV
            oof_csv_path = f"{self.report_base_path}_test_oof_outliers.csv"
            oof_df.to_csv(oof_csv_path, index=False)

            # 3. Create Combined CSV
            # Identify outlier boolean/count flags vs invariant shared columns (metadata + metric values)
            flag_cols = [c for c in master_df.columns if c.endswith('_outlier') or 'Total Metrics' in c or 'Total_Metrics' in c]
            shared_cols = [c for c in master_df.columns if c not in flag_cols]

            raw_flags = master_df[flag_cols].add_suffix('_raw')
            oof_flags = oof_df[flag_cols].add_suffix('_oof')

            # Assemble shared metadata/metrics with the raw and oof flags
            combined_df = pd.concat([master_df[shared_cols], raw_flags, oof_flags], axis=1)

            # 4. Generate binary integer AD status columns (0 = Inside AD, 1 = Outside AD)
            raw_total_col = next((c for c in combined_df.columns if 'Total' in c and c.endswith('_raw')), None)
            oof_total_col = next((c for c in combined_df.columns if 'Total' in c and c.endswith('_oof')), None)

            if raw_total_col:
                combined_df['Status_AD_raw'] = (combined_df[raw_total_col] >= 1).astype(int)
            if oof_total_col:
                combined_df['Status_AD_oof'] = (combined_df[oof_total_col] >= 1).astype(int)

            combined_csv_path = f"{self.report_base_path}_test_combined_outliers.csv"
            combined_df.to_csv(combined_csv_path, index=False)

            # 5. Save clean Excel copy (<endpointname>_AD_test_results.xlsx)
            out_dir = os.path.dirname(self.report_base_path) or "."
            raw_ep = os.path.basename(self.report_base_path).split("_CV_")[0]
            endpoint_map = {
                "ACHE": "AChE",
                "CARCINOGEN": "Carcinogen",
                "ER_ALPHA": "ER_alpha",
                "ER_BETA": "ER_beta",
                "HERG": "hERG",
                "SKIN_REACTION": "Skin_Reaction",
                "TR_ALPHA": "TR_alpha",
            }
            ep_name = endpoint_map.get(raw_ep, raw_ep)
            combined_xlsx_path = os.path.join(out_dir, f"{ep_name}_AD_test_results.xlsx")
            combined_df.to_excel(combined_xlsx_path, index=False)

            self.logger.info(
                f"Saved output tables: {raw_csv_path}, {oof_csv_path}, "
                f"{combined_csv_path}, and {combined_xlsx_path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to save output files: {e}")

    def _run_cv_fold_loop(self, cv, scores_for_clustering, X_train_full, y_train_full, train_struc):
        oof_raw_metrics  = []
        oof_y_true       = []
        oof_y_pred       = []
     
        descriptors = self.config['descriptors']
        smiles_col  = self.config.get('smiles_col', 'canon_smiles')
     
        splitter    = cv.auto_split(
            scores=scores_for_clustering,
            y=y_train_full,
            df=train_struc,
            smiles_column=smiles_col
        )
        total_splits = cv.n_splits

        smiles_list = train_struc['canon_smiles'].tolist()
        fps = precompute_fingerprints(smiles_list, logger=self.logger)
        disjointness_list = []
     
        for i, (train_idx, val_idx) in enumerate(splitter):
            fold_num = i + 1
            self.logger.info(f"--- OOF Fold {fold_num}/{total_splits}: collecting raw metrics ---")
     
            X_train_fold  = X_train_full.iloc[train_idx]
            X_val_fold    = X_train_full.iloc[val_idx]
            y_train_fold  = y_train_full.iloc[train_idx]
            y_val_fold    = y_train_full.iloc[val_idx]
            struc_train   = train_struc.iloc[train_idx]
            struc_val     = train_struc.iloc[val_idx]
     
            fold_ad = self._adtest_call(
                X_train=X_train_fold,
                y_train=y_train_fold,
                descriptors=descriptors,
                train_struc=struc_train
            )
     
            y_pred_fold = self._get_qsar_predictions(
                struc_val,
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold, is_cv=True
            )
     
            val_raw_metrics = fold_ad.get_raw_metrics(X_val_fold, y_pred_fold)

            oof_raw_metrics.append(val_raw_metrics)
            oof_y_true.append(y_val_fold)
            oof_y_pred.append(y_pred_fold)
     
            disjointness_list.extend(
                compute_fold_disjointness(fps, train_idx, val_idx, fold_num)
            )

            self.logger.info(f"Fold {fold_num} complete.")
     
        master_oof_raw_df = pd.concat(oof_raw_metrics, axis=0)
        master_oof_y_true = pd.concat(oof_y_true,      axis=0)
        master_oof_y_pred = pd.concat(oof_y_pred,      axis=0)
     
        return master_oof_raw_df, master_oof_y_true, master_oof_y_pred, pd.DataFrame(disjointness_list)

    def _run_cv_analysis(self):
        self.logger.info("--- Step 2: Starting Clustered Cross-Validation Workflow ---")
    
        prep_res = d._prepare_main_descriptors(
            self.config, self.logger,
            self.X_train_full, self.y_train_full, self.train_struc,
            self.X_test_full, self.y_test_full, self.test_struc
        )
        if prep_res is None:
            return (None,) * 6
        X_train_main, y_train_main, train_struc_main, \
            X_test_main, y_test_main, test_struc_main = prep_res
    
        cluster_desc = self.config.get('cluster_descriptors') or self.config['descriptors']
        if cluster_desc != self.config['descriptors']:
            self.logger.info(f"Using FP-preprocessed train_struc for clustering (from initial '{cluster_desc}' preprocessing).")
            train_struc_clust = self.train_struc
        else:
            train_struc_clust = train_struc_main

        scores_df = self._get_clustering_scores(self.X_train_full, self.y_train_full)
        if scores_df is None:
            return (None,) * 6
    
        X_train_aligned, y_train_aligned, scores_aligned, _, train_struc_aligned = \
            u._align_data_indices(
                X_train_main, y_train_main, train_struc_main,
                scores_df, self.y_train_full, self.logger
            )

        common_idx = X_train_aligned.index
        train_struc_clust_aligned = train_struc_clust.loc[
            train_struc_clust.index.intersection(common_idx)
        ]

        X_train_aligned = X_train_aligned.reset_index(drop=True)
        y_train_aligned = y_train_aligned.reset_index(drop=True)
        scores_aligned = scores_aligned.reset_index(drop=True)
        train_struc_aligned = train_struc_aligned.reset_index(drop=True)
        train_struc_clust_aligned = train_struc_clust_aligned.reset_index(drop=True)

        cv = ClusteringCrossValidator(
            n_splits=self.config.get('n_splits', 5),
            n_clusters=self.config.get('n_clusters', 5),
            method=self.config.get('cluster_algorithm', 'kmeans'),
            logger=self.logger,
            df=train_struc_clust_aligned,
            random_state=42,
            smiles_column='canon_smiles')
        
        cluster_summary_df = cv.get_cluster_summary(X=scores_aligned, y=y_train_aligned)
        
        self.logger.info("Phase 1: collecting Out-of-Fold raw AD metrics...")
        (master_oof_raw_df, master_oof_y_true, 
        master_oof_y_pred, disjointness_df) = self._run_cv_fold_loop(
                cv=cv,
                scores_for_clustering=scores_aligned,
                X_train_full=X_train_aligned,
                y_train_full=y_train_aligned,
                train_struc=train_struc_aligned
            )
    
        self.logger.info("Phase 2: training master AD and QSAR models on full training set...")
        master_ad = self._adtest_call(
            X_train=X_train_aligned,
            y_train=y_train_aligned,
            descriptors=self.config['descriptors'],
            train_struc=train_struc_aligned
        )

        y_ext_pred = self._get_qsar_predictions(
                test_struc_main,
                X_train_aligned, y_train_aligned,
                X_test_main, y_test_main)
    
        self.logger.info("Calculating raw AD metrics for the external test set...")
        raw_ext_metrics = master_ad.get_raw_metrics(X_test_main, y_ext_pred)
    
        # 1. Thresholds & Outlier Table using Full Training Set
        raw_thresholds = master_ad.get_final_thresholds()
        test_outlier_table_full = master_ad._generate_unique_outlier_table(
            results_df=raw_ext_metrics,
            y_test=y_test_main,
            thresholds=raw_thresholds,
            struc_df=test_struc_main,
            y_pred_qsar=y_ext_pred
        )

        # 2. Thresholds & Outlier Table using OOF Metrics (swapping metric ref on master_ad)
        orig_train_metrics = master_ad.train_metrics_raw_df
        master_ad.train_metrics_raw_df = master_oof_raw_df
        oof_thresholds = master_ad.get_final_thresholds()
        master_ad.train_metrics_raw_df = orig_train_metrics  # restore original

        test_outlier_table_oof = master_ad._generate_unique_outlier_table(
            results_df=raw_ext_metrics,
            y_test=y_test_main,
            thresholds=oof_thresholds,
            struc_df=test_struc_main,
            y_pred_qsar=y_ext_pred
        )
    
        # Outlier table directly represents final AD assessment without scaled merging
        master_df = test_outlier_table_full

        all_plot_paths = []
        generate_plots = self.config.get('generate_plots', True)

        if generate_plots:
            self.logger.info("Generating plots for external test set and CV diagnostics...")

            # 1. External test set plots
            general_ad_plots_path = self._generate_ad_plots(master_ad, y_ext_pred, X_test_main, y_test_main)
            if general_ad_plots_path:
                if isinstance(general_ad_plots_path, list):
                    all_plot_paths.extend(general_ad_plots_path)
                else:
                    all_plot_paths.append(general_ad_plots_path)

            # 2. OOF Cross-Validation plots
            cv_plot_paths = self._generate_cv_plots(
                ad_analyzer=master_ad,
                oof_metrics_raw_df=master_oof_raw_df,
                y_train_oof=master_oof_y_true,
                y_pred_oof=master_oof_y_pred,
                save=True,
                plotter=master_ad.plot
            )
            if cv_plot_paths:
                all_plot_paths.extend(cv_plot_paths)

            # 3. Disjointness plot
            if disjointness_df is not None and not disjointness_df.empty:
                disjointness_plot_path = master_ad.plot.plot_fold_disjointness(disjointness_df)
                if disjointness_plot_path:
                    all_plot_paths.append(disjointness_plot_path)
        else:
            self.logger.info("Skipping plot generation for CV analysis.")

        return {
           "master_ad": master_ad,
            "master_oof_raw_df": master_oof_raw_df,
            "raw_ext_metrics": raw_ext_metrics,
            "raw_thresholds": raw_thresholds,
            "oof_thresholds": oof_thresholds,
            "test_outlier_table_full": test_outlier_table_full,
            "test_outlier_table_oof": test_outlier_table_oof,
            "master_df": master_df,
            "disjointness_df": disjointness_df,
            "cluster_summary_df": cluster_summary_df,
            "y_ext_pred": y_ext_pred,
            "X_test_main": X_test_main,
            "y_test_main": y_test_main,
            "test_struc_main": test_struc_main,
            "plot_paths": all_plot_paths
        }
    