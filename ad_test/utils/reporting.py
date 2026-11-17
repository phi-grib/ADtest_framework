"""
Handles the generation of final reports in different formats.

- Human-readable .docx report with summaries, plots, and log excerpts.
- Machine-readable .yaml report with structured results.

Refactored for Paper 1
"""

import os
import pandas as pd
import yaml

from docx import Document
from typing import Dict, List, Union, Optional


class ReportGenerator:
    """
    Orchestrates the generation of YAML and DOCX reports for AD analysis.
    """

    def __init__(self, 
                 output_dir: str, 
                 prefix: str, 
                 config: dict, 
                 report_data: Optional[Union[Dict, List[Dict]]] = None,
                 main_dataset_summary: Optional[pd.DataFrame] = None,
                 external_test_cv_report: Optional[dict] = None,
                 raw_thresholds: Optional[dict] = None,
                 oof_thresholds: Optional[dict] = None):
        
        self.output_dir = output_dir
        self.prefix = prefix
        self.config = config
        
        # Core datasets & reports
        self.report_data = report_data
        self.main_dataset_summary = main_dataset_summary
        self.external_test_cv_report = external_test_cv_report
        
        # Dual Thresholds for Paper 1
        self.raw_thresholds = raw_thresholds
        self.oof_thresholds = oof_thresholds
        
        self.is_cv_run = self.config.get('cluster_analysis', False)
        self.main_report_template = report_data if not self.is_cv_run else None
        self.document = None 

    def generate(self):
        """Generates all enabled report formats."""
        self._generate_yaml()
        self._generate_docx()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. YAML Report Generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_yaml(self):
        """Generates a machine-readable YAML report."""
        yaml_path = os.path.join(self.output_dir, f'{self.prefix}.yaml')
        
        try:
            report = {
                'analysis_setup': self._get_serializable_setup()
            }

            if self.main_dataset_summary is not None:
                report['overall_dataset_summary'] = self.main_dataset_summary.to_dict('index')

            # Add Dual Thresholds — cast numpy scalars to native float so
            # PyYAML serializes them as plain numbers instead of falling back
            # to '!!python/object/apply:numpy...' pickled representations.
            if self.raw_thresholds:
                report['raw_thresholds'] = {k: float(v) for k, v in self.raw_thresholds.items()}
            if self.oof_thresholds:
                report['oof_thresholds'] = {k: float(v) for k, v in self.oof_thresholds.items()}

            if not self.is_cv_run and self.main_report_template:
                 report['model_report'] = self._get_serializable_model_report()
                 self._add_single_results_to_yaml(report)

            if self.external_test_cv_report:
                report['external_test_set_performance'] = self._get_serializable_external_report()

            with open(yaml_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, sort_keys=False)
            print(f"YAML report saved to: {yaml_path}")

        except Exception as e:
            print(f"Error generating YAML report: {e}")

    def _get_serializable_setup(self) -> dict:
        setup = {
            'applicability_domain': {
                'descriptors': self.config.get('descriptors'),
                'model': self.config.get('model'),
                'n_components': self.config.get('n_components')
            }
        }
        if self.config.get('use_qsar', False):
            setup['QSAR_model'] = {
                'algorithm': self.config.get('qsar_model', {}).get('type', 'random_forest'),
                'parameters': self.config.get('qsar_model', {}).get('params', {})
            }
        if self.is_cv_run:
            setup['clustering'] = self._get_serializable_clustering_setup()
        return setup

    def _get_serializable_clustering_setup(self) -> dict:
        algo = self.config.get('cluster_algorithm', '').lower()
        setup = {'algorithm': self.config.get('cluster_algorithm')}
        if algo in ['kmeans', 'hierarchical', 'dbscan']:
            setup['cluster_descriptors'] = self.config.get('cluster_descriptors') or self.config.get('descriptors')
            setup['cluster_model'] = self.config.get('cluster_model') or self.config.get('model')
        elif algo == 'atc':
            setup['atc_level'] = self.config.get('atc_level')
        setup['n_clusters'] = self.config.get('n_clusters')
        return setup

    def _get_serializable_model_report(self):
        val = self.main_report_template.get('model_report')
        if val is None: return None
        return val.to_dict('index') if hasattr(val, 'to_dict') else val

    def _add_single_results_to_yaml(self, report: dict):
        if not self.main_report_template: return

        met_report = self.main_report_template.get('metrics_report')
        if met_report is not None:
            report['metrics_report'] = met_report.to_dict('index')
        
        report['test_outlier_summary'] = self._get_test_summary_dict(self.main_report_template.get('test_outlier_summary'))
        
        if self.main_report_template.get('qsar_metrics'):
            report['qsar_metrics'] = self.main_report_template.get('qsar_metrics')

        for cm_key in ['confusion_matrix_global', 'confusion_matrix_inside_AD', 'confusion_matrix_outside_AD']:
            if self.main_report_template.get(cm_key):
                report[cm_key] = self.main_report_template.get(cm_key)

    def _get_test_summary_dict(self, summary):
        if summary is None: return {}
        if isinstance(summary, dict): return summary
        if not summary.empty: return summary.to_dict().get('Value', {})
        return {}

    def _serialize_outlier_summary(self, val) -> dict:
        """
        Converts the 'test_outlier_summary' table (Raw vs OOF AD coverage +
        prediction-error rates, built by 'pipeline._summarize_outliers') into a
        nested dict keyed by AD strategy, so it survives YAML serialization
        instead of collapsing to '{}'.

        Falls back to the single-column 'Value' Series format used elsewhere
        in this class (e.g. single-run 'test_outlier_summary'), and to a
        'Method'-indexed table if present, for backward compatibility.

        Args:
            val: Either a pd.DataFrame (with an 'AD Strategy' or 'Method' column),
                 a pd.DataFrame with a single 'Value' column, or a plain dict.

        Returns:
            dict: JSON/YAML-serializable nested dictionary.
        """
        if val is None:
            return {}
        if isinstance(val, dict):
            return val
        if not hasattr(val, 'columns'):
            return {}

        if 'AD Strategy' in val.columns:
            return val.set_index('AD Strategy').to_dict('index')
        if 'Method' in val.columns:
            return val.to_dict('index')
        if not val.empty:
            return val.to_dict().get('Value', {})
        return {}
    
    def _get_serializable_external_report(self) -> dict:
        ext = self.external_test_cv_report
        clean = {}
        if 'dataset_summary' in ext: 
            clean['dataset_summary'] = ext['dataset_summary'].to_dict('index')
        if 'metrics_report' in ext: 
            clean['metrics_report'] = ext['metrics_report'].to_dict('index')
        if 'test_outlier_summary' in ext:
            clean['test_outlier_summary'] = self._serialize_outlier_summary(ext['test_outlier_summary'])
        
        # Capture nested QSAR metric dictionaries if passed in the external dict
        if 'qsar_metrics_raw' in ext:
            clean['qsar_metrics_raw'] = ext['qsar_metrics_raw']
        if 'qsar_metrics_oof' in ext:
            clean['qsar_metrics_oof'] = ext['qsar_metrics_oof']

        # Flag-multiplicity ("Flagged by N metrics") and exclusive single-metric counts
        for key in ['flag_multiplicity_raw', 'flag_multiplicity_oof',
                    'exclusive_flags_raw', 'exclusive_flags_oof']:
            if ext.get(key):
                clean[key] = ext[key]
    
        return clean

    # ─────────────────────────────────────────────────────────────────────────
    # 2. DOCX Report Generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_docx(self):
        """Generates a human-readable DOCX report."""
        docx_path = os.path.join(self.output_dir, f'{self.prefix}.docx')
        try:
            self.document = Document()
            self.document.add_heading('Applicability Domain (AD) Analysis Report', 0)
            
            self._add_analysis_setup_to_docx()
            if not self.is_cv_run and self.main_report_template:
                self._add_df_to_docx(self.main_report_template.get('model_report'), 'AD Model Parameters')
            
            self._add_df_to_docx(self.main_dataset_summary, 'Overall Dataset Summary')

            # Add Dual Thresholds explicitly to the Word doc
            if self.raw_thresholds or self.oof_thresholds:
                self._add_thresholds_to_docx()

            if not self.is_cv_run:
                self._add_single_results_to_docx()

            if self.external_test_cv_report:
                self._add_external_results_to_docx()

            self.document.save(docx_path)
            print(f"DOCX report saved to: {docx_path}")
        except Exception as e:
            print(f"Error generating DOCX report: {e}")

    def _add_analysis_setup_to_docx(self):
        doc = self.document
        doc.add_heading('Analysis Setup', level=1)
        p = doc.add_paragraph()
        p.add_run('Applicability Domain Model:').bold = True
        p.add_run(f"\n  - Descriptors: {self.config.get('descriptors', 'N/A')}")
        p.add_run(f"\n  - Model: {self.config.get('model', 'N/A')}")
        p.add_run(f"\n  - N Components: {self.config.get('n_components', 'N/A')}")
        
        if self.config.get('use_qsar', False):
            p.add_run('\nQSAR Model:').bold = True
            p.add_run(f"\n  - Algorithm: {self.config.get('qsar_model', {}).get('type', 'random_forest')}")
            p.add_run(f"\n  - Parameters: {self.config.get('qsar_model', {}).get('params', {})}")
        
        if self.is_cv_run:
            self._add_clustering_setup_to_docx(p)

    def _add_clustering_setup_to_docx(self, paragraph):
        algo = self.config.get('cluster_algorithm', 'N/A')
        paragraph.add_run('\nClustering Setup:').bold = True
        paragraph.add_run(f"\n  - Algorithm: {algo}")
        paragraph.add_run(f"\n  - N Clusters: {self.config.get('n_clusters', 'N/A')}")

        if algo.lower() in ['kmeans', 'hierarchical', 'dbscan']:
            desc = self.config.get('cluster_descriptors') or self.config.get('descriptors', 'N/A')
            model = self.config.get('cluster_model') or self.config.get('model', 'N/A')
            paragraph.add_run(f"\n  - Descriptors: {desc}")
            paragraph.add_run(f"\n  - Model: {model}")
        elif algo.lower() in ['maxmin', 'scaffolds']:
            paragraph.add_run(f"\n  - Based on: Molecular Fingerprints/Scaffolds")
        elif algo.lower() == 'atc':
            paragraph.add_run(f"\n  - Based on: ATC Level {self.config.get('atc_level', 'N/A')}")

    def _add_thresholds_to_docx(self):
        self.document.add_heading('Calculated AD Thresholds', level=1)
        
        # Convert threshold dicts to a DataFrame for easy table generation
        thresh_data = {}
        all_keys = set()
        if self.raw_thresholds:
            thresh_data['Raw (Training)'] = self.raw_thresholds
            all_keys.update(self.raw_thresholds.keys())
        if self.oof_thresholds:
            thresh_data['OOF (Cross-Validation)'] = self.oof_thresholds
            all_keys.update(self.oof_thresholds.keys())
            
        if not thresh_data: return

        df_thresh = pd.DataFrame(index=list(all_keys))
        for col_name, data_dict in thresh_data.items():
            df_thresh[col_name] = pd.Series(data_dict)
            
        df_thresh.index.name = 'Metric'
        self._add_df_to_docx(df_thresh, 'Raw vs. OOF Boundaries')

    def _add_single_results_to_docx(self):
        if not self.main_report_template: return
        doc = self.document
        doc.add_heading('Analysis Results', level=1)
        self._add_df_to_docx(self.main_report_template.get('metrics_report'), 'AD Metrics & Outliers')
        self._add_df_to_docx(self.main_report_template.get('test_outlier_summary'), 'Test Set Outlier Summary')
        
        qsar = self.main_report_template.get('qsar_metrics')
        if qsar:
            self._add_metrics_table(qsar.get('global', {}), "QSAR Global Model Metrics")
            self._add_metrics_table(qsar.get('inside_ad', {}), "QSAR Metrics Inside AD")
            self._add_metrics_table(qsar.get('outside_ad', {}), "QSAR Metrics Outside AD")

    def _add_external_results_to_docx(self):
        doc = self.document
        doc.add_heading('External Test Set Performance', level=1)
        ext = self.external_test_cv_report
        doc.add_paragraph("Evaluation of the master AD model against the independent external test set.")
        self._add_df_to_docx(ext.get('test_outlier_summary'), 'Test Set Outlier Summary (External)')
        
        if 'qsar_metrics_raw' in ext and ext['qsar_metrics_raw']:
            doc.add_heading('QSAR Metrics (Raw Training Thresholds)', level=2)
            self._add_metrics_table(ext['qsar_metrics_raw'].get('inside_ad', {}), "Inside AD")
            self._add_metrics_table(ext['qsar_metrics_raw'].get('outside_ad', {}), "Outside AD")

        if 'qsar_metrics_oof' in ext and ext['qsar_metrics_oof']:
            doc.add_heading('QSAR Metrics (OOF Thresholds)', level=2)
            self._add_metrics_table(ext['qsar_metrics_oof'].get('inside_ad', {}), "Inside AD")
            self._add_metrics_table(ext['qsar_metrics_oof'].get('outside_ad', {}), "Outside AD")

        if ext.get('flag_multiplicity_raw') or ext.get('flag_multiplicity_oof'):
            doc.add_heading('AD Metric Flagging Diagnostics', level=2)
            self._add_metrics_table(ext.get('flag_multiplicity_raw', {}), "Flag Multiplicity per Compound (Raw)")
            self._add_metrics_table(ext.get('flag_multiplicity_oof', {}), "Flag Multiplicity per Compound (OOF)")
            self._add_metrics_table(ext.get('exclusive_flags_raw', {}), "Compounds Flagged Exclusively by One Metric (Raw)")
            self._add_metrics_table(ext.get('exclusive_flags_oof', {}), "Compounds Flagged Exclusively by One Metric (OOF)")
            
    def _add_df_to_docx(self, df: Optional[pd.DataFrame], title: str):
        """Helper to add a DataFrame as a table."""
        if df is None or df.empty: return
        
        self.document.add_heading(title, level=2)
        df_for_table = df.reset_index()
        table = self.document.add_table(rows=1, cols=len(df_for_table.columns), style='Table Grid')
        
        for j, col_name in enumerate(df_for_table.columns):
            table.rows[0].cells[j].text = str(col_name)
            
        for _, row in df_for_table.iterrows():
            row_cells = table.add_row().cells
            for j, value in enumerate(row.values):
                if isinstance(value, float):
                    row_cells[j].text = f"{value:.3f}"
                else:
                    row_cells[j].text = str(value)

    def _add_metrics_table(self, metrics_dict, title):
        if not metrics_dict: return
        self.document.add_heading(title, level=3)
        table = self.document.add_table(rows=1, cols=2, style='Table Grid')
        table.rows[0].cells[0].text, table.rows[0].cells[1].text = 'Metric', 'Value'
        for k, v in metrics_dict.items():
            if k in ['TP', 'TN', 'FP', 'FN']: continue
            row = table.add_row().cells
            row[0].text, row[1].text = k, f"{v:.3f}" if isinstance(v, float) else str(v)


def generate_report(report_data, output_dir, prefix, config, **kwargs):
    """
    Legacy entry point maintained for backward compatibility.
    Accepts kwargs like raw_thresholds, oof_thresholds.
    """
    generator = ReportGenerator(
        output_dir=output_dir,
        prefix=prefix,
        config=config,
        report_data=report_data,
        **kwargs
    )
    generator.generate()