"""
Context management for ADTest

Author: Eric March (eric.march@upf.edu)
Created: 2026-04-30
Refactored: 2026-05-04
"""

import argparse
import os
import yaml
import sys

from .pipeline import ADTestPipeline
from .utils import utils as u
from .utils.logger import LOG


class ADTestContext:
    """
    Application context and CLI dispatcher for ADTest.

    This class orchestrates the execution of different commands (run, config, report)
    and manages the translation of CLI arguments into pipeline configurations.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_output_dir = args.output or 'results'
        self.logger = None

    def execute(self):
        """Dispatches the command to the appropriate handler."""
        if self.args.command == 'run':
            self.run_cmd()
        else:
            sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Command Handlers
    # ─────────────────────────────────────────────────────────────────────────

    def run_cmd(self):
        """
        Handler for '-c run'. Supports YAML-driven and direct-CLI modes.
        """
        # --- MODE 1: YAML file ---
        if self.args.config:
            self._run_yaml_mode()
            return

        # --- MODE 2: Direct CLI flags ---
        os.makedirs(self.base_output_dir, exist_ok=True)
        self.logger = LOG(
            name='adtest_bootstrap',
            log_file_path=os.path.join(self.base_output_dir, 'adtest_bootstrap.log'),
            quiet=getattr(self.args, 'quiet', False)
        )

        if not self._validate_run_args():
            sys.exit(1)

        dataset_config = self._build_config_from_args()
        self._process_dataset_list([dataset_config])

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Private Orchestration Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _run_yaml_mode(self):
        """Handles execution when a YAML configuration file is provided."""
        if not os.path.isfile(self.args.config):
            print(f"FATAL: Configuration file not found at '{self.args.config}'")
            sys.exit(1)

        try:
            with open(self.args.config, 'r') as fh:
                master_config = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            print(f"FATAL: Error parsing YAML configuration file: {exc}")
            sys.exit(1)

        if not isinstance(master_config, dict):
            print("FATAL: Configuration file is empty or invalid.")
            sys.exit(1)

        datasets = master_config.get('datasets') or []
        if not isinstance(datasets, list):
            print("FATAL: 'datasets' must be a list in the configuration file.")
            sys.exit(1)

        self.base_output_dir = master_config.get('base_output_dir', self.base_output_dir)
        global_quiet = master_config.get('quiet', False)
        
        for ds in datasets:
            if 'quiet' not in ds:
                ds['quiet'] = global_quiet

        self._process_dataset_list(datasets)

    def _process_dataset_list(self, datasets: list):
        """Iterates over a list of dataset configs and runs the pipeline."""
        for dataset_config in datasets:
            dataset_name = dataset_config.get('name', 'dataset')

            output_dir = os.path.join(self.base_output_dir, dataset_name)
            os.makedirs(output_dir, exist_ok=True)

            filename_prefix = u.generate_filename_prefix(dataset_config)
            log_file = os.path.join(output_dir, f'{filename_prefix}.log')
            report_base_path = os.path.join(output_dir, f'{filename_prefix}')

            dataset_logger = LOG(name=f"{dataset_name}", log_file_path=log_file, quiet=dataset_config.get('quiet', False))

            try:
                pipeline = ADTestPipeline(
                    config=dataset_config,
                    logger=dataset_logger,
                    filename_prefix=filename_prefix,
                    report_base_path=report_base_path
                )
                pipeline.run()
            except Exception as exc:
                dataset_logger.critical(
                    f"Critical error processing {dataset_name}: {exc}",
                    exc_info=True
                )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Private Translation & Validation Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _build_config_from_args(self) -> dict:
        """Translates argparse Namespace into the pipeline's nested config dict."""
        args = self.args
        qsar_config = {}
        if args.use_qsar:
            qsar_config = {
                'type':   args.qsar_algorithm or 'random_forest',
                'params': {}
            }

        return {
            'name':              args.endpoint or 'dataset',
            'train_path':        args.train,
            'test_path':         getattr(args, 'test', None),
            'id_col':            args.id_col or 'ID',
            'smiles_col':        args.smiles_col or 'SMILES',
            'activity_col':      args.activity_col or 'activity',
            'prediction_column': args.prediction_col or False,
            'descriptors':       args.descriptors or 'rdkit',
            'model':             args.model or 'pca',
            'n_components':      args.n_components,
            'alpha':             args.alpha or 0.05,
            'use_qsar':          args.use_qsar,
            'qsar_model':        qsar_config,
            'cluster_analysis':    (args.action == 'cv'),
            'cluster_algorithm':   args.cluster_algorithm or 'kmeans',
            'cluster_descriptors': getattr(args, 'cluster_descriptors', None),
            'n_clusters':          args.n_clusters,
            'n_splits':            args.n_splits or 5,
            'atc_level':           args.atc_level or 1,
            'generate_plots':      not getattr(args, 'no_plots', False),
            'threshold_strategy':  getattr(args, 'threshold_strategy', 'balanced'),
            'threshold_weight':    getattr(args, 'threshold_weight', 0.5),
            'tiered_ad':           getattr(args, 'tiered_ad', False),
            'quiet':               getattr(args, 'quiet', False),
        }