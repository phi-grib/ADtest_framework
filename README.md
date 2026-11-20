# ADtest

A package for robust Applicability Domain (AD) analysis in chemoinformatics and QSAR modeling.

Applicability Domain analysis defines the boundaries within which a QSAR model's predictions can be trusted, by comparing new (test) compounds against the chemical space covered by the training set. This package implements multiple AD metrics (DModX, Hotelling's $T^2$, k-NN distance, LOF, KDE, Isolation Forest, Autoencoder reconstruction error, ADSal) combined with a clustering-based, stratified cross-validation strategy to produce both conventional (training-set) and out-of-fold (OOF) AD thresholds, reducing resubstitution bias and tautology in the AD boundary estimation.

## 🚀 Getting Started

### ⚙️ Prerequisites

* **Operating System**: Developed and tested exclusively on **Ubuntu Linux** (tested on **Ubuntu 24.04 LTS** and **26.04 LTS**). Compatibility with **macOS** (likely functional via Conda) and **Windows** has not been tested and is provided without warranty.
* **Python 3.10**: the project is built for this specific Python version.
* **Conda**: a package and environment management system, highly recommended.

### 🛠️ Installation

1. **Create a Conda environment** using the `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate adtest_framework
```

2. **Install the package** in editable/development mode. This makes the `adtest` CLI command available:

```bash
pip install -e .
```

## 💻 Usage

`ADtest` is driven entirely through a YAML configuration file — there is no direct-CLI-flags mode. Each YAML defines one or more datasets to process (train/test paths, descriptors, AD model, and clustering/cross-validation settings), and the pipeline runs end-to-end: data loading, descriptor calculation, AD model fitting, (optional) clustered cross-validation for OOF thresholds, and report generation (YAML + DOCX + diagnostic plots).

### Command

```bash
adtest -c run -cfg <path_to_config>.yaml
```

| Flag | Required | Description |
| :--- | :--- | :--- |
| `-c`, `--command` | Yes | Command to execute. Currently only `run` is supported. |
| `-cfg`, `--config` | Yes | Path to the YAML configuration file describing the dataset(s) and analysis settings. |
| `-o`, `--output` | No | Base output directory for results, logs, plots, and reports (default: `results/`). Each dataset/endpoint gets its own subdirectory: `<output>/<endpoint>/`. |
| `-q`, `--quiet` | No | Suppress standard logging output. |

### Example

```bash
adtest -c run -cfg ad_ames_test.yaml
```

### Output

For each dataset defined in the YAML, `ADtest` produces, under `<output>/<dataset_name>/`:

* **`<prefix>.yaml`** — machine-readable report: analysis setup, dataset summary, raw and OOF AD thresholds, and external test set performance (AD coverage and prediction-error rates inside/outside the AD, per thresholding strategy).
* **`<prefix>.docx`** — human-readable report with the same information formatted as tables.
* **`<prefix>_test_raw_outliers.csv` / `<prefix>_test_oof_outliers.csv` / `<prefix>_test_combined_outliers.csv`** — full per-compound outlier tables (raw AD metrics, per-metric outlier flags, and total flag counts) for the raw and OOF thresholding strategies.
* **`<prefix>_AD_test_results.xlsx`** — multi-sheet Excel workbook containing the combined outlier results and summaries.
* **Diagnostic plots** (PCA score plots, fold disjointness, traffic-light validation plots), unless `generate_plots: false` is set in the config.
* **`<prefix>.log`** — full run log.

### Configuration File

The YAML config accepts (per dataset entry under `datasets:`):

* `name`, `train_path`, `test_path`, `id_col`, `smiles_col`, `activity_col`
* `descriptors` (`rdkit` or `fps`), `model` (`pca`; `pls` is not supported for this workflow)
* `n_components`, `alpha` (percentile for AD thresholds)
* `use_qsar`, `qsar_model` (algorithm + params) or `prediction_column` (pre-computed predictions)
* `cluster_analysis` (enables clustered CV for OOF thresholds), `cluster_algorithm` (`scaffolds` only supported here), `cluster_descriptors`, `n_clusters`, `n_splits`, `atc_level`
* `generate_plots`, `quiet`

See the example YAMLs in the repository for a complete reference.

## 🔬 Reproducibility (Manuscript Tables & Figures)

Because raw intermediate result files and trained QSAR model archives are large, they are archived under a single **Zenodo** DOI rather than hosted directly in this GitHub repository. To reproduce all tables and figures from the manuscript locally from scratch, run the core pipeline across all benchmark endpoints and execute the post-processing scripts located in `postprocessing/` from the **root directory of the repository**.

### 1. Run the Core Pipeline & Organize Outputs

1. Execute the master benchmark YAML configuration (`ad_final_cv_analysis_files.yaml`) to generate per-endpoint outputs under `results/<ENDPOINT>/`:

```bash
adtest -c run -cfg ad_final_cv_analysis_files.yaml -o results/
```

2. Create the aggregated directories expected by the post-processing scripts and gather the per-endpoint YAML reports and combined outlier CSVs:

```bash
mkdir -p results/final_analysis/yaml_dir
mkdir -p results/final_analysis/outlier_tables
mkdir -p results/final_analysis/figures

cp results/*/*.yaml results/final_analysis/yaml_dir/
cp results/*/*_test_combined_outliers.csv results/final_analysis/outlier_tables/
```

### 2. Execute Post-Processing & Figure Generation Scripts

Run the following scripts **in order** from the root directory of the package:

1. **Aggregate Raw vs. OOF Coverage, Relative Risk (RR), and Enrichment Factor (EF):**

   ```bash
   python postprocessing/ad_ood_analysis.py
   ```

   * *Reads:* `results/final_analysis/yaml_dir/*.yaml`
   * *Generates:* `results/final_analysis/outlier_tables/ad_ood_comparison.csv` (long-format table comparing `%Outside`, `%ErrIn`, `%ErrOut`, `RR`, and `EF` across In-domain and Out-of-domain endpoints).

2. **Compute Threshold-Free ROC-AUC Metrics:**

   ```bash
   python postprocessing/ad_auc_analysis.py
   ```

   * *Reads:* `results/final_analysis/outlier_tables/*_test_combined_outliers.csv` and `ad_ood_comparison.csv`
   * *Generates:* `results/final_analysis/outlier_tables/ad_auc_table.csv` and `ad_full_comparison_with_auc.csv` (adds `AUC_flagcount` and per-metric continuous ROC-AUCs).

3. **Generate Master Comparison Tables & Main Figures (Figures 1 & 2):**

   ```bash
   python postprocessing/generate_ad_figures.py
   ```

   * *Reads:* `results/final_analysis/yaml_dir/*.yaml` and `results/final_analysis/outlier_tables/*_test_combined_outliers.csv`
   * *Generates (in `results/final_analysis/figures/`):*
     * `ad_full_comparison_with_auc_mcc_f1.csv` & `ad_group_summary_final.csv` (summary statistics with MCC and F1)
     * `fig1_per_metric_AUC_by_domain.png` (horizontal bar chart of individual AD metric discriminative power)
     * `fig2A_group_summary_in_domain.png` & `fig2B_group_summary_ood.png` (Mean ± SEM bar charts for `%Outside`, `RR`, and `EF`)

4. **Verify Chemical Domain Shift ($S_{\mathrm{max}}$ Structural Overlap):**

   ```bash
   python postprocessing/ad_domain_shift_proof.py
   ```

   * *Reads:* `ad_final_cv_analysis_files.yaml` and the raw training/test dataset files
   * *Generates (in `results/final_analysis/figures/`):* `domain_shift_smax_values.csv` and `supfig1_domain_shift_smax_boxplot.png` (Mann–Whitney $U$ verification of Max Tanimoto similarity between In-domain and Out-of-domain benchmarks).

5. **Generate Per-Endpoint Dumbbell Shift Plots (Supplementary Figures 2–5):**

   ```bash
   python postprocessing/generate_dumbbell_plots.py
   ```

   * *Reads:* `results/final_analysis/figures/ad_full_comparison_with_auc_mcc_f1.csv`
   * *Generates (in `results/final_analysis/figures/`):* 2-panel horizontal dumbbell plots showing Raw $\rightarrow$ OOF shifts per endpoint (`supfig2_pctOutside_dumbbell.png`, `supfig3_RR_dumbbell.png`, `supfig4_EF_outside_dumbbell.png`, `supfig5_AUC_dumbbell.png`).

6. **Extract Representative Compound Case Studies (AMES & AChE):**

   ```bash
   python postprocessing/extract_case_study_structures.py
   ```

   * *Reads:* `results/final_analysis/outlier_tables/` (AMES and AChE outlier tables) and the AChE training set
   * *Generates:* `results/final_analysis/AMES_case_studies_table_n6.xlsx`, `results/final_analysis/AChE_case_studies_table_n6.xlsx`, and 2D molecular structure PNGs in `results/final_analysis/case_study_images/`.
