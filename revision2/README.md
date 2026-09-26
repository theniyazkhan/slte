# Second Revision Execution Suite (`revision2/`)

This directory contains the self-contained execution scripts, synthetic data generators, analysis suite, and figure reproduction scripts for the second journal revision of **Synthetic Data Efficacy in Supervised Learning (SLTE)**.

---

## Quick Start

### 1. Environment Setup

Install analysis dependencies:
```bash
pip install -r requirements_analysis.txt
```

For synthetic data generation (CTGAN / TVAE / Gaussian Copula):
```bash
pip install -r requirements_generation.txt
```

---

## Reproducing Manuscript Figures

To regenerate Figures 1 through 5 in PNG format:
```bash
python make_figures.py
```
Outputs are written to `results/` (and saved locally as `fig1_lcs.png` through `fig5_realonly.png`):
- **Fig. 1**: Label Confidence Score (LCS) distribution & threshold cuts.
- **Fig. 2**: Precision-Recall curve for Gradient Boosting baseline on real data.
- **Fig. 3**: AUC-ROC across synthetic mixing ratios (with vs without SLTE).
- **Fig. 4**: Minority class F1 score across synthetic mixing ratios.
- **Fig. 5**: Real-only subsample learning curve vs. synthetic mixture training.

---

## Running the Evaluation Benchmark

The primary evaluation script is `slte_revision2.py`.

### Syntax:
```bash
python slte_revision2.py <synthetic_csv> <tag> <mode> [medical|adult]
```

### Parameters:
- `<synthetic_csv>`: Path to the synthetic dataset file (e.g., `synthetic/synthetic_full.csv`).
- `<tag>`: Output file prefix tag (e.g., `primary`, `medical_ctgan_s1`, `adult_ctgan_s1`).
- `<mode>`: Execution level:
  - `full`: Complete analysis suite (all ratios, DeLong tests, DCR, stage statistics, permutation tests).
  - `replica`: Stage statistics, permutation nulls, DCR, held-out check, TSTR ratios, and DeLong tests.
  - `stats`: Fast check for stage statistics, permutation nulls, DCR, and held-out checks only.
- `[dataset]`: Target dataset, either `medical` (default) or `adult`.

### Examples:

Run full evaluation on primary Medical CTGAN synthetic set:
```bash
python slte_revision2.py synthetic/synthetic_full.csv primary full medical
```

Run evaluation on Adult Census CTGAN synthetic set:
```bash
python slte_revision2.py synthetic/synth_adult_ctgan_default_s1.csv adult_ctgan_s1 full adult
```

---

## Running Synthetic Data Generators

Synthetic datasets can be generated or retrained using `gen_run.py`:

```bash
python gen_run.py <dataset> <kind> <seed>
```

- `<dataset>`: `medical` | `adult`
- `<kind>`: `ctgan_default` | `ctgan_nolog` | `tvae` | `copula`
- `<seed>`: Random seed (e.g. `1`, `2`)

---

## Directory Structure

```text
revision2/
├── README.md                   # This documentation file
├── requirements_analysis.txt   # Pinned packages for analysis and plotting
├── requirements_generation.txt # Dependencies for synthetic data generators
├── common.py                   # Data loaders & model suite for Medical dataset
├── adult_common.py             # Data loaders & model suite for Adult dataset
├── slte_revision2.py           # Master execution script for revision evaluation
├── gen_run.py                  # Generative model runner (CTGAN, TVAE, Copula)
├── gen_common.py               # Generative model helpers & checkpointing
├── extras.py                   # Paired DeLong statistical significance tests
├── make_figures.py             # Plotting script for Figures 1 to 5
├── adult/                      # UCI Adult Census Income source dataset (`adult-all.csv`)
├── synthetic/                  # Pre-generated synthetic datasets
└── results/                    # Saved output metrics, DeLong tables, and manuscript figures
```
