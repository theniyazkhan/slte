# Synthetic Data Efficacy in Supervised Learning (SLTE)

This repository contains the complete replication code, preprocessing pipelines, experimental benchmarks, and journal revision artifacts for evaluating synthetic tabular data generation (CTGAN) and downstream supervised learning efficacy.

---

## Dataset

Experiments are conducted on the [Medical Appointment No-Shows Dataset](https://www.kaggle.com/datasets/joniarroba/noshowappointments) (`MedicalAppointment.csv`):
* **Sample Size**: 110,527 patient appointment records
* **Class Distribution**: ~4:1 class imbalance (No-Show vs. Show)
* **Features Extracted**: Patient demographics, scheduling lead time, appointment temporal features (`ScheduledMonth`, day of week), medical conditions, and SMS reminders.

---

## Repository Structure

```text
├── MedicalAppointment.csv          # Raw Kaggle source dataset (110,527 records)
├── Initial_Experiments.ipynb       # Exploratory data analysis and baseline exploratory runs
├── SLTE_Pipeline.ipynb             # Core pipeline: preprocessing, CTGAN generation, TSTR evaluation
├── Analysis_Figures.ipynb          # Reproduction scripts for all manuscript figures and tables
├── baseline_results.json           # Baseline model performances on real-only data
├── day2_summary_log.json           # Execution logs and model training metadata
├── delta_slte_vs_noSlte.csv        # Performance delta (Δ) between SLTE and baseline configurations
├── key_numbers_for_paper.json      # Central statistics and numerical values reported in the paper
├── lcs_distribution.csv            # Longest common subsequence metrics for sequence similarity
├── results_pivot_accuracy.csv      # Formatted accuracy matrix across models and runs
├── slte_results.csv                # Primary SLTE experimental results across classifier suites
├── synthetic_full.csv              # Full synthetic dataset generated via CTGAN
├── tstr_results.csv                # Train on Synthetic, Test on Real (TSTR) benchmark metrics
├── tstr_results.json               # Serialized TSTR evaluation outputs
│
└── revision/                       # Journal revision suite: statistical controls, ablations, & privacy audits
    ├── revise_slte.ipynb           # Master execution notebook for all revision experiments
    ├── dcr_privacy_results.csv     # Distance to Closest Record (DCR) privacy evaluation results
    ├── real_only_learning_curve.csv# Sample efficiency curve across real-data subsample sizes
    ├── reference_set_ablation.csv  # Sensitivity analysis across reference set sample sizes
    ├── seed_variance_results.csv   # Stability across multiple random initialization seeds
    ├── slte_all_ratios.csv         # Augmentation sweep across synthetic-to-real ratios
    ├── stage_ablation_results.csv  # Component-wise ablation of pipeline processing stages
    └── synthetic_nolog.csv         # Synthetic benchmark generated without log transformations
