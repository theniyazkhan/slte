# Can Synthetic Data Replace Real Data? Evaluating the Efficacy, Limits, and Label Trustworthiness of Generative Data in Supervised Learning

This repository contains the official implementation, replication package, and experimental results for the **SLTE (Synthetic Label Trustworthiness Estimation)** framework. This research was presented at the *International Conference on Emerging Frontiers in Advanced Sciences and Technologies (EFAST 2026)* and is currently under review at the *Arabian Journal for Science and Engineering (AJSE)*.

---

## 📌 Project Overview
While synthetic tabular data generation (e.g., via CTGAN) offers a promising avenue for privacy-preserving AI in sensitive domains like healthcare, its limits in supervised learning are often overlooked. This study systematically evaluates the Train-Synthetic-Test-Real (TSTR) protocol across multiple mixing ratios, empirical controls, and machine learning classifiers.

### Key Findings
* **The 20% Threshold:** Mixing synthetic data remains reliable up to a 20% ratio; performance degradation accelerates significantly as synthetic proportions increase beyond 60%.
* **Directional Label Asymmetry:** Systematic label evaluation demonstrates that mislabeled generative samples follow directional flip patterns driven by reference-set neighborhood dynamics in imbalanced tabular distributions.
* **The SLTE Framework:** SLTE estimates sample-level label trustworthiness and filters out corrupted or destabilizing synthetic instances, recovering lost downstream model performance across standard classifiers.

---

## 📂 Repository Structure

```text
├── data/
│   ├── MedicalAppointment.csv        # Baseline public dataset (No-Show records)
│   ├── synthetic_full.csv            # Full synthetic dataset generated via CTGAN
│   └── lcs_distribution.csv          # Label Trustworthiness / LCS distribution scores
│
├── notebooks/
│   ├── Initial_Experiments.ipynb     # TSTR baseline and multi-ratio mixing experiments
│   ├── SLTE_Pipeline.ipynb           # Core SLTE implementation, evaluation, and filtering
│   └── Analysis_Figures.ipynb        # Visualization generation and result aggregation
│
├── results/
│   ├── tstr_results.csv / .json      # Comprehensive raw performance metrics across classifiers
│   ├── baseline_results.json         # Real-data-only control baseline metrics
│   ├── slte_results.csv              # Model performance after applying SLTE filtering
│   ├── delta_slte_vs_noSlte.csv      # Comparative performance recovery analysis
│   └── Paper_Results_All_Tables.xlsx # Compiled tables used in the final manuscript
│
├── revision/                         # AJSE journal revision suite & ablation experiments
│   ├── revise_slte.ipynb             # Master revision notebook (pipeline, controls & ablations)
│   ├── synthetic_nolog.csv           # Unlogged synthetic generation benchmark data
│   ├── slte_all_ratios.csv           # Extended evaluation across fine-grained mixing ratios
│   ├── real_only_learning_curve.csv  # Sample-efficiency learning curve benchmarks on real data
│   ├── reference_set_ablation.csv    # Sensitivity analysis on reference set size and balance
│   ├── stage_ablation_results.csv    # Ablation benchmarks isolating individual SLTE components
│   ├── seed_variance_results.csv     # Multi-seed stability and variance test results
│   └── dcr_privacy_results.csv       # Distance to Closest Record (DCR) privacy evaluation
│
└── README.md
