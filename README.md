
# Can Synthetic Data Replace Real Data? Evaluating the Efficacy, Limits, and Label Trustworthiness of Generative Data in Supervised Learning

This repository contains the official implementation, replication package, and experimental results for the **SLTE (Synthetic Label Trustworthiness Estimation)** framework. This research was presented at the *International Conference on Emerging Frontiers in Advanced Sciences and Technologies (EFAST 2026)* and is currently under review at the *Arabian Journal for Science and Engineering (AJSE)*.

---

## 📌 Project Overview

While synthetic data generation (e.g., via CTGAN) offers a promising avenue for privacy-preserving AI in sensitive domains like healthcare, its limits in supervised learning are often overlooked. This study systematically evaluates the **Train-Synthetic-Test-Real (TSTR)** protocol across multiple mixing ratios and machine learning classifiers. 

### Key Findings
* **The 20% Threshold:** Mixing synthetic data remains highly reliable and safe up to a **20% ratio**. However, performance degradation becomes severe beyond **60%**.
* **Systematic Label Corruption:** We discovered that mislabeled generative samples do not merely exhibit random noise. Instead, **96.6% of mislabeled synthetic samples** show a systematic, directional bias—running **25.5 percentage points higher** than random noise baselines.
* **The SLTE Solution:** To combat this, we developed **SLTE**, a framework that evaluates and filters out untrustworthy synthetic labels. By implementing SLTE, we successfully **recovered up to 66.6% of lost model performance** across standard classifiers.

---

## 📂 Repository Structure

```text
├── data/
│   ├── MedicalAppointment.csv        # Baseline public dataset (No-Show records)
│   ├── synthetic_full.csv            # Full synthetic dataset generated via CTGAN
│   └── lcs_distribution.csv          # Label Trustworthiness / LCS distribution scores
│
├── notebooks/
│   ├── Initial_Experiments.ipynb        # TSTR baseline and multi-ratio mixing experiments
│   ├── SLTE_Pipeline.ipynb # SLTE implementation, label evaluation, and filtering
│   └── Analysis_Figures.ipynb # Collecting results, figurs
│
├── results/
│   ├── tstr_results.csv / .json      # Comprehensive raw performance metrics across classifiers
│   ├── baseline_results.json         # Real-data-only control baseline metrics
│   ├── slte_results.csv              # Model performance after applying SLTE filtering
│   ├── delta_slte_vs_noSlte.csv      # Comparative performance recovery analysis
│   └── Paper_Results_All_Tables.xlsx # Compiled tables used in the final manuscript
│
└── README.md
