# Synthetic Data Efficacy in Supervised Learning (SLTE)

Code, data and results for the paper "Can Synthetic Data Replace Real Data? Evaluating the Efficacy,
Limits, and Label Trustworthiness of Generative Data in Supervised Learning".

The paper tests how far synthetic tabular data (CTGAN, and in the second revision also TVAE and a
Gaussian copula) can replace real training data under the train-on-synthetic, test-on-real (TSTR)
protocol. It also proposes Synthetic Label Trustworthiness Estimation (SLTE), a check that scores each
synthetic label against a model trained on real data, relabels the least consistent samples and
down-weights uncertain ones.

---

## Start here: `revision2/` (current version)

[`revision2/`](revision2/) regenerates every number, table and figure of the current manuscript
(second revision). Its [README](revision2/README.md) lists every command, the generation runs and the
output file behind each table and figure.

```bash
git clone https://github.com/theniyazkhan/slte.git
cd slte/revision2
pip install -r requirements_analysis.txt
python make_figures.py        # redraws Figures 1 to 5 from the saved results
```

The earlier material is kept unchanged as the record of the earlier versions:

| Paper version | Folder |
|---|---|
| Second revision (current) | `revision2/` |
| First revision | `revision/` |
| First submission | notebooks and result files in the repository root |

`revision2/` uses the same synthetic set as the earlier versions (`synthetic_full.csv`, byte-identical).
With its pinned package versions it reproduces the first-revision results exactly for Naive Bayes,
Decision Tree, Random Forest and Gradient Boosting (Logistic Regression within 0.0002 AUC-ROC).

---

## Datasets

**Medical Appointment No-Shows** (primary dataset, `MedicalAppointment.csv`, from
[Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments), under the licence stated there)
* 110,527 appointments: 88,208 attended and 22,319 no-shows (20.2%), about 4 attended per no-show
* One record with age -1 is removed, leaving 110,526
* 12 predictors: gender, age, neighbourhood (frequency-encoded), scholarship, hypertension, diabetes,
  alcoholism, handicap, SMS reminder, waiting days, and the weekdays of scheduling and of the appointment
* Stratified 80/20 split with `random_state = 42`: 88,420 training and 22,106 test records

**UCI Adult** (second dataset, added in the second revision, `revision2/adult/adult-all.csv`)
* 48,842 records, income above 50K for 23.9%
* Stratified 80/20 split with `random_state = 42`: 39,073 training and 9,769 test records
* B. Becker and R. Kohavi, UCI Machine Learning Repository, https://doi.org/10.24432/C5XW20 (CC BY 4.0)

---

## Repository structure

### `revision2/`: second revision (current)

| File | Contents |
|---|---|
| `README.md` | Commands, environments, and the output file behind each table and figure |
| `common.py`, `adult_common.py` | Data loading, train/test split and the five classifiers (no-show and Adult data) |
| `gen_common.py`, `gen_run.py` | Seeded, checkpointed synthetic data generation (CTGAN, TVAE, Gaussian copula) |
| `slte_revision2.py` | SLTE and every analysis of the current manuscript |
| `extras.py` | Real-only versus mixed DeLong tests for every generation run |
| `make_figures.py` | Figures 1 to 5 |
| `requirements_analysis.txt`, `requirements_generation.txt` | Pinned package versions |
| `results/` | Output files behind every number in the manuscript, and the five figures |
| `synthetic/` | The ten synthetic sets used in the paper |
| `adult/adult-all.csv` | UCI Adult data |

### `revision/`: first revision

| File | Contents |
|---|---|
| `revise_slte.ipynb` | All first-revision experiments |
| `slte_all_ratios.csv` | TSTR with and without SLTE at all six mixing ratios |
| `seed_variance_results.csv` | Classifier seeds 42, 1 and 2 |
| `reference_set_ablation.csv` | SLTE with smaller real reference sets |
| `stage_ablation_results.csv` | SLTE component ablation at 100% synthetic data |
| `real_only_learning_curve.csv` | Real-only training at reduced real-record counts |
| `dcr_privacy_results.csv` | Distance to closest record (DCR) |
| `synthetic_nolog.csv` | CTGAN set generated with `log_frequency = False` (conditional sampling without log-frequency weighting; the data themselves are not log-transformed) |

### Repository root: first submission (notebooks run in this order)

| File | Contents |
|---|---|
| `Initial_Experiments.ipynb` | Step 1: preprocessing, CTGAN training and `synthetic_full.csv`, real-data baseline, TSTR at six mixing ratios |
| `SLTE_Pipeline.ipynb` | Step 2: SLTE (Label Confidence Score, kNN relabelling, confidence-based weights) and TSTR with SLTE |
| `Analysis_Figures.ipynb` | Step 3: figures, tables and key numbers of the first submission |
| `MedicalAppointment.csv` | Kaggle no-show data (110,527 records), used by every version |
| `synthetic_full.csv` | CTGAN synthetic set of the primary run (88,420 records; run C0 in the second revision) |
| `tstr_results.csv`, `tstr_results.json` | TSTR metrics without SLTE |
| `slte_results.csv` | TSTR metrics with SLTE |
| `delta_slte_vs_noSlte.csv` | AUC-ROC with and without SLTE, and the difference |
| `lcs_distribution.csv` | Label Confidence Score (LCS), original and corrected label, confidence track and weight of each synthetic record |
| `baseline_results.json` | The five classifiers trained on 100% real data |
| `results_pivot_accuracy.csv` | Accuracy by mixing ratio and classifier, without SLTE |
| `day2_summary_log.json` | SLTE summary: LCS mean and SD, records per confidence track, labels corrected |
| `key_numbers_for_paper.json` | Numbers quoted in the first submission |
