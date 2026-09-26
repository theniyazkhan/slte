"""Figures 1 to 5 of the manuscript, drawn from the revision-2 outputs in the same style as the
first-revision figures (matplotlib defaults, as in revise_slte.ipynb)."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_real, load_synth
from slte_revision2 import SLTEReference

t = pd.read_csv('tstr_primary.csv'); ro = pd.read_csv('realonly_primary.csv')
RAT = ['0%', '20%', '40%', '60%', '80%', '100%']


def series(clf, mode, metric, ratios=RAT):
    q = t[(t.Classifier == clf) & (t.Mode == mode)].set_index('Ratio').loc[ratios]
    return q[metric].values


# ---- Fig. 1: LCS distribution
SF = 'synthetic_full.csv' if os.path.exists('synthetic_full.csv') else 'synthetic/synthetic_full.csv'
X_train, X_test, y_train, y_test = load_real(); sX, sy = load_synth(SF, X_train.columns)
_, _, lcs = SLTEReference(X_train, y_train, sX).apply(sy.values.astype(int))
plt.figure(figsize=(6, 4), dpi=300)
plt.hist(lcs, bins=50, color='#2ca02c', edgecolor='black', alpha=0.7)
plt.axvline(0.40, color='red', linestyle='--', linewidth=1.5, label='kNN Correction Cutoff (0.40)')
plt.axvline(0.70, color='blue', linestyle='--', linewidth=1.5, label='High-Weight Cutoff (0.70)')
plt.xlabel('Label Confidence Score (LCS)', fontsize=11)
plt.ylabel('Synthetic Sample Count', fontsize=11)
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout(); plt.savefig('fig1_lcs.png', dpi=300, bbox_inches='tight'); plt.close()

# ---- Fig. 2: precision-recall curve of Gradient Boosting on 100% real data
recalls, precisions = np.load('gb_pr_curve_primary.npy')
import json
op = json.load(open('out_primary.json'))['gb_operating_point']
plt.figure(figsize=(5, 4), dpi=300)
plt.plot(recalls, precisions, linewidth=2, label=f"Gradient Boosting (AP={op['ap']:.3f})")
plt.axhline(op['base_rate'], linestyle='--', color='gray', label=f"Random baseline ({op['base_rate']:.3f})")
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.legend(loc='upper right', fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('fig2_pr.png', dpi=300, bbox_inches='tight'); plt.close()

# ---- Fig. 3: AUC-ROC across mixing ratios, with (solid) and without (dashed) SLTE
clfs = ['Decision Tree', 'Gradient Boosting', 'Logistic Regression', 'Naive Bayes', 'Random Forest']
plt.figure(figsize=(7.2, 5.4), dpi=300)
for clf in clfs:
    plt.plot(RAT, series(clf, 'SLTE', 'AUC'), marker='o', label=f'{clf} (SLTE)')
    plt.plot(RAT, series(clf, 'No-SLTE', 'AUC'), marker='x', linestyle='--', alpha=0.6, label=f'{clf} (No-SLTE)')
plt.xlabel('Synthetic Data Ratio', fontsize=11); plt.ylabel('AUC Score', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left', ncol=2, fontsize=8)
plt.tight_layout(); plt.savefig('fig3_auc.png', dpi=300, bbox_inches='tight'); plt.close()

# ---- Fig. 4: minority-class F1 across mixing ratios, with (solid) and without (dashed) SLTE
style = {'Decision Tree': ('#1f77b4', 'o'), 'Random Forest': ('#bcbd22', 's'), 'Gradient Boosting': ('#2ca02c', '^'),
         'Logistic Regression': ('#9467bd', 'D'), 'Naive Bayes': ('#e377c2', 'v')}
plt.figure(figsize=(6.8, 4.5), dpi=300)
for clf, (col, mk) in style.items():
    plt.plot(RAT, series(clf, 'SLTE', 'F1'), color=col, marker=mk, linewidth=2, label=f'{clf} (SLTE)')
    plt.plot(RAT, series(clf, 'No-SLTE', 'F1'), color=col, marker='x', linestyle='--', alpha=0.75, label=f'{clf} (No-SLTE)')
plt.xlabel('Synthetic Data Ratio', fontsize=11); plt.ylabel('F1 Score', fontsize=11)
plt.ylim(-0.02, 0.52)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig('fig4_f1.png', dpi=300, bbox_inches='tight'); plt.close()

# ---- Fig. 5: Gradient Boosting, real-only training against mixed training at matched real-record counts
R5 = ['0%', '20%', '40%', '60%', '80%']
gb_real = ro[ro.Classifier == 'Gradient Boosting'].set_index('Ratio').loc[R5]['AUC'].values
plt.figure(figsize=(7, 4.5), dpi=300)
plt.plot(R5, series('Gradient Boosting', 'SLTE', 'AUC', R5), marker='o', color='#2ca02c', label='GB (SLTE Mix)')
plt.plot(R5, series('Gradient Boosting', 'No-SLTE', 'AUC', R5), marker='x', linestyle='--', color='#d62728', label='GB (No-SLTE Mix)')
plt.plot(R5, gb_real, marker='s', linestyle='-.', color='#1f77b4', label='GB (Real-Only Subsample)')
plt.xlabel('Synthetic Ratio in Mixture / Real Data Reduction', fontsize=11)
plt.ylabel('AUC-ROC Score', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left', fontsize=9)
plt.tight_layout(); plt.savefig('fig5_realonly.png', dpi=300, bbox_inches='tight'); plt.close()
print('figures written')
