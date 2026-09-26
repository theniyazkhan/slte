"""
SLTE revision 2: single script that regenerates every number reported in the manuscript.

Pinned environment (reproduces the Colab revision run exactly for NB, DT, RF and GB):
    scikit-learn 1.6.1, numpy 2.0.2, scipy 1.15.3, pandas 2.2.2

Usage:
    python slte_revision2.py <synthetic_csv> <tag> <mode> [medical|adult]
    mode = full    : every analysis (primary generation run)
    mode = replica : stage statistics, permutation null, DCR, held-out check, TSTR at all ratios, DeLong
    mode = stats   : stage statistics, permutation null, DCR and held-out check only
"""
import os, sys, json, time, pickle
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
                             average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve)
from common import load_real, load_synth, make_classifiers as _make_classifiers
from adult_common import load_real_adult, load_synth_adult

RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLF_ORDER = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Naive Bayes']
CLF_KW = {}   # set in main(): Logistic Regression on standardised inputs for the Adult data


def make_classifiers(seed=42):
    return _make_classifiers(seed, **CLF_KW)


# ----------------------------------------------------------------- helpers
def ece(y, p, bins=10):
    """Expected calibration error with equal-width bins."""
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def all_metrics(y, p, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {'AUC': roc_auc_score(y, p), 'PR_AUC': average_precision_score(y, p),
            'F1': f1_score(y, pred, zero_division=0), 'MacroF1': f1_score(y, pred, average='macro', zero_division=0),
            'Precision': precision_score(y, pred, zero_division=0), 'Recall': recall_score(y, pred, zero_division=0),
            'Accuracy': accuracy_score(y, pred), 'BalAcc': balanced_accuracy_score(y, pred),
            'Brier': brier_score_loss(y, p), 'ECE': ece(np.asarray(y), p),
            'PosPreds': int(pred.sum()), 'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong(y, pa, pb):
    """Paired DeLong test (Sun and Xu, 2014). Returns AUC_a, AUC_b, diff, CI low, CI high, z, p."""
    y = np.asarray(y); order = np.argsort(-y, kind='mergesort'); ys = y[order]
    m = int((ys == 1).sum()); n = int((ys == 0).sum())
    P = np.vstack([np.asarray(pa)[order], np.asarray(pb)[order]])
    tx = np.array([_midrank(P[r, :m]) for r in range(2)])
    ty = np.array([_midrank(P[r, m:]) for r in range(2)])
    tz = np.array([_midrank(P[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    d = aucs[0] - aucs[1]; se = np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-15))
    z = d / se; p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(d), float(d - 1.96 * se), float(d + 1.96 * se), float(z), float(p)


def holm(pvals):
    p = np.asarray(pvals); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


# --------------------------------------------------------- SLTE components
class SLTEReference:
    """Anchor and kNN fitted on real reference data. Their outputs on a synthetic feature
    matrix do not depend on the synthetic labels, so any label vector can be scored cheaply."""

    def __init__(self, X_ref, y_ref, X_syn):
        sc = StandardScaler().fit(X_ref)
        Xr = pd.DataFrame(sc.transform(X_ref), columns=X_ref.columns, index=X_ref.index)
        Xs = pd.DataFrame(sc.transform(X_syn), columns=X_syn.columns, index=X_syn.index)
        anchor = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42).fit(Xr, y_ref)
        self.proba = anchor.predict_proba(Xs); self.classes = list(anchor.classes_)
        self.knn_pred = KNeighborsClassifier(n_neighbors=7).fit(Xr, y_ref).predict(Xs)

    def apply(self, labels):
        labels = np.asarray(labels)
        cols = np.array([self.classes.index(v) for v in labels])
        lcs = self.proba[np.arange(len(labels)), cols]
        corrected = labels.copy(); low = lcs < 0.40
        corrected[low] = self.knn_pred[low]
        w = np.where(lcs >= 0.70, 1.0, np.where(lcs >= 0.40, 0.7, 0.5))
        return corrected, w, lcs


def stage_stats(labels, corrected, lcs):
    labels = np.asarray(labels); ch = corrected != labels
    n1 = int((labels == 1).sum()); n0 = int((labels == 0).sum())
    f10 = int(((labels == 1) & (corrected == 0)).sum()); f01 = int(((labels == 0) & (corrected == 1)).sum())
    low = lcs < 0.40
    return {'n': int(len(labels)), 'balance': float(labels.mean()),
            'lcs_mean': float(lcs.mean()), 'lcs_sd': float(lcs.std()), 'lcs_skew': float(skew(lcs)),
            'high': int((lcs >= 0.70).sum()), 'mid': int(((lcs >= 0.40) & (lcs < 0.70)).sum()), 'low': int(low.sum()),
            'low_pos': int((low & (labels == 1)).sum()), 'low_neg': int((low & (labels == 0)).sum()),
            'changed': int(ch.sum()), 'flip_1to0': f10, 'flip_0to1': f01,
            'pct_1to0_of_changes': 100.0 * f10 / max(int(ch.sum()), 1),
            'changed_rate': 100.0 * ch.mean(),
            'pos_overturned_rate': 100.0 * f10 / max(n1, 1), 'neg_overturned_rate': 100.0 * f01 / max(n0, 1)}


def build_mixture(X_train, y_train, synth_X, lab_raw, lab_slte, w_slte, ratio, mode, seed=42):
    N = len(X_train); n_syn = int(N * ratio); n_real = N - n_syn
    rs = np.random.RandomState(seed)
    ridx = rs.choice(len(X_train), n_real, replace=False) if n_real > 0 else np.array([], dtype=int)
    sidx = rs.choice(len(synth_X), n_syn, replace=False) if n_syn > 0 else np.array([], dtype=int)
    pX, py, pw = [], [], []
    if n_real > 0:
        pX.append(X_train.iloc[ridx]); py.append(y_train.values[ridx]); pw.append(np.ones(n_real))
    if n_syn > 0:
        pX.append(synth_X.iloc[sidx])
        py.append(lab_slte[sidx] if mode == 'SLTE' else lab_raw[sidx])
        pw.append(w_slte[sidx] if mode == 'SLTE' else np.ones(n_syn))
    return pd.concat(pX), np.concatenate(py), np.concatenate(pw), sidx


def tstr(X_train, y_train, X_test, y_test, synth_X, lab_raw, lab_slte, w_slte, seed=42, ratios=RATIOS, keep_proba=True):
    rows, probas = [], {}
    for ratio in ratios:
        for mode in ['No-SLTE', 'SLTE']:
            Xm, ym, wm, _ = build_mixture(X_train, y_train, synth_X, lab_raw, lab_slte, w_slte, ratio, mode, seed)
            for name, clf in make_classifiers(seed).items():
                clf.fit(Xm, ym, sample_weight=wm)
                p = clf.predict_proba(X_test)[:, 1]; pred = clf.predict(X_test)
                r = {'Seed': seed, 'Ratio': f"{int(ratio*100)}%", 'Mode': mode, 'Classifier': name}
                r.update(all_metrics(y_test.values, p, pred)); rows.append(r)
                if keep_proba:
                    probas[(r['Ratio'], mode, name)] = p
    return pd.DataFrame(rows), probas


# ----------------------------------------------------------------- main
def main(synth_path, tag, mode, dataset='medical'):
    """Every section writes its own output and is skipped when that output already exists,
    so an interrupted run can simply be started again."""
    CLF_KW['scaled_lr'] = (dataset == 'adult')
    t0 = time.time(); jpath = f'out_{tag}.json'
    out = json.load(open(jpath)) if os.path.exists(jpath) else {'tag': tag, 'synth_path': synth_path, 'dataset': dataset}
    save = lambda: json.dump(out, open(jpath, 'w'), indent=1, default=float)
    log = lambda s: print(f"[{tag} {time.time()-t0:5.0f}s] {s}", flush=True)
    if dataset == 'medical':
        X_train, X_test, y_train, y_test = load_real()
        synth_X, synth_y = load_synth(synth_path, X_train.columns)
    else:
        X_train, X_test, y_train, y_test, fm = load_real_adult()
        synth_X, synth_y = load_synth_adult(synth_path, X_train.columns, fm)
    lab_raw = synth_y.values.astype(int); yv = y_test.values

    ref = SLTEReference(X_train, y_train, synth_X)
    corr, w, lcs = ref.apply(lab_raw)
    if 'stage' not in out:
        out['stage'] = stage_stats(lab_raw, corr, lcs); save(); log(f"stage {out['stage']}")

    if 'perm_null' not in out:   # 1,000 permutations of the synthetic labels (class balance preserved)
        rng = np.random.RandomState(42); perm = []
        for _ in range(1000):
            lp = rng.permutation(lab_raw); cp, _, lp_lcs = ref.apply(lp); perm.append(stage_stats(lp, cp, lp_lcs))
        pdist = np.array([s['pct_1to0_of_changes'] for s in perm]); cdist = np.array([s['changed_rate'] for s in perm])
        out['perm_null'] = {'first': perm[0], 'mean_pct_1to0': float(pdist.mean()), 'sd_pct_1to0': float(pdist.std()),
                            'q025': float(np.percentile(pdist, 2.5)), 'q975': float(np.percentile(pdist, 97.5)),
                            'p_obs_ge': float((pdist >= out['stage']['pct_1to0_of_changes']).mean()),
                            'p_obs_le': float((pdist <= out['stage']['pct_1to0_of_changes']).mean()),
                            'mean_pos_overturned': float(np.mean([s['pos_overturned_rate'] for s in perm])),
                            'mean_changed_rate': float(cdist.mean()), 'changed_q025': float(np.percentile(cdist, 2.5)),
                            'changed_q975': float(np.percentile(cdist, 97.5))}
        for nm, pr in [('prior_real', y_train.mean()), ('prior_synth', lab_raw.mean())]:
            lr_ = np.random.RandomState(42).choice([0, 1], size=len(lab_raw), p=[1 - pr, pr])
            cr, _, lr_lcs = ref.apply(lr_); out[nm] = stage_stats(lr_, cr, lr_lcs)
        save(); log(f"perm null {out['perm_null']['mean_pct_1to0']:.2f} [{out['perm_null']['q025']:.2f},{out['perm_null']['q975']:.2f}]")

    if 'dcr' not in out:   # DCR on 5,000-record subsamples (same design as the revision notebook)
        sc = StandardScaler().fit(X_train); Xtr_s = sc.transform(X_train); Xsy_s = sc.transform(synth_X)
        rs = np.random.RandomState(42); s_idx = rs.choice(len(synth_X), 5000, replace=False); r_idx = rs.choice(len(X_train), 5000, replace=False)
        nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(Xtr_s)
        d_sr = nn.kneighbors(Xsy_s[s_idx], n_neighbors=1)[0].ravel(); d_rr = nn.kneighbors(Xtr_s[r_idx], n_neighbors=2)[0][:, 1]
        out['dcr'] = {k: {'min': float(v.min()), 'p5': float(np.percentile(v, 5)), 'median': float(np.median(v)), 'mean': float(v.mean()),
                          'exact_zero': int((v == 0).sum())} for k, v in [('synth_to_real', d_sr), ('real_to_real', d_rr)]}
        save(); log('dcr done')

    if 'heldout' not in out:   # anchor and kNN refitted on 80% of the real training data, applied to the other 20%
        Xa, Xv, ya, yv_ = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train, random_state=42)
        hv = SLTEReference(Xa, ya, Xv); ch, _, hl = hv.apply(yv_.values.astype(int))
        out['heldout'] = {'n_reference': int(len(Xa)), **stage_stats(yv_.values.astype(int), ch, hl)}
        hs = SLTEReference(Xa, ya, synth_X); cs, _, ls = hs.apply(lab_raw)   # same reduced anchor on the synthetic labels
        out['synth_with_heldout_anchor'] = stage_stats(lab_raw, cs, ls)
        save(); log(f"held-out {out['heldout']['changed_rate']:.2f}% vs synthetic {out['synth_with_heldout_anchor']['changed_rate']:.2f}%")
    if mode == 'stats':
        log('DONE'); return

    # main TSTR (seed 42) with every metric and saved probabilities
    ppath = f'probas_{tag}.pkl'
    if os.path.exists(f'tstr_{tag}.csv') and os.path.exists(ppath):
        res = pd.read_csv(f'tstr_{tag}.csv'); probas = pickle.load(open(ppath, 'rb'))['probas']
    elif os.path.exists(f'tstr_{tag}.csv') and os.path.exists(f'delong_slte_{tag}.csv') and os.path.exists(f'delong_sub_{tag}.csv'):
        res = pd.read_csv(f'tstr_{tag}.csv'); probas = None
    else:
        res, probas = tstr(X_train, y_train, X_test, y_test, synth_X, lab_raw, corr, w, seed=42)
        res.to_csv(f'tstr_{tag}.csv', index=False)
        with open(ppath, 'wb') as f:
            pickle.dump({'y_test': yv, 'probas': probas}, f)
        log('tstr done')

    def proba(ratio, m, name):   # stored probabilities, or an exact refit of that single model
        if probas is not None:
            return probas[(ratio, m, name)]
        Xm, ym, wm, _ = build_mixture(X_train, y_train, synth_X, lab_raw, corr, w, int(ratio[:-1]) / 100, m, 42)
        clf = make_classifiers(42)[name]; clf.fit(Xm, ym, sample_weight=wm); return clf.predict_proba(X_test)[:, 1]

    if not os.path.exists(f'delong_slte_{tag}.csv'):   # SLTE vs No-SLTE, every classifier, 20-100% (Holm over 25)
        dl = []
        for name in CLF_ORDER:
            for r in ['20%', '40%', '60%', '80%', '100%']:
                a = delong(yv, proba(r, 'SLTE', name), proba(r, 'No-SLTE', name))
                dl.append({'Classifier': name, 'Ratio': r, 'AUC_SLTE': a[0], 'AUC_NoSLTE': a[1], 'diff': a[2],
                           'ci_low': a[3], 'ci_high': a[4], 'z': a[5], 'p': a[6]})
        dl = pd.DataFrame(dl); dl['p_holm'] = holm(dl['p'].values); dl.to_csv(f'delong_slte_{tag}.csv', index=False)
    if not os.path.exists(f'delong_sub_{tag}.csv'):   # substitution penalty: each ratio (No-SLTE) vs 0% real (Holm over 25)
        ds = []
        for name in CLF_ORDER:
            p0 = proba('0%', 'No-SLTE', name)
            for r in ['20%', '40%', '60%', '80%', '100%']:
                a = delong(yv, proba(r, 'No-SLTE', name), p0)
                ds.append({'Classifier': name, 'Ratio': r, 'AUC_mixed': a[0], 'AUC_real': a[1], 'diff': a[2],
                           'ci_low': a[3], 'ci_high': a[4], 'z': a[5], 'p': a[6]})
        ds = pd.DataFrame(ds); ds['p_holm'] = holm(ds['p'].values); ds.to_csv(f'delong_sub_{tag}.csv', index=False)
        log('delong done')

    if mode == 'full':
        if 'gb_operating_point' not in out:   # 0% Gradient Boosting operating point (same fit as the baseline table)
            pg = proba('0%', 'No-SLTE', 'Gradient Boosting')
            pr_, rc_, th_ = precision_recall_curve(yv, pg); f1s = 2 * pr_[:-1] * rc_[:-1] / np.clip(pr_[:-1] + rc_[:-1], 1e-12, None)
            bi = int(np.nanargmax(f1s)); op = {'default': all_metrics(yv, pg, (pg >= 0.5).astype(int)), 'ap': float(average_precision_score(yv, pg)),
                                            'base_rate': float(yv.mean()), 'auc_check': float(roc_auc_score(yv, pg))}
            tf = float(th_[bi]); op['maxf1'] = {'t': tf, **all_metrics(yv, pg, (pg >= tf).astype(int))}
            for tgt in [0.5, 0.8]:
                ok = np.where(rc_[:-1] >= tgt)[0]; i = ok[np.argmax(pr_[:-1][ok])]; t = float(th_[i])
                op[f'recall{int(tgt*100)}'] = {'t': t, **all_metrics(yv, pg, (pg >= t).astype(int))}
            out['gb_operating_point'] = op; save()
            np.save(f'gb_pr_curve_{tag}.npy', np.vstack([rc_, pr_])); log('operating point done')

        if not (os.path.exists(f'realonly_{tag}.csv') and os.path.exists(f'delong_realonly_{tag}.csv')):
            ro_rows = []; ro_probas = {}
            for ratio in [0.0, 0.2, 0.4, 0.6, 0.8]:
                n_real = len(X_train) - int(len(X_train) * ratio); ridx = np.random.RandomState(42).choice(len(X_train), n_real, replace=False)
                for name, clf in make_classifiers(42).items():
                    clf.fit(X_train.iloc[ridx], y_train.iloc[ridx]); p = clf.predict_proba(X_test)[:, 1]; pred = clf.predict(X_test)
                    r = {'Ratio': f"{int(ratio*100)}%", 'Real_N': n_real, 'Classifier': name}; r.update(all_metrics(yv, p, pred)); ro_rows.append(r)
                    ro_probas[(r['Ratio'], name)] = p
            pd.DataFrame(ro_rows).to_csv(f'realonly_{tag}.csv', index=False)
            dr = []
            for name in CLF_ORDER:
                for r in ['20%', '40%', '60%', '80%']:
                    for m in ['No-SLTE', 'SLTE']:
                        a = delong(yv, ro_probas[(r, name)], proba(r, m, name))
                        dr.append({'Classifier': name, 'Ratio': r, 'Versus': m, 'AUC_real': a[0], 'AUC_mixed': a[1], 'diff': a[2],
                                   'ci_low': a[3], 'ci_high': a[4], 'z': a[5], 'p': a[6]})
            dr = pd.DataFrame(dr); dr['p_holm'] = holm(dr['p'].values); dr.to_csv(f'delong_realonly_{tag}.csv', index=False)
            log('real-only done')

        _, _, _, sidx = build_mixture(X_train, y_train, synth_X, lab_raw, corr, w, 1.0, 'No-SLTE', 42)
        Xs100 = synth_X.iloc[sidx]
        if not os.path.exists(f'stage_ablation_{tag}.csv'):   # 100% synthetic, row order identical to the TSTR 100% condition
            ab = []
            for arm, (lab, ww) in {'No-SLTE': (lab_raw, np.ones(len(lab_raw))), 'Stage 2 only': (corr, np.ones(len(lab_raw))),
                                   'Stage 3 only': (lab_raw, w), 'Full SLTE': (corr, w)}.items():
                for name, clf in make_classifiers(42).items():
                    clf.fit(Xs100, lab[sidx], sample_weight=ww[sidx]); p = clf.predict_proba(X_test)[:, 1]
                    r = {'Arm': arm, 'Classifier': name}; r.update(all_metrics(yv, p, clf.predict(X_test))); ab.append(r)
            pd.DataFrame(ab).to_csv(f'stage_ablation_{tag}.csv', index=False); log('stage ablation done')

        need_csv = not os.path.exists(f'reference_ablation_{tag}.csv')
        if need_csv or 'reference_stats' not in out:   # reference sets of 500, 1,000, 5,000 and the full training set
            rf = []; ref_stats = {}
            for n_ref in [500, 1000, 5000, len(X_train)]:
                idx = np.arange(len(X_train)) if n_ref == len(X_train) else np.random.RandomState(42).choice(len(X_train), n_ref, replace=False)
                rr = SLTEReference(X_train.iloc[idx], y_train.iloc[idx], synth_X); cr, wr, lr_ = rr.apply(lab_raw)
                ref_stats[str(n_ref)] = stage_stats(lab_raw, cr, lr_)
                if need_csv:
                    for name, clf in make_classifiers(42).items():
                        clf.fit(Xs100, cr[sidx], sample_weight=wr[sidx]); p = clf.predict_proba(X_test)[:, 1]
                        r = {'N_Ref': n_ref, 'Classifier': name}; r.update(all_metrics(yv, p, clf.predict(X_test))); rf.append(r)
            if need_csv:
                pd.DataFrame(rf).to_csv(f'reference_ablation_{tag}.csv', index=False)
            out['reference_stats'] = ref_stats; save(); log('reference ablation done')

        for s in [1, 2, 3, 4]:   # classifier seeds (seed controls mixture sampling and estimator seeds)
            fp = f'seeds_{tag}_s{s}.csv'
            if not os.path.exists(fp):
                r_s, _ = tstr(X_train, y_train, X_test, y_test, synth_X, lab_raw, corr, w, seed=s, keep_proba=False)
                r_s.to_csv(fp, index=False); log(f'seed {s} done')
        pd.concat([res] + [pd.read_csv(f'seeds_{tag}_s{s}.csv') for s in [1, 2, 3, 4]]).to_csv(f'seeds_{tag}.csv', index=False)
    save(); log('DONE')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else 'medical')
