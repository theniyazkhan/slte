"""Statistical tests across the complete experiment (run after slte_revision2.py).
 (1) Real-only versus mixed training, DeLong for every classifier and ratio, for EVERY generation run with
     saved probabilities (Holm correction over the 40 tests of each run) -> extra_realonly_<tag>.csv
 (2) Gradient Boosting (and every other classifier) trained on the 20% real subset versus all real records.
Real-only probabilities are computed once per dataset and cached (same subsets as slte_revision2.py)."""
import os, sys, json, pickle
import numpy as np, pandas as pd
from common import load_real, make_classifiers
from adult_common import load_real_adult
from slte_revision2 import delong, holm, all_metrics, CLF_ORDER


def realonly_probas(dataset):
    fp = f'realonly_probas_{dataset}.pkl'
    if os.path.exists(fp):
        return pickle.load(open(fp, 'rb'))
    if dataset == 'medical':
        X_train, X_test, y_train, y_test = load_real()
    else:
        X_train, X_test, y_train, y_test, _ = load_real_adult()
    pr, rows = {}, []
    for ratio in [0.0, 0.2, 0.4, 0.6, 0.8]:
        n_real = len(X_train) - int(len(X_train) * ratio)
        ridx = np.random.RandomState(42).choice(len(X_train), n_real, replace=False)
        for name, clf in make_classifiers(42, scaled_lr=(dataset == 'adult')).items():
            clf.fit(X_train.iloc[ridx], y_train.iloc[ridx]); p = clf.predict_proba(X_test)[:, 1]
            pr[(f'{int(ratio*100)}%', name)] = p
            r = {'Ratio': f'{int(ratio*100)}%', 'Real_N': n_real, 'Classifier': name}
            r.update(all_metrics(y_test.values, p, clf.predict(X_test))); rows.append(r)
    out = {'y_test': y_test.values, 'probas': pr, 'table': pd.DataFrame(rows)}
    pickle.dump(out, open(fp, 'wb'))
    out['table'].to_csv(f'realonly_{dataset}.csv', index=False)
    return out


def main(tags):
    res = json.load(open('extras.json')) if os.path.exists('extras.json') else {}
    for ds in ['medical', 'adult']:
        ro = realonly_probas(ds); y = ro['y_test']; d = {}
        for name in CLF_ORDER:
            a = delong(y, ro['probas'][('80%', name)], ro['probas'][('0%', name)])
            d[name] = {'auc_20pct_real': a[0], 'auc_all_real': a[1], 'diff': a[2], 'ci_low': a[3], 'ci_high': a[4], 'z': a[5], 'p': a[6]}
        res[f'subset_vs_all_{ds}'] = d
    for tag in tags:
        ds = 'adult' if tag.startswith('adult') else 'medical'
        pp = f'probas_{tag}.pkl'
        if not os.path.exists(pp):
            print('skip (no probabilities yet)', tag); continue
        pm = pickle.load(open(pp, 'rb')); ro = realonly_probas(ds); y = ro['y_test']
        assert np.array_equal(y, pm['y_test'])
        rows = []
        for name in CLF_ORDER:
            for r in ['20%', '40%', '60%', '80%']:
                for m in ['No-SLTE', 'SLTE']:
                    a = delong(y, ro['probas'][(r, name)], pm['probas'][(r, m, name)])
                    rows.append({'Classifier': name, 'Ratio': r, 'Versus': m, 'AUC_real': a[0], 'AUC_mixed': a[1], 'diff': a[2],
                                 'ci_low': a[3], 'ci_high': a[4], 'z': a[5], 'p': a[6]})
        d = pd.DataFrame(rows); d['p_holm'] = holm(d['p'].values); d.to_csv(f'extra_realonly_{tag}.csv', index=False)
        print(tag, 'real-only significantly better in', int(((d.p_holm < 0.05) & (d['diff'] > 0)).sum()), 'of', len(d))
    json.dump(res, open('extras.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
