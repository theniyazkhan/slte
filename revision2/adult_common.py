# Second dataset: UCI Adult (48,842 records, income >50K = 23.9%).
# Same split protocol as the primary dataset: stratified 80/20, random_state = 42.
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

ADULT_FILE = 'adult/adult-all.csv'
ADULT_COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
A_TARGET = 'income'
A_NOMINAL = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
A_DISCRETE = ['workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
              'native-country', A_TARGET]


def load_adult_raw():
    d = pd.read_csv(ADULT_FILE, header=None, names=ADULT_COLS, skipinitialspace=True, keep_default_na=False)
    d = d.drop(columns=['fnlwgt', 'education'])          # census weight; duplicate of education-num
    d[A_TARGET] = (d[A_TARGET] == '>50K').astype(int)
    return d


def adult_split():
    d = load_adult_raw(); X = d.drop(columns=[A_TARGET]); y = d[A_TARGET]
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)


def load_gen_train_adult():
    Xtr, Xte, ytr, yte = adult_split()
    return pd.concat([Xtr, ytr], axis=1)


def encode_adult(X, fm):
    """Frequency-encode nominal columns (maps estimated on the real training split); sex -> 0/1."""
    X = X.copy()
    for c in A_NOMINAL:
        X[c] = X[c].map(fm[c]).astype(float).fillna(0.0)
    X['sex'] = (X['sex'] == 'Male').astype(int)
    return X


def load_real_adult():
    Xtr, Xte, ytr, yte = adult_split()
    fm = {c: Xtr[c].value_counts(normalize=True) for c in A_NOMINAL}
    return encode_adult(Xtr, fm), encode_adult(Xte, fm), ytr, yte, fm


def load_synth_adult(path, cols, fm):
    s = pd.read_csv(path, keep_default_na=False)
    X = encode_adult(s.drop(columns=[A_TARGET]), fm)[cols]
    return X, s[A_TARGET].astype(int)


def postprocess_adult(s):
    s = s.copy()
    s['age'] = s['age'].clip(17, 90).round().astype(int)
    s['capital-gain'] = s['capital-gain'].clip(0).round().astype(int)
    s['capital-loss'] = s['capital-loss'].clip(0).round().astype(int)
    s['hours-per-week'] = s['hours-per-week'].clip(1, 99).round().astype(int)
    s[A_TARGET] = s[A_TARGET].astype(int)
    return s
