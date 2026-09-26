"""
Synthetic data generation for the revision (round 2).

Usage:  python gen_run.py <dataset> <kind> <seed>
    dataset : medical | adult
    kind    : ctgan_default  (CTGAN, log_frequency=True, the library default, as in the primary run)
              ctgan_nolog    (CTGAN, log_frequency=False)
              tvae           (TVAE, no conditional vector)
              copula         (Gaussian copula with empirical marginals)

CTGAN uses the ORIGINAL configuration of the primary run (Initial_Experiments.ipynb):
epochs 300, batch 500, generator/discriminator (256, 256), learning rates 2e-4, 1 discriminator
step, pac 10; library defaults embedding_dim 128, generator_decay 1e-6, discriminator_decay 1e-6.
Training is checkpointed every 10 epochs (weights, optimiser states and RNG states), so an
interrupted run resumes to exactly the same result.
"""
import os, sys, time, pickle
import numpy as np, pandas as pd, torch
torch.set_num_threads(int(os.environ.get('GEN_THREADS', '1')))
import ctgan.synthesizers.ctgan as ct_mod
import ctgan.synthesizers.tvae as tv_mod
from ctgan import CTGAN, TVAE
from scipy.stats import norm, rankdata

EVERY = 10


class CheckpointedEpochs:
    """Drop-in for tqdm(range(epochs)) inside CTGAN.fit / TVAE.fit that saves and restores state."""
    path = None

    def __init__(self, iterable, disable=True, **kw):
        self.epochs = list(iterable)

    def set_description(self, *a, **k):
        pass

    def __iter__(self):
        loc = sys._getframe(1).f_locals          # the fit() frame
        model = loc['self']
        mods = {k: v for k, v in loc.items() if isinstance(v, torch.nn.Module)}
        mods.update({'self.' + k: v for k, v in vars(model).items() if isinstance(v, torch.nn.Module)})
        opts = {k: v for k, v in loc.items() if isinstance(v, torch.optim.Optimizer)}
        start = 0
        if self.path and os.path.exists(self.path):
            ck = torch.load(self.path, weights_only=False)
            for k, m in mods.items(): m.load_state_dict(ck['mods'][k])
            for k, o in opts.items(): o.load_state_dict(ck['opts'][k])
            np.random.set_state(ck['np']); torch.set_rng_state(ck['torch'])
            start = ck['epoch'] + 1
            print(f'resumed after epoch {ck["epoch"]}', flush=True)
        for i in self.epochs[start:]:
            yield i
            if self.path and ((i + 1) % EVERY == 0 or i == self.epochs[-1]):
                tmp = self.path + '.tmp'
                torch.save({'epoch': i, 'mods': {k: m.state_dict() for k, m in mods.items()},
                            'opts': {k: o.state_dict() for k, o in opts.items()},
                            'np': np.random.get_state(), 'torch': torch.get_rng_state()}, tmp)
                os.replace(tmp, self.path)
                print(f'epoch {i + 1} saved {time.strftime("%H:%M:%S")}', flush=True)
                if os.environ.get('KILL_AFTER') == str(i + 1):   # test hook: simulate an interruption
                    os._exit(3)


ct_mod.tqdm = CheckpointedEpochs
tv_mod.tqdm = CheckpointedEpochs


def gaussian_copula(train_df, n, seed, nominal=()):
    """Gaussian copula with empirical marginals: normal scores, correlation matrix, inverse ECDF."""
    rng = np.random.RandomState(seed); df = train_df.copy(); maps = {}
    for c in nominal:   # nominal strings -> integer codes ordered by descending frequency
        order = df[c].value_counts().index.tolist(); maps[c] = order
        df[c] = df[c].map({v: i for i, v in enumerate(order)})
    X = df.values.astype(float); N, d = X.shape
    Z = np.column_stack([norm.ppf(rankdata(X[:, j]) / (N + 1)) for j in range(d)])
    R = np.corrcoef(Z, rowvar=False)
    U = norm.cdf(rng.multivariate_normal(np.zeros(d), R, size=n))
    out = pd.DataFrame({c: np.quantile(X[:, j], U[:, j], method='inverted_cdf') for j, c in enumerate(df.columns)})
    for c in nominal:
        out[c] = out[c].round().astype(int).map(dict(enumerate(maps[c])))
    return out


def main(dataset, kind, seed):
    if dataset == 'medical':
        from gen_common import load_gen_train, DISCRETE, postprocess, TARGET
        tr = load_gen_train(); discrete = DISCRETE; post = postprocess; nominal = ()
    else:
        from adult_common import load_gen_train_adult, A_DISCRETE, postprocess_adult, A_TARGET as TARGET, A_NOMINAL
        tr = load_gen_train_adult().reset_index(drop=True); discrete = A_DISCRETE; post = postprocess_adult
        nominal = A_NOMINAL + ['sex']
    out = f'synth_{dataset}_{kind}_s{seed}.csv'
    if os.path.exists(out):
        print('exists', out); return
    t = time.time()
    if kind == 'copula':
        s = gaussian_copula(tr, len(tr), seed, nominal)
    else:
        CheckpointedEpochs.path = f'ckpt_{dataset}_{kind}_s{seed}.pt'
        if kind == 'tvae':
            m = TVAE(epochs=300, batch_size=500, enable_gpu=False)
        else:
            m = CTGAN(epochs=300, batch_size=500, generator_dim=(256, 256), discriminator_dim=(256, 256),
                      generator_lr=2e-4, discriminator_lr=2e-4, discriminator_steps=1, pac=10, verbose=False,
                      log_frequency=(kind == 'ctgan_default'), enable_gpu=False)
        m.set_random_state(seed)
        m.fit(tr, discrete)
        s = m.sample(len(tr))
    s = post(s)
    s.to_csv(out, index=False)
    print(f"DONE {dataset} {kind} seed={seed} balance={s[TARGET].mean():.4f} rows={len(s)} time={time.time()-t:.0f}s", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
