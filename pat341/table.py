"""Overall table: multivariate Panda arms (official eval dirs, same 9347 test trajectories), median sMAPE / MSE at L128 / 256 /
512 for ctx 512 / 1024 / 2048, plus paired bootstrap 95% CI of median(arm) - median(ref) vs REFS (300 resamples, seed 0, as
pat336/cmp.py). Missing eval dirs are skipped."""
import sys, os; sys.path.insert(0, '.')
import numpy as np
from pat318.agg import load
E = 'pat318/eval'
ARMS = [('Panda released', 'released_ls99_ctx{c}'), ('rope (retrained Panda)', 'p318_rope_s99_ls99_ctx{c}'),
        ('nope', 'p318_nope_s99_ls99_ctx{c}'), ('path', 'p318_path_s99_ls99_ctx{c}'), ('rope_path', 'p318_rope_path_s99_ls99_ctx{c}'),
        ('path 50k', 'p318_path_s99_50k_ls99_ctx{c}'), ('path_shared 50k', 'p335_path_shared_s99_50k_ls99_ctx{c}'),
        ('rope1_path 50k', 'p338_rope1_path_s99_50k_ls99_ctx{c}')]
REFS = [('Panda released', 'released_ls99_ctx{c}'), ('path', 'p318_path_s99_ls99_ctx{c}'), ('path 50k', 'p318_path_s99_50k_ls99_ctx{c}')]
rng = np.random.default_rng(0)
for c in (512, 1024, 2048):
    for m in ('smape', 'mse'):
        for L in (128, 256, 512):
            for name, d in ARMS:
                p = os.path.join(E, d.format(c=c))
                if not os.path.isdir(p): continue
                x = load(p, L, m); row = f'ctx{c} {m} L{L} | {name}: {np.median(x):.4f}'
                for rn, rd in REFS:
                    q = os.path.join(E, rd.format(c=c))
                    if rn == name or not os.path.isdir(q): continue
                    y = load(q, L, m)
                    if len(x) != len(y): continue
                    I = [rng.integers(0, len(y), len(y)) for _ in range(300)]; bs = [np.median(x[i]) - np.median(y[i]) for i in I]
                    row += f' | vs {rn} {np.median(x) - np.median(y):+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]'
                print(row, flush=True)
