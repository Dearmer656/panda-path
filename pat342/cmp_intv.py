"""PAT-342: 10k nope1_lie checkpoint, eval-only interventions vs none (same 9347 test trajectories). median sMAPE / MSE at L128 /
512, paired bootstrap 95% CI of median(intv) - median(none) (300 resamples, seed 0, as pat341/table.py). usage: python pat342/cmp_intv.py CTX"""
import sys, os; sys.path.insert(0, '.')
import numpy as np
from pat318.agg import load
c = int(sys.argv[1]); E = 'pat318/eval'; rng = np.random.default_rng(0)
ref = f'{E}/p342_n1lie10k_none_ls99_ctx{c}'
for m in ('smape', 'mse'):
    for L in (128, 512):
        y = load(ref, L, m); row = f'ctx{c} {m} L{L} none {np.median(y):.3f}'
        for iv in ('k0', 's0', 'id', 'perm', 'none_logn'):
            p = f'{E}/p342_n1lie10k_{iv}_ls99_ctx{c}'
            if not os.path.isdir(p): continue
            x = load(p, L, m); I = [rng.integers(0, len(y), len(y)) for _ in range(300)]; bs = [np.median(x[i]) - np.median(y[i]) for i in I]
            row += f' | {iv} {np.median(x):.3f} {np.median(x) - np.median(y):+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]'
        print(row, flush=True)
