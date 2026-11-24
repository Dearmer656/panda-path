"""paired comparison of two official eval dirs (same 9347 test trajectories): median per arm, share arm < ref,
bootstrap 95% CI of median(arm) - median(ref) (300 resamples, seed 0, as pat318/cmp_pos.py).
usage: python -m pat336.cmp ARM_DIR REF_DIR"""
import sys; sys.path.insert(0, '.')
import numpy as np
from pat318.agg import load
a, r = sys.argv[1], sys.argv[2]; rng = np.random.default_rng(0)
for m in ('smape', 'mse'):
    for L in (128, 256, 512):
        x, y = load(a, L, m), load(r, L, m); assert len(x) == len(y)
        I = [rng.integers(0, len(y), len(y)) for _ in range(300)]; bs = [np.median(x[i]) - np.median(y[i]) for i in I]
        print(f'{m} L{L}: arm {np.median(x):.4f} ref {np.median(y):.4f}  arm<ref {np.mean(x < y):.2f}  diff CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]', flush=True)
