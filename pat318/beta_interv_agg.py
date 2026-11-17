"""PAT-318: aggregate the beta intervention (job p318bi). Per ctx: medians of sMAPE / MSE at L128 (also L512), paired
bootstrap 95% CI of median(arm) - median(Original), fraction of trajectories where Original < arm, beta stats, and the R0
check that c=1 equals the original evaluation trajectory by trajectory."""
import sys, json; sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from pat318.agg import load
E = 'pat318/eval/'; P = 'p318ut_path_s99_st100000_ls99'
rng = np.random.default_rng(0)
ARMS = [('Original', 'bic1'), ('β×0', 'bic0'), ('β×0.5', 'bic0.5'), ('β×1.5', 'bic1.5'), ('shuf0', 'bis0'), ('shuf1', 'bis1'), ('shuf2', 'bis2')]
for ctx in (512, 1024, 2048):
    d = {n: f'{E}{P}_{t}_ctx{ctx}' for n, t in ARMS}
    d = {n: p for n, p in d.items() if Path(p, 'metrics_pred512.csv').exists()}
    if 'Original' not in d: continue
    for m in ('smape', 'mse'):
        for L in (128, 512):
            orig = load(f'{E}{P}_ctx{ctx}', L, m); o = load(d['Original'], L, m)
            r0 = np.abs(o - orig).max()
            I = [rng.integers(0, len(o), len(o)) for _ in range(1000)]
            row = f'ctx{ctx} {m} L{L} R0 maxdiff {r0:.2e} | [nope {np.median(load(f"{E}p318ut_nope_s99_st100000_ls99_ctx{ctx}", L, m)):.3f} rope {np.median(load(f"{E}p318ut_rope_s99_st100000_ls99_ctx{ctx}", L, m)):.3f}] Original {np.median(o):.3f}'
            sh = []
            for n in d:
                if n == 'Original': continue
                x = load(d[n], L, m); bs = [np.median(x[i]) - np.median(o[i]) for i in I]
                row += f' | {n} {np.median(x):.3f} Δ[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}] orig<arm {np.mean(o < x):.2f}'
                if n.startswith('shuf'): sh.append(np.median(x))
            if sh: row += f' | shuffled {np.mean(sh):.3f}±{np.std(sh, ddof=1) if len(sh) > 1 else 0:.3f} (n={len(sh)})'
            print(row, flush=True)
    for n, p in d.items():
        s = json.load(open(Path(p, 'beta_stats.json')))
        print(f'ctx{ctx} beta {n}: mean {s["all"]["mean"]:.3f} (orig {s["all"]["mean_orig"]:.3f}) std {s["all"]["std"]:.3f} clip {s["all"]["clip_frac"]:.4f} | per-layer mean '
              + ' '.join(f'{l["mean"]:.2f}' for l in s['layers']) + ' | per-layer clip ' + ' '.join(f'{l["clip_frac"]:.3f}' for l in s['layers']))
