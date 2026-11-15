"""PAT-318 mechanism check (criteria fixed before running): does path's gain over rope grow with the irregularity of the
trajectory's own dynamics? Per test trajectory (official windows, pat321/data/test_dump.pkl, aligned with the official
per-system metric lists): features from the full trajectory (channels z-scored)
  speed_cv  CV of ||x_{t+1}-x_t||              (non-uniform phase velocity -> index distance != dynamical distance)
  recur_cv  CV of upward mean-crossing intervals of channel 0 (irregular return times)
  lyap_per  maximum_lyapunov_estimated * period (dysts metadata of the base system), kyd = Kaplan-Yorke dimension
gain = smape_rope - smape_path (per trajectory, >0 = path better). Spearman rho, CI by resampling SYSTEMS (2000x);
partial rho controlling for rope's own smape (rank residuals). GO: rho(speed_cv) > 0 with CI > 0 and partial rho > 0 at
ctx512 L128 and ctx1024 L128."""
import ast, json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
R = Path(__file__).resolve().parents[2]
EV = R / 'pat318/eval'
def load_sys(d, L, m):
    df = pd.read_csv(Path(d) / f'metrics_pred{L}.csv')
    return df, np.concatenate([np.asarray(ast.literal_eval(x), float) for x in df[m]])
recs = []
with open(R / 'pat321/data/test_dump.pkl', 'rb') as f:
    while True:
        try: recs.append(pickle.load(f))
        except EOFError: break
df, _ = load_sys(EV / 'p318ut_rope_s99_st100000_ls99_ctx512', 128, 'smape')
names = np.concatenate([[s] * int(n) for s, n in zip(df.system, df.n_system_samples)])
dims = np.concatenate([[d] * int(n) for d, n in zip(df.system_dims, df.n_system_samples)])
assert len(names) == len(recs), (len(names), len(recs))
assert all(r[0].shape[0] == d for r, d in zip(recs, dims)), 'dump/metric order mismatch'
meta = json.load(open(R.parent / 'dysts/dysts/data/chaotic_attractors.json'))
def feats(ts):
    z = (ts - ts.mean(1, keepdims=True)) / (ts.std(1, keepdims=True) + 1e-8)
    sp = np.linalg.norm(np.diff(z, axis=1), axis=0); x = z[0]
    up = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]; iv = np.diff(up)
    return sp.std() / (sp.mean() + 1e-12), (iv.std() / iv.mean() if len(iv) >= 3 else np.nan)
F = np.array([feats(r[0]) for r in recs])
def m(s, k):
    b = s.split('_')[0]; v = meta.get(b, {}).get(k)
    return float(v) if isinstance(v, (int, float)) else np.nan
lyap = np.array([m(s, 'maximum_lyapunov_estimated') * m(s, 'period') for s in names])
kyd = np.array([m(s, 'kaplan_yorke_dimension') for s in names])
X = {'speed_cv': F[:, 0], 'recur_cv': F[:, 1], 'lyap_per': lyap, 'kyd': kyd}
print(f'n_traj {len(recs)} n_sys {len(set(names))}; nan frac ' + ' '.join(f'{k} {np.isnan(v).mean():.3f}' for k, v in X.items()))
usys = np.unique(names); sidx = {s: np.where(names == s)[0] for s in usys}; rng = np.random.default_rng(0)
boots = [np.concatenate([sidx[s] for s in rng.choice(usys, len(usys))]) for _ in range(2000)]
def rho(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); return spearmanr(a[ok], b[ok])[0]
def prho(a, b, c):
    ok = ~(np.isnan(a) | np.isnan(b) | np.isnan(c)); ra, rb, rc = (rankdata(v[ok]) for v in (a, b, c))
    res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
    return np.corrcoef(res(ra), res(rb))[0, 1]
out = {}
for ctx in (512, 1024, 2048):
    for L in (128, 512):
        _, r = load_sys(EV / f'p318ut_rope_s99_st100000_ls99_ctx{ctx}', L, 'smape')
        _, p = load_sys(EV / f'p318ut_path_s99_st100000_ls99_ctx{ctx}', L, 'smape')
        g = r - p
        for k, v in X.items():
            rh = rho(v, g); bs = [rho(v[i], g[i]) for i in boots[:500]]; pr = prho(v, g, r)
            q = np.nanpercentile(v, [20, 40, 60, 80]); bins = np.digitize(v, q)
            qm = [float(np.median(g[bins == b])) for b in range(5)]
            out[f'ctx{ctx}_L{L}_{k}'] = dict(rho=rh, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], partial=pr, quintile_median_gain=qm)
            print(f'ctx{ctx} L{L} {k:9s} rho {rh:+.3f} CI[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] partial|rope {pr:+.3f}  gain by quintile ' + ' '.join(f'{x:+.2f}' for x in qm), flush=True)
print('corr between features: ' + ' '.join(f'{a}-{b} {rho(X[a], X[b]):+.2f}' for a in X for b in X if a < b))
json.dump(out, open(Path(__file__).with_suffix('.json'), 'w'), indent=1)
