"""PAT-335 exploratory: physical-Jacobian commutators of the held-out base40 systems, vs the learned transport's.

ODE identification: data/base40/test_zeroshot.json lists (sample_idx, params) per system, but the arrow file -> parameter
mapping and the sampling time step are not stored. For each trajectory file we therefore fit, over all listed parameter
sets, dx/dn ~= c * f_params(x) (central differences in samples, least-squares scale c = time per sample) and keep the
best set if its relative residual is < 5 % (else: ODE unavailable for that file -> reported as a limitation).
Skew40 systems (driver-response couplings) are not identified and are excluded.
Statistics on the same context window as pat336/props.py (x[:, T0 - ctx : T0], T0 = 3072), at patch time scale 16 samples:
  J_p    = Jacobian of the vector field at the patch start x_{16 p}; normalised C = ||[J_p, J_{p-1}]|| / (||J_p|| ||J_{p-1}||)
  Phi_p  = patch propagator prod_s expm(c J(x_s)) over the 16 samples of patch p; normalised with ||Phi - I||
  and the same with the patch order permuted (control). Only dimensionless normalised statistics are compared with the
  latent ones (rank correlation across systems); latent and physical matrix entries are never compared directly.
usage: python -m pat336.jac OUT [COMM_CSV ...]   (COMM_CSV from pat336.props; Spearman per checkpoint / ctx / layer-mean)"""
import sys, json, csv, numpy as np
from pathlib import Path
from scipy.linalg import expm
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]; T0, PATCH = 3072, 16
rng = np.random.default_rng(0)


def rhs(s, X):
    try:
        out = np.asarray(s._rhs(*X.T, 0.0, *s.param_list), dtype=float)
        if out.shape == (X.shape[1], X.shape[0]): return out.T
    except Exception:
        pass
    return np.stack([np.asarray(s._rhs(*x, 0.0, *s.param_list), dtype=float) for x in X])


def jac(s, x):
    d = len(x); J = np.zeros((d, d))
    for a in range(d):
        h = 1e-6 * max(1.0, abs(x[a])); e = np.zeros(d); e[a] = h
        J[:, a] = (rhs(s, (x + e)[None])[0] - rhs(s, (x - e)[None])[0]) / (2 * h)
    return J


def norm_comm(X, c):
    I = np.eye(X.shape[-1]) * c; C = np.linalg.norm(X[1:] @ X[:-1] - X[:-1] @ X[1:], axis=(1, 2))
    den = np.linalg.norm(X[1:] - I, axis=(1, 2)) * np.linalg.norm(X[:-1] - I, axis=(1, 2))
    return np.where(den > 1e-12, C / np.maximum(den, 1e-30), np.nan)


def main(out, comms):
    from gluonts.dataset.common import FileDataset
    import dysts.flows as F
    meta = json.load(open(ROOT / 'data/base40/test_zeroshot.json')); rows = []
    for d in sorted((ROOT / 'data/final_base40/test_zeroshot').iterdir()):
        f = sorted(d.glob('*.arrow'), key=lambda p: int(p.name.split('_')[0]))[0]
        x = np.asarray(list(FileDataset(path=f, freq='h', one_dim_target=False))[0]['target'], dtype=float).T   # [T, D]
        s = getattr(F, d.name)(); best = (np.inf, None, None)
        dx = (x[2:] - x[:-2]) / 2; xm = x[1:-1]
        for e in meta.get(d.name, []):
            s.params = dict(e['params']); s.param_list = [s.params[kk] for kk in sorted(s.params)]
            try: fx = rhs(s, xm)
            except Exception: continue
            if not np.isfinite(fx).all(): continue
            c = float((dx * fx).sum() / max((fx * fx).sum(), 1e-30)); res = float(np.linalg.norm(dx - c * fx) / np.linalg.norm(dx))
            if res < best[0]: best = (res, e, c)
        res, e, c = best
        row = dict(system=d.name, file=f.name, fit_resid=res, sample_idx=None if e is None else e['sample_idx'], dt=c)
        if e is None or res > .05:
            row['status'] = 'ODE not identified (resid %.3f)' % res; rows.append(row); print(row, flush=True); continue
        s.params = dict(e['params']); s.param_list = [s.params[kk] for kk in sorted(s.params)]
        w = x[T0 - 512:T0]; NP = 512 // PATCH
        Js = np.stack([jac(s, w[p * PATCH]) for p in range(NP)])
        Phi = []
        for p in range(NP):
            P = np.eye(x.shape[1])
            for t in range(p * PATCH, (p + 1) * PATCH): P = expm(c * jac(s, w[t])) @ P
            Phi.append(P)
        Phi = np.stack(Phi); perm = rng.permutation(NP)
        row.update(status='ok', J_adj=np.nanmean(norm_comm(Js, 0)), J_perm=np.nanmean(norm_comm(Js[perm], 0)),
                   Phi_adj=np.nanmean(norm_comm(Phi, 1)), Phi_perm=np.nanmean(norm_comm(Phi[perm], 1)))
        rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    keys = ['system', 'file', 'sample_idx', 'fit_resid', 'dt', 'status', 'J_adj', 'J_perm', 'Phi_adj', 'Phi_perm']
    with open(out + '_jac.csv', 'w', newline='') as fo:
        wr = csv.DictWriter(fo, keys); wr.writeheader(); [wr.writerow(r) for r in rows]
    ok = {r['system']: r for r in rows if r['status'] == 'ok'}; print(f'{len(ok)}/{len(rows)} base systems with identified ODE', flush=True)
    for cf in comms:
        import pandas as pd
        df = pd.read_csv(cf); df = df[df.system.isin(ok)]
        for (ck, ctx), g in df.groupby(['ckpt', 'ctx']):
            lat = g.groupby('system').comm_n.mean(); latp = g.groupby('system').comm_pn.mean(); sy = [s_ for s_ in lat.index if s_ in ok]
            for phys in ('J_adj', 'Phi_adj'):
                rho, pv = spearmanr([lat[s_] for s_ in sy], [ok[s_][phys] for s_ in sy])
                rho2, pv2 = spearmanr([lat[s_] - latp[s_] for s_ in sy], [ok[s_][phys] - ok[s_][phys.replace('adj', 'perm')] for s_ in sy])
                print(f'{ck} ctx{ctx}: Spearman(latent comm_n layer-mean, {phys}) = {rho:+.2f} (p {pv:.2f}); adj-perm: {rho2:+.2f} (p {pv2:.2f}); n = {len(sy)}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
