"""PAT-323 step 2 (CPU): error covariance structure (A), direction stability (B), historical-error predictability (C).
Input: pat323/resid.pkl from dump_resid.py. Per system (coordinates differ across systems), per lead h separately.
Splits by trajectory: A = traj 0,3,6; B = 1,4,7; V = 2,5,8 (held out). Fit set = A u B.
Pre-fixed criteria (PAT-323): each lead judged alone, holds if >= 3 of the 4 leads {1,16,64,128}; system-bootstrap 95% CI.
  concentrated: median top-1 explained variance >= 0.5 and median erank/D <= 0.6
  stable (k=1): median S_1(A,V) and S_1(B,V) >= 0.7 and CI of median(S_1 - 1/D) > 0
  predictive: pooled R2_OOS(correct) > 0 and CI of R2(correct) - mean_seed R2(shuffled) > 0
usage: python pat323/analyze.py pat323/resid.pkl OUTDIR [MIN_SCALE]
Deviation (after seeing the as-run output): 22 systems have a channel that is nearly constant over x[0:512] (std < 1e-2 vs
median 0.99), so the residual / scale explodes (up to 1e8) and 5 systems carry 97% of the pooled SST. Main analysis uses
MIN_SCALE = 1e-2, sensitivity 1e-3; the as-run result (MIN_SCALE = 0) is kept and reported."""
import sys, json, pickle, numpy as np
from pathlib import Path
R = pickle.load(open(sys.argv[1], 'rb')); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
HS = R['h']; LEADS = [1, 16, 64, 128]; K = 4; STRIDE = R['stride']
A, B, V = [0, 3, 6], [1, 4, 7], [2, 5, 8]
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
rng0 = np.random.default_rng(0)
MIN_SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0   # drop systems whose first-context std of any channel/trajectory < MIN_SCALE
systems = sorted(n for n, v in R['systems'].items() if 'scale' not in v or v['scale'].min() >= MIN_SCALE)
print('MIN_SCALE', MIN_SCALE, 'kept', len(systems), 'of', len(R['systems']))


def cov(X):
    mu = X.mean(0); return mu, np.cov(X, rowvar=False, ddof=1).reshape(X.shape[1], X.shape[1])


def eig_desc(C):
    w, U = np.linalg.eigh(C); return np.clip(w[::-1], 0, None), U[:, ::-1]


def S(Ua, Ub, k):
    return float(np.sum((Ua[:, :k].T @ Ub[:, :k]) ** 2) / k)


def hist_feature(E, g):
    """E [n_traj, O, D] -> (r [n, O', D], e [n, O', D]) with r_o = mean of E[o-g-K+1 .. o-g] (all eligible: o' <= o - g)."""
    O = E.shape[1]; idx = np.arange(g + K - 1, O)
    r = np.stack([E[:, o - g - K + 1:o - g + 1].mean(1) for o in idx], 1)
    return r, E[:, idx]


def ridge_fit(Xs, Ys, alpha):
    X = np.concatenate(Xs); Y = np.concatenate(Ys); mx, my = X.mean(0), Y.mean(0); Xc = X - mx
    lam = alpha * len(X) * (np.trace(Xc.T @ Xc) / len(X) / X.shape[1] + 1e-12)
    Bm = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (Y - my)); return Bm, mx, my


def ridge_pred(m, X):
    Bm, mx, my = m; return (X - mx) @ Bm + my


def ridge_cv(rs, es):
    """rs, es: lists over fit trajectories of [O', D]; leave-one-trajectory-out alpha choice, then refit on all."""
    best = None
    for a in ALPHAS:
        sse = 0.
        for i in range(len(rs)):
            m = ridge_fit([r for j, r in enumerate(rs) if j != i], [e for j, e in enumerate(es) if j != i], a)
            sse += np.sum((es[i] - ridge_pred(m, rs[i])) ** 2)
        if best is None or sse < best[0]: best = (sse, a)
    return ridge_fit(rs, es, best[1]), best[1]


res = {h: [] for h in LEADS}
for name in systems:
    s = R['systems'][name]; Dm = s['dim']
    for h in LEADS:
        E = s['E'][:, :, HS.index(h)].astype(np.float64)                  # [9, O, D]
        row = dict(system=name, D=Dm)
        # A: covariance on fit set
        Xf = E[A + B].reshape(-1, Dm); mu, C = cov(Xf); lam, U = eig_desc(C); p = lam / lam.sum()
        row.update(eig=p.tolist(), ev={k: float(p[:k].sum()) for k in (1, 2, 5, 10)}, erank=float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
                   mu_norm=float(np.linalg.norm(mu)), mu_frac=float(mu @ mu / np.trace(C)))
        # B: stability
        Us = {nm: eig_desc(cov(E[ix].reshape(-1, Dm))[1])[1] for nm, ix in (('A', A), ('B', B), ('V', V))}
        st = {}
        for k in (1, 2, 5):
            if k >= Dm: continue
            rnd = np.mean([S(np.linalg.qr(rng0.standard_normal((Dm, Dm)))[0], Us['V'], k) for _ in range(1000)])
            st[k] = dict(AB=S(Us['A'], Us['B'], k), AV=S(Us['A'], Us['V'], k), BV=S(Us['B'], Us['V'], k), rand_emp=float(rnd), rand=k / Dm)
        row['stab'] = st
        # C: historical predictability
        g = max(1, -(-h // STRIDE))
        r, e = hist_feature(E, g)                                          # [9, O', D]
        fit_r, fit_e = [r[i] for i in A + B], [e[i] for i in A + B]
        mfit = np.concatenate(fit_e).mean(0)
        eV = np.concatenate([e[i] for i in V])
        sst = np.sum((eV - mfit) ** 2)
        m, alpha = ridge_cv(fit_r, fit_e)
        sse = dict(zero=float(np.sum(eV ** 2)), mean=float(sst), correct=float(np.sum((eV - ridge_pred(m, np.concatenate([r[i] for i in V]))) ** 2)))
        for sd in (0, 1, 2):
            rg = np.random.default_rng(1000 + sd)
            rp = np.stack([ri[rg.permutation(len(ri))] for ri in r])
            ms, _ = ridge_cv([rp[i] for i in A + B], fit_e)
            sse[f'shuf{sd}'] = float(np.sum((eV - ridge_pred(ms, np.concatenate([rp[i] for i in V]))) ** 2))
        row.update(sst=float(sst), sse=sse, alpha=alpha, nV=int(eV.size))
        res[h].append(row)
json.dump(res, open(OUT / 'per_system.json', 'w'))

# ---------- summaries ----------
def boot(f, n, B=2000, seed=0):
    g = np.random.default_rng(seed); v = [f(g.integers(0, n, n)) for _ in range(B)]; return np.percentile(v, [2.5, 97.5])
lines, verdict = [], {}
dims = sorted({r['D'] for r in res[LEADS[0]]})
lines.append(f'systems {len(res[LEADS[0]])}, dims ' + str({d: sum(r["D"] == d for r in res[LEADS[0]]) for d in dims}))
for h in LEADS:
    rs = res[h]; n = len(rs); Dv = np.array([r['D'] for r in rs])
    top1 = np.array([r['ev'][1] for r in rs]); er = np.array([r['erank'] / r['D'] for r in rs])
    conc = np.median(top1) >= 0.5 and np.median(er) <= 0.6
    lines.append(f'\n=== lead h={h}')
    lines.append(f'[A] top-1 EV median {np.median(top1):.3f} (isotropic 1/D median {np.median(1 / Dv):.3f}); top-2 {np.median([r["ev"][2] for r in rs]):.3f}; '
                 f'top-5 {np.median([r["ev"][5] for r in rs]):.3f}; erank/D median {np.median(er):.3f}; erank median {np.median([r["erank"] for r in rs]):.2f}; '
                 f'|mu| median {np.median([r["mu_norm"] for r in rs]):.3f}; |mu|^2/trC median {np.median([r["mu_frac"] for r in rs]):.4f}  -> concentrated {conc}')
    for d in dims:
        sub = [r for r in rs if r['D'] == d]
        if len(sub) >= 5:
            lines.append(f'    D={d} (n={len(sub)}): eig frac median ' + ' '.join(f'{x:.3f}' for x in np.median([r["eig"] for r in sub], 0)) +
                         f' | erank {np.median([r["erank"] for r in sub]):.2f}')
    for k in (1, 2, 5):
        sub = [r for r in rs if k in r['stab'] or str(k) in r['stab']]
        if not sub: continue
        gk = lambda r: r['stab'][k] if k in r['stab'] else r['stab'][str(k)]
        AB, AV, BV = (np.array([gk(r)[x] for r in sub]) for x in ('AB', 'AV', 'BV'))
        rd, rde = np.array([gk(r)['rand'] for r in sub]), np.array([gk(r)['rand_emp'] for r in sub])
        ci = boot(lambda i: np.median((AV - rd)[i]), len(sub))
        line = f'[B] k={k} (n={len(sub)}): S(A,B) {np.median(AB):.3f} S(A,V) {np.median(AV):.3f} S(B,V) {np.median(BV):.3f} | random k/D {np.median(rd):.3f} (empirical {np.median(rde):.3f}) | median S(A,V)-k/D CI [{ci[0]:+.3f},{ci[1]:+.3f}]'
        if k == 1:
            stab = np.median(AV) >= 0.7 and np.median(BV) >= 0.7 and ci[0] > 0; line += f'  -> stable {stab}'
        lines.append(line)
    sst = np.array([r['sst'] for r in rs]); nV = np.array([r['nV'] for r in rs])
    SSE = {k: np.array([r['sse'][k] for r in rs]) for k in rs[0]['sse']}
    r2 = {k: 1 - v.sum() / sst.sum() for k, v in SSE.items()}
    shuf = lambda i: np.mean([1 - SSE[f'shuf{s}'][i].sum() / sst[i].sum() for s in range(3)])
    corr = lambda i: 1 - SSE['correct'][i].sum() / sst[i].sum()
    ci = boot(lambda i: corr(i) - shuf(i), n)
    sh = [r2[f'shuf{s}'] for s in range(3)]
    pred = r2['correct'] > 0 and ci[0] > 0
    lines.append('[C] MSE ' + ' '.join(f'{k} {v.sum() / nV.sum():.4f}' for k, v in SSE.items()))
    lines.append(f'[C] R2_OOS pooled: zero {r2["zero"]:+.4f} mean {r2["mean"]:+.4f} shuffled {np.mean(sh):+.4f}±{np.std(sh, ddof=1):.4f} correct {r2["correct"]:+.4f} | '
                 f'correct-shuffled CI [{ci[0]:+.4f},{ci[1]:+.4f}] | per-system median R2 correct {np.median(1 - SSE["correct"] / sst):+.4f} '
                 f'shuffled {np.median(1 - np.mean([SSE[f"shuf{s}"] for s in range(3)], 0) / sst):+.4f} | systems with R2(correct)>0: {np.mean(1 - SSE["correct"] / sst > 0):.2f}  -> predictive {pred}')
    verdict[h] = dict(concentrated=bool(conc), stable=bool(stab), predictive=bool(pred))
tot = {q: sum(verdict[h][q] for h in LEADS) for q in ('concentrated', 'stable', 'predictive')}
lines.append('\nverdict per lead ' + json.dumps(verdict) + '\nholds (>=3/4): ' + json.dumps({q: v >= 3 for q, v in tot.items()}))
txt = '\n'.join(lines); print(txt); open(OUT / 'summary.txt', 'w').write(txt)
