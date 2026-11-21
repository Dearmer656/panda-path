"""PAT-330 next task: does the segmented error EVOLUTION g add information beyond whole-window and final-segment distance?
Protocol = PAT-330 step 1: dysts, ctx 512 normalisation, LQ 96 = 6 x 16 segments, whole own history (s >= 96, s + H <= t),
H 128 / 256 / 512, value (raw copy) readout; fallback = backbone f. Everything selected on val, test only reported.
  d_j(s) = mean((q_j - z_{s,j})^2), j = 1 (oldest) .. 6 (newest);  Dw = mean_j d_j (whole-window Euclid)
  g(s)   = mean_{j=1..5} log((d_{j+1} + eps) / (d_j + eps)), eps = 1e-4      (NOTE: telescopes to (log d_6' - log d_1') / 5)
mode pool  (P321_OUT=pat321/data): unified pool = top-200 candidates by Dw; arms rank the pool and average the top-k copies
           (k in {1,3,5,10,20} on val median sMAPE):
             A    score Dw
             B    ridge fit (train queries) of log future error on [d_1..d_6]          (learned segment-weighted distance)
             D_B  B features + g
             Bl   ridge on [log d_1..log d_6]   Dl  Bl + g   (control: g is linear in log d_1, log d_6 -> Dl == Bl expected)
           diagnostics: residualise log future error on [log Dw, log d_6] (R1) and on [log Dw, log d_6, log d_1] (R2), fitted on
           val; per-query Spearman(residual, g) on test; pairwise ranking accuracy of g inside matched strata
           (|d log Dw| < 0.1 and |d log d_6| < 0.2, direction = sign from val); per trajectory, bootstrap over trajectories.
mode blk   (P330_BLOCKS=6 P321_OUT=pat330/blk): C = PAT-330 segmented-feature FreshCast scorer, method in {r_all, top1, smap}
           chosen on val (gamma = 0); D_C = same method, weights w' ~ w * exp(gamma * g), gamma on val (g from the scorer's own
           six log segment errors).
usage: P321_LQ=96 P321_OUT=pat321/data python pat330/evo.py pool
       P321_LQ=96 P330_BLOCKS=6 P321_OUT=pat330/blk python pat330/evo.py blk"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in ('pat327', 'pat329'): sys.path.insert(0, str(ROOT / p))
import numpy as np, torch
import fc_corr as C  # noqa: E402
from mse_eval import per_traj, LAM96  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); assert FC.LQ == 96 and FC.CTX == 512
EPS, K, KS, NB, L = 1e-4, 200, (1, 3, 5, 10, 20), 6, 96
O = ROOT / 'pat330' / 'out'; HS = (128, 256, 512)


def gstat(d):
    """literal definition: mean of the 5 consecutive log ratios."""
    ld = torch.log(d + EPS); return (ld[..., 1:] - ld[..., :-1]).mean(-1)


def errs(p, y, d):
    return {'smape': per_traj(((p - y).abs() / (p.abs() + y.abs() + 1e-10)).mean(1), d['ids'], d['inst']),
            'mse': per_traj((p - y).square().mean(1), d['ids'], d['inst'])}


# ------------------------------------------------------------------ pool mode
@torch.no_grad()
def pool_batch(se, d, idx, H, with_f=True):
    rows, t = d['rows'][idx].to(dev), d['t'][idx].to(dev); X = se.S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
    win = X[ar[:, None], t[:, None] - FC.CTX + torch.arange(FC.CTX, device=dev)]
    cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
    q = Z[ar[:, None], t[:, None] + torch.arange(-L, 0, device=dev)]
    s = torch.arange(L, T + 1, device=dev)
    dj = (Z.unfold(1, L, 1) - q[:, None]).square().reshape(B, -1, NB, L // NB).mean(-1)        # [B, T', 6]
    Dw = dj.mean(-1); ok = s[None] + H <= t[:, None]
    v, j = Dw.masked_fill(~ok, float('inf')).topk(K, 1, largest=False)
    okk = torch.isfinite(v); sk = s[j]; djk = dj.gather(1, j[..., None].expand(-1, -1, NB))
    fut = Z[ar[:, None, None], sk.clamp(max=T - H)[..., None] + torch.arange(H, device=dev)].clamp(-FC.CLIP, FC.CLIP)
    y = Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)]
    efut = (fut - y[:, None]).square().mean(-1)
    f = ((d['f'][idx, :H].to(dev) - cm[:, None]) / cs[:, None]) if with_f else None
    return dict(dj=djk, Dw=v.masked_fill(~okk, 1.0), ok=okk, fut=fut, y=y, efut=efut, f=f, g=gstat(djk))


def design(b, kind):
    dj, g = b['dj'], b['g'][..., None]
    return {'B': dj, 'D_B': torch.cat([dj, g], -1), 'Bl': torch.log(dj + EPS), 'Dl': torch.cat([torch.log(dj + EPS), g], -1)}[kind]


def fit_ridge(X, y, lam=1e-3):
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-8); Xs = (X - mu) / sd
    A = torch.cat([Xs, torch.ones(len(Xs), 1, device=dev)], 1)
    w = torch.linalg.solve(A.T @ A + lam * len(A) * torch.eye(A.shape[1], device=dev), A.T @ y)
    return mu, sd, w


def apply_ridge(m, X):
    mu, sd, w = m; return ((X - mu) / sd) @ w[:-1] + w[-1]


def batches(n, bs=64): return [torch.arange(i, min(i + bs, n)) for i in range(0, n, bs)]


def spearman(a, b, ok):
    """per row Spearman over ok entries (ranks among ok only)."""
    big = torch.finfo(a.dtype).max
    ra = a.masked_fill(~ok, big).argsort(1).argsort(1).float(); rb = b.masked_fill(~ok, big).argsort(1).argsort(1).float()
    n = ok.sum(1).float(); m = (n - 1) / 2
    ra, rb = (ra - m[:, None]) * ok, (rb - m[:, None]) * ok
    return (ra * rb).sum(1) / (ra.square().sum(1) * rb.square().sum(1)).sqrt().clamp_min(1e-12)


def tboot(x, rng, B=2000):
    x = x[np.isfinite(x)]; idx = rng.integers(0, len(x), (B, len(x))); bs = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run_pool(H):
    rng0 = np.random.default_rng(0)
    dtr, setr = FC.load_set('train', dev); sel = torch.from_numpy(rng0.choice(len(dtr['rows']), 8192, replace=False))
    Xs = {k: [] for k in ('B', 'D_B', 'Bl', 'Dl')}; Ys = []
    for idx in batches(len(sel)):
        b = pool_batch(setr, dtr, sel[idx], H, with_f=False); ok = b['ok']
        for k in Xs: Xs[k].append(design(b, k)[ok])
        Ys.append(torch.log(b['efut'] + EPS)[ok])
    Ytr = torch.cat(Ys); M = {k: fit_ridge(torch.cat(v), Ytr) for k, v in Xs.items()}; del setr, dtr
    coef = {k: M[k][2][:-1].cpu().numpy().round(3).tolist() for k in M}
    out, diag = {}, {}
    for name in ('val', 'test'):
        d, se = FC.load_set(name, dev); P = {}; Dg = {k: [] for k in ('raw', 'r1', 'r2', 'pair', 'npair')}; cache = []
        for idx in batches(len(d['rows'])):
            b = pool_batch(se, d, idx, H); ok = b['ok']; anyv = ok.any(1)
            sc = {'A': b['Dw'], **{k: apply_ridge(M[k], design(b, k)) for k in M}}
            for arm, v in sc.items():
                order = v.masked_fill(~ok, float('inf')).argsort(1)
                for k in KS:
                    oo = order[:, :k]; w = ok.gather(1, oo).float()
                    p = (b['fut'].gather(1, oo[..., None].expand(-1, -1, H)) * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1)
                    P.setdefault(f'{arm}_k{k}', []).append(torch.where(anyv[:, None], p, b['f']).cpu())
            P.setdefault('f', []).append(b['f'].cpu()); P.setdefault('y', []).append(b['y'].cpu())
            le = torch.log(b['efut'] + EPS)
            cache.append(dict(lDw=torch.log(b['Dw'] + EPS), ld6=torch.log(b['dj'][..., -1] + EPS), ld1=torch.log(b['dj'][..., 0] + EPS),
                              le=le, g=b['g'], ok=ok))
        P = {k: torch.cat(v) for k, v in P.items()}; y = P.pop('y')
        out[name] = (d, P, y)
        # diagnostics: residualisers fitted on val (stored on the first pass)
        if name == 'val':
            okc = torch.cat([c['ok'].reshape(-1) for c in cache]); cat = lambda k: torch.cat([c[k].reshape(-1) for c in cache])[okc]
            R1 = fit_ridge(torch.stack([cat('lDw'), cat('ld6')], 1), cat('le'), 1e-6)
            R2 = fit_ridge(torch.stack([cat('lDw'), cat('ld6'), cat('ld1')], 1), cat('le'), 1e-6)
        rs = {'raw': [], 'r1': [], 'r2': [], 'pair': [], 'npair': []}
        for c in cache:
            ok = c['ok']; Bq = ok.shape[0]
            r1 = c['le'] - apply_ridge(R1, torch.stack([c['lDw'], c['ld6']], -1))
            r2 = c['le'] - apply_ridge(R2, torch.stack([c['lDw'], c['ld6'], c['ld1']], -1))
            rs['raw'].append(spearman(c['g'], c['le'], ok).cpu()); rs['r1'].append(spearman(c['g'], r1, ok).cpu()); rs['r2'].append(spearman(c['g'], r2, ok).cpu())
            m = ok[:, :, None] & ok[:, None] & ((c['lDw'][:, :, None] - c['lDw'][:, None]).abs() < 0.1) & ((c['ld6'][:, :, None] - c['ld6'][:, None]).abs() < 0.2)
            m &= torch.triu(torch.ones(K, K, dtype=torch.bool, device=dev), 1)[None]
            agree = (torch.sign(c['g'][:, :, None] - c['g'][:, None]) == torch.sign(c['le'][:, :, None] - c['le'][:, None])).float()
            npr = m.sum((1, 2)).float(); rs['pair'].append(((agree * m).sum((1, 2)) / npr.clamp_min(1)).cpu()); rs['npair'].append(npr.cpu())
        rs = {k: torch.cat(v) for k, v in rs.items()}
        diag[name] = rs
    # direction from val, evaluate on test
    dv, dt = out['val'][0], out['test'][0]; rng = np.random.default_rng(0); rep = {}
    for k in ('raw', 'r1', 'r2'):
        sv = float(np.nanmean(per_traj(torch.nan_to_num(diag['val'][k], nan=0.0), dv['ids'], dv['inst']).numpy()))
        tt = per_traj(torch.nan_to_num(diag['test'][k], nan=0.0), dt['ids'], dt['inst']).numpy()
        rep[k] = (sv, tboot(tt, rng), float((np.sign(tt) == np.sign(sv)).mean()))
    sgn = np.sign(rep['r1'][0]) or 1.0
    pv = diag['test']['pair']; npv = diag['test']['npair']; use = npv >= 10
    acc = torch.where(use, pv if sgn > 0 else 1 - pv, torch.full_like(pv, float('nan')))
    # per trajectory mean of available queries
    ids = dt['ids'][use]; accq = acc[use]
    pt = per_traj(accq, ids, dt['inst']).numpy() if use.any() else np.array([np.nan])
    rep['pair'] = (tboot(pt, rng), float(use.float().mean()), float(npv[use].median()) if use.any() else 0.0)
    # val selection of k per arm
    sel_k, T, V = {}, {}, {}
    for arm in ('A', 'B', 'D_B', 'Bl', 'Dl'):
        vk = {k: float(errs(out['val'][1][f'{arm}_k{k}'], out['val'][2], dv)['smape'].median()) for k in KS}
        k = min(vk, key=vk.get); sel_k[arm] = k
        V[arm] = errs(out['val'][1][f'{arm}_k{k}'], out['val'][2], dv); T[arm] = errs(out['test'][1][f'{arm}_k{k}'], out['test'][2], dt)
    V['f'] = errs(out['val'][1]['f'], out['val'][2], dv); T['f'] = errs(out['test'][1]['f'], out['test'][2], dt)
    print(f'== pool H{H}: ridge coefs {coef}', flush=True)
    print(f'   val-chosen k {sel_k}', flush=True)
    for k in ('raw', 'r1', 'r2'):
        sv, (m, lo, hi), fr = rep[k]
        print(f'   diag Spearman(g, {k:3s} log future err): val {sv:+.4f} | test per-traj mean {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | traj with val sign {fr:.3f}', flush=True)
    (m, lo, hi), fu, npm = rep['pair']
    print(f'   diag matched-strata pair accuracy (dir from val R1): {m:.4f} [{lo:.4f}, {hi:.4f}] (queries with >=10 pairs {fu:.3f}, median pairs {npm:.0f})', flush=True)
    for a in T: print(f'   test {a:4s} sMAPE {100 * float(T[a]["smape"].median()):6.2f}  MSE {float(T[a]["mse"].median()):.4f} | val sMAPE {100 * float(V[a]["smape"].median()):6.2f}', flush=True)
    torch.save({'T': T, 'V': V, 'k': sel_k, 'diag': rep, 'coef': coef, 't': dt['t'], 'rows': dt['rows']}, O / f'evo_pool_H{H}.pt')


# ------------------------------------------------------------------ blk mode
@torch.no_grad()
def run_blk(H):
    fcm = C.load_fc(H, dev); lam = LAM96[H]; G = [round(x, 2) for x in np.arange(-3, 3.01, 0.5)]; res = {}
    for name in ('val', 'test'):
        d, se = FC.load_set(name, dev); P = {}; Y = []
        for idx in batches(len(d['rows'])):
            rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
            y = FC.target(se, rows, t, b, H); Y.append(y.cpu())
            f = (d['f'][idx, :H].to(dev) - b['cm'][:, None]) / b['cs'][:, None]
            g = gstat(torch.exp(b['z'][..., -NB:]) - EPS)          # z holds log(e_k + 1e-4), oldest .. newest
            for gm in G:
                w2 = w * torch.exp(gm * g).masked_fill(~b['valid'], 0.0); w2 = w2 / w2.sum(1, keepdim=True).clamp_min(1e-30)
                w2 = torch.where(b['any'][:, None], w2, w)
                pr = {'r_all': (w2[..., None] * R).sum(1), 'top1': R.gather(1, w2.argmax(1)[:, None, None].expand(-1, 1, H))[:, 0]}
                pr['smap'] = pr['r_all'] + C.smap_corr(w2, qp, cp, R, [lam])[0]
                for m, p in pr.items(): P.setdefault((m, gm), []).append(torch.where(b['any'][:, None], p, f).cpu())
        Y = torch.cat(Y); res[name] = (d, {k: errs(torch.cat(v), Y, d) for k, v in P.items()})
    dv, Ev = res['val']; dt, Et = res['test']
    mC = min(('r_all', 'top1', 'smap'), key=lambda m: float(Ev[(m, 0.0)]['smape'].median()))
    gD = min(G, key=lambda gm: float(Ev[(mC, gm)]['smape'].median()))
    print(f'== blk H{H}: C = {mC} (val sMAPE {100 * float(Ev[(mC, 0.0)]["smape"].median()):.2f}); D_C gamma = {gD} (val {100 * float(Ev[(mC, gD)]["smape"].median()):.2f})', flush=True)
    print('   val sMAPE by gamma: ' + ' '.join(f'{gm:+.1f}:{100 * float(Ev[(mC, gm)]["smape"].median()):.2f}' for gm in G), flush=True)
    T = {'C': Et[(mC, 0.0)], 'D_C': Et[(mC, gD)]}; V = {'C': Ev[(mC, 0.0)], 'D_C': Ev[(mC, gD)]}
    for a in T: print(f'   test {a:4s} sMAPE {100 * float(T[a]["smape"].median()):6.2f}  MSE {float(T[a]["mse"].median()):.4f}', flush=True)
    torch.save({'T': T, 'V': V, 'method': mC, 'gamma': gD, 't': dt['t'], 'rows': dt['rows']}, O / f'evo_blk_H{H}.pt')


if __name__ == '__main__':
    mode = sys.argv[1]
    for H in HS: (run_pool if mode == 'pool' else run_blk)(H)
