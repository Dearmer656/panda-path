"""PAT-334 metric fix (reported only, criteria unchanged). Recomputes point forecasts on test and reports:
  vpt_z   old definition (sMAPE > 30 on context-standardised Z; reproduces ens.py, sanity)
  vpt_raw Gilpin: instance channel-mean per-step sMAPE on the RAW scale first > 30 (H if never)
  vpt_n   Pathak 2018: per-step instance error sqrt(mean_c (p - y)^2) in context-std units first > 0.4 (H if never)
Arms: f (Panda), M2 (G1 point config), R (re-anchor, val-chosen S/W from re_*.pt), top1 (rank-0 analog of the
highest-weight M2 head), ensm (weighted mean of the val-chosen ens), chr_med (Chronos sample median; ctx2048 only).
Also CRPS from ens_*.pt: mean + 95% bootstrap CI of the mean paired difference, trimmed mean (drop trajectories with any
query whose truth max |Z| over H > 10; fixed before results), win rate (share of trajectories where ens < ref).
usage: same env as mh.py + P334_H"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import mh as M
import reanchor as RA
from mh import G, FC, dev, chunks, backbone, TAG, head, futs
from mse_eval import per_traj  # noqa: E402
O = ROOT / 'pat334' / 'out'; TRIM = 10.0


def first_bad(e, loc, ni, thr):
    cnt = torch.zeros(ni, device=dev).index_add_(0, loc, torch.ones(len(loc), device=dev))
    ei = torch.zeros(ni, e.shape[1], device=dev).index_add_(0, loc, e) / cnt[:, None]
    bad = ei > thr
    return torch.where(bad.any(1), bad.float().argmax(1), torch.full((ni,), e.shape[1], device=dev, dtype=torch.long)).float()[loc]


def smape(p, y): return 200 * (p - y).abs() / (p.abs() + y.abs() + 1e-10)


def vpt_n(p, y, loc, ni):                                     # Z units: sqrt of instance channel-mean squared error
    cnt = torch.zeros(ni, device=dev).index_add_(0, loc, torch.ones(len(loc), device=dev))
    se = torch.zeros(ni, p.shape[1], device=dev).index_add_(0, loc, (p - y).square()) / cnt[:, None]
    bad = se.sqrt() > 0.4
    return torch.where(bad.any(1), bad.float().argmax(1), torch.full((ni,), p.shape[1], device=dev, dtype=torch.long)).float()[loc]


@torch.no_grad()
def run(H):
    pk = torch.load(O / f'mh_{TAG}_H{H}.pt', weights_only=False); hm = pk['hm']; key = pk['pick'][f'{hm}_M2'][0]
    mode, fam, wn, k, tau = RA.parse(key); hs = M.FAM[fam]
    ens = torch.load(O / f'ens_{TAG}_H{H}.pt', weights_only=False); kb = ens['k']
    gR = tuple(torch.load(O / f're_{TAG}_H{H}.pt', weights_only=False)['R'])
    cf = O / f'chronos_{TAG}_test.pt'; chr_s = torch.load(cf)['samples'] if cf.exists() else None
    d = torch.load(FC.OUT / 'test.pt'); S = d['S'].to(dev); fall = backbone('test', d)
    arms = ['f', 'M2', 'R', 'top1', 'ensm'] + (['chr_med'] if chr_s is not None else [])
    E = {a: {m: [] for m in ('vpt_z', 'vpt_raw', 'vpt_n', 'smape_z')} for a in arms}; FL = []
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        y = Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)]
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)
        ph, has, dns, F0, HD = [], [], [], [], []
        for (l, o) in hs:
            js, vs, dn = head(Z, t, H, l, o, False, mode, loc, rep, ni)
            val = torch.isfinite(vs); fu = futs(Z, js, H); v1 = vs[:, :1]
            wfull = val.float() if wn == 'u' else torch.exp(-(vs - v1) / v1.abs().clamp_min(1e-8)).masked_fill(~val, 0.0)
            w = wfull[:, :k]
            ph.append((w[..., None] * fu[:, :k]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)); has.append(val[:, 0])
            dns.append(torch.where(val[:, 0], dn, torch.full_like(dn, float('inf')))); F0.append(fu[:, 0]); HD.append((fu, wfull))
        ph, has, dns, F0 = torch.stack(ph, 1), torch.stack(has, 1), torch.stack(dns, 1), torch.stack(F0, 1); anyh = has.any(1)
        if tau == 'uni': a = has.float()
        elif tau == 'best': a = torch.nn.functional.one_hot(dns.argmin(1), len(hs)).float() * has.float()
        else: a = torch.softmax((-dns / tau).masked_fill(~has, float('-inf')), 1).nan_to_num(0.0)
        a = a / a.sum(1, keepdim=True).clamp_min(1e-12)
        P = {'f': f}
        P['M2'] = torch.where(anyh[:, None], (a[..., None] * ph).sum(1), f)
        P['top1'] = torch.where(anyh[:, None], F0[ar, a.argmax(1)], f)
        xs = torch.cat([fu[:, :kb] for fu, _ in HD], 1)
        ws = torch.cat([a[:, i:i + 1] * wf[:, :kb] / wf[:, :kb].sum(1, keepdim=True).clamp_min(1e-12) for i, (_, wf) in enumerate(HD)], 1)
        ws = ws / ws.sum(1, keepdim=True).clamp_min(1e-12)
        P['ensm'] = torch.where(anyh[:, None], (ws[..., None] * xs).sum(1), f)
        P['R'] = RA.forecast(Z, t, f, H, gR[0], gR[1], RA.parse(key), loc, rep, ni, [Z.unfold(1, l, 1) for (l, o) in hs])
        if chr_s is not None: P['chr_med'] = ((chr_s[idx].to(dev).float()[..., :H].median(1).values - cm[:, None]) / cs[:, None]).clamp(-M.CLIP, M.CLIP)
        yr = y * cs[:, None] + cm[:, None]
        for nm, p in P.items():
            pr = p * cs[:, None] + cm[:, None]
            E[nm]['vpt_z'].append(first_bad(smape(p, y), loc, ni, 30).cpu()); E[nm]['vpt_raw'].append(first_bad(smape(pr, yr), loc, ni, 30).cpu())
            E[nm]['vpt_n'].append(vpt_n(p, y, loc, ni).cpu()); E[nm]['smape_z'].append((smape(p, y) / 200).mean(1).cpu())
        FL.append((y.abs().amax(1) > TRIM).float().cpu())
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    pt = lambda v: per_traj(torch.cat(v)[inv], d['ids'], d['inst'])
    R = {a: {m: pt(v) for m, v in mm.items()} for a, mm in E.items()}; flag = pt(FL) > 0
    # sanity: M2 point sMAPE reproduces mh.py, f / ensm vpt_z reproduce ens.py
    dM = float((R['M2']['smape_z'] - pk['A'][f'{hm}_M2']['smape']).abs().max()); dF = float((R['f']['vpt_z'] - ens['A']['f']['vpt']).abs().max())
    dE = float((R['ensm']['vpt_z'] - ens['A']['ens']['vpt']).abs().max())
    print(f'== {TAG} H{H}: M2 = {key}, R = S{gR[0]} W{gR[1]}, ens k = {kb}; sanity M2 smape {dM:.1e}, f vpt_z {dF:.1e}, ens vpt_z {dE:.1e}', flush=True)
    assert dM < 1e-2 and dF < 1e-3 and dE < 1e-3
    print(f'   VPT (steps) median / mean   [trajectories {len(flag)}]', flush=True)
    for a in arms: print(f'   {a:8s} ' + '  '.join(f'{m} {float(R[a][m].median()):6.1f} / {float(R[a][m].mean()):6.1f}' for m in ('vpt_z', 'vpt_raw', 'vpt_n')), flush=True)
    rng = np.random.default_rng(0)
    for x in ('M2', 'R', 'top1', 'ensm') + (('chr_med',) if chr_s is not None else ()):
        o = [f'{m} med %+.1f [%+.1f, %+.1f]' % G.C.boot(R[x][m].numpy(), R['f'][m].numpy(), rng)[:3] for m in ('vpt_raw', 'vpt_n')]
        print(f'   {x:8s} - f  ' + ' | '.join(o), flush=True)
    A = ens['A']; keep = ~flag
    print(f'   CRPS: trimmed drops {int(flag.sum())} / {len(flag)} trajectories (truth max|Z| > {TRIM})', flush=True)
    for r in [r for r in ('f', 'chr', 'clim') if r in A]:
        e, b = A['ens']['crps'].numpy().astype(np.float64), A[r]['crps'].numpy().astype(np.float64); dd = e - b; n = len(dd)
        bs = np.array([dd[rng.integers(0, n, n)].mean() for _ in range(2000)])
        kk = keep.numpy(); dk = dd[kk]; bk = np.array([dk[rng.integers(0, len(dk), len(dk))].mean() for _ in range(2000)])
        print(f'   ens - {r:4s} CRPS mean {e.mean():.4f} vs {b.mean():.4f}: {dd.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] | '
              f'trimmed {e[kk].mean():.4f} vs {b[kk].mean():.4f}: {dk.mean():+.4f} [{np.percentile(bk, 2.5):+.4f}, {np.percentile(bk, 97.5):+.4f}] | win {(dd < 0).mean():.3f}', flush=True)
    torch.save({'R': R, 'flag': flag}, O / f'vpt2_{TAG}_H{H}.pt')


if __name__ == '__main__':
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else (128, 256, 512)): run(H)
