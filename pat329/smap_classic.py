"""PAT-329: classical (training-free) S-map and simplex on the trajectory's own history vs S-map on FreshCast weights.

Library: every 96-step window of [0, t) whose next H steps are observed (window end s in [96, t - H]), raw values,
context-standardised (as the official metric). Coordinates: 12 delay points (every 8th step of the window, SUB of pat327).
  smap-classic  w_s = exp(-theta d_s / dbar) (d_s Euclidean to the current window's coordinates, dbar = mean over the
                library), weighted ridge with intercept -> H-step forecast  (pat327 smap_corr: ybar + (x_t - xbar) B)
                theta in {0, 1, 2, 4, 8}, lambda in {0.01, 0.1, 1}: chosen on val (median sMAPE)
  simplex       13 nearest (E + 1) windows, w = exp(-d / d_1), weighted mean of their futures
Compared with smap-fc / r-only (pat327/out/inst_H*.pt) and the Panda base (pat321/data/inst_fc_H*_s99.pt), same order.
usage: python pat329/smap_classic.py --H 128"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('P321_OUT', str(ROOT / 'pat321' / 'data'))
sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np
import torch
import fc_corr as C  # noqa: E402
FC = C.FC

OUT = ROOT / 'pat329' / 'out'
LQ = FC.LQ
THETAS = (0.0, 1.0, 2.0, 4.0, 8.0)
LAMS = (0.01, 0.1, 1.0)
KNN = 13


def library(S, rows, t, H, dev):
    """windows [B, N, LQ], futures [B, N, H], valid [B, N], query window [B, LQ]; context-standardised."""
    B = len(rows)
    X = S[rows]                                                                           # [B, T]
    ctx = X.gather(1, t[:, None] - FC.CTX + torch.arange(FC.CTX, device=dev)[None])
    cm, cs = ctx.mean(1), ctx.std(1, unbiased=False).clamp_min(1e-10)
    Z = (X - cm[:, None]) / cs[:, None]
    N = int((t - H - LQ).max()) + 1
    s = LQ + torch.arange(N, device=dev)                                                  # window end (exclusive)
    valid = s[None] <= (t - H)[:, None]
    sc = torch.minimum(s[None].expand(B, -1), (t - H)[:, None].clamp_min(LQ))
    win = Z[torch.arange(B, device=dev)[:, None, None], sc[..., None] + torch.arange(-LQ, 0, device=dev)]
    fut = Z[torch.arange(B, device=dev)[:, None, None], sc[..., None] + torch.arange(H, device=dev)]
    q = Z[torch.arange(B, device=dev)[:, None], t[:, None] + torch.arange(-LQ, 0, device=dev)]
    assert bool(((sc + H) <= t[:, None])[valid].all()), 'leakage'
    return win.clamp(-FC.CLIP, FC.CLIP), fut.clamp(-FC.CLIP, FC.CLIP), valid, q.clamp(-FC.CLIP, FC.CLIP)


@torch.no_grad()
def predict(name, H, dev, bs):
    d = torch.load(Path(os.environ['P321_OUT']) / f'{name}.pt')
    S = d['S'].to(dev)
    outs = {f'smap_{th}_{l}': [] for th in THETAS for l in LAMS}; outs['simplex'] = []; outs['y'] = []
    for i in range(0, len(d['rows']), bs):
        rows, t = d['rows'][i:i + bs].to(dev), d['t'][i:i + bs].to(dev)
        win, fut, valid, q = library(S, rows, t, H, dev)
        Xd, Xq = win[..., C.SUB], q[:, C.SUB]
        dist = (Xd - Xq[:, None]).norm(dim=-1).masked_fill(~valid, float('inf'))
        dbar = (dist.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1))[:, None].clamp_min(1e-8)
        has = valid.any(1)[:, None]
        y = (S[rows[:, None], t[:, None] + torch.arange(H, device=dev)] - 0)                 # raw; standardise below
        ctx = S[rows[:, None], t[:, None] - FC.CTX + torch.arange(FC.CTX, device=dev)]
        cm, cs = ctx.mean(1, keepdim=True), ctx.std(1, unbiased=False, keepdim=True).clamp_min(1e-10)
        y = (y - cm) / cs
        last = ((d['f'][i:i + bs, :H].to(dev) - cm) / cs)                                  # no library: backbone (as pat327)
        outs['y'].append(y.cpu())
        for th in THETAS:
            w = torch.exp(-th * dist / dbar).masked_fill(~valid, 0)
            w = w / w.sum(1, keepdim=True).clamp_min(1e-30)
            r0 = (w[..., None] * fut).sum(1)
            for l, c in zip(LAMS, C.smap_corr(w, q, win, fut, LAMS)):
                outs[f'smap_{th}_{l}'].append(torch.where(has, r0 + c, last).cpu())
        k = min(KNN, dist.shape[1])
        dk, ik = dist.topk(k, 1, largest=False)
        wk = torch.exp(-dk / dk[:, :1].clamp_min(1e-8)).masked_fill(~torch.isfinite(dk), 0)
        wk = wk / wk.sum(1, keepdim=True).clamp_min(1e-30)
        sx = (wk[..., None] * fut.gather(1, ik[..., None].expand(-1, -1, H))).sum(1)
        outs['simplex'].append(torch.where(has, sx, last).cpu())
    return d, {k: torch.cat(v) for k, v in outs.items()}


def main(a):
    dev = torch.device('cuda'); H = a.H
    dv, pv = predict('val', H, dev, a.bs)
    vs = {k: float(FC.smape_traj(v, pv['y'], dv['ids'], dv['inst']).median()) for k, v in pv.items() if k.startswith('smap')}
    best = min(vs, key=vs.get)
    print(f'[val] H{H} best {best} {vs[best]:.3f} | simplex {FC.smape_traj(pv["simplex"], pv["y"], dv["ids"], dv["inst"]).median():.3f}', flush=True)
    dt, pt = predict('test', H, dev, a.bs)
    sm = {'smap_classic': FC.smape_traj(pt[best], pt['y'], dt['ids'], dt['inst']).numpy(),
          'simplex': FC.smape_traj(pt['simplex'], pt['y'], dt['ids'], dt['inst']).numpy()}
    p7 = torch.load(ROOT / 'pat327' / 'out' / f'inst_H{H}_s99.pt'); p1 = torch.load(ROOT / 'pat321' / 'data' / f'inst_fc_H{H}_s99.pt')
    sm['smap_fc'], sm['r_only'], sm['panda'] = p7['smap'].numpy(), p7['fc'].numpy(), p1['base'].numpy()
    print(f"[check] r-only median pat327 {np.median(sm['r_only']):.4f} vs pat321 {np.median(p1['r'].numpy()):.4f}; n {len(sm['r_only'])} {len(sm['smap_classic'])}", flush=True)
    assert len(sm['r_only']) == len(sm['smap_classic']) == len(sm['panda'])
    rng = np.random.default_rng(0)
    res = dict(best=best, val=vs, median={k: float(np.median(v)) for k, v in sm.items()}, cmp={})
    for x, y in [('smap_classic', 'smap_fc'), ('smap_classic', 'r_only'), ('smap_classic', 'panda'), ('simplex', 'r_only'), ('smap_fc', 'panda')]:
        res['cmp'][f'{x}-{y}'] = C.boot(sm[x], sm[y], rng)
    print(f'[test] H{H} ' + ' '.join(f'{k} {v:.3f}' for k, v in res['median'].items()), flush=True)
    for k, (d0, lo, hi, fr) in res['cmp'].items():
        print(f'   {k:22s} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] frac {fr:.3f}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({k: torch.from_numpy(v) for k, v in sm.items()}, OUT / f'inst_H{H}.pt')
    json.dump(res, open(OUT / f'results_H{H}.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--H', type=int, default=128); ap.add_argument('--bs', type=int, default=24)
    main(ap.parse_args())
