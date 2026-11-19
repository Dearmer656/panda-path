"""PAT-329: context parroting (Zhang & Gilpin, arXiv 2505.11349) on our windows, vs the S-map family.
  parrot_ctx   paper version: 1-nearest neighbour of the last D points inside the backbone context (t - C, t); copy what
               followed the match, extended periodically (period t - s) when the horizon exceeds the copied segment
  parrot_full  same 1-NN copy but searching the whole own history [0, t) with fully observed H-step continuations
               (s + H <= t), i.e. the same information as r-only / S-map
D (match length) chosen on val from {8, 16, 32, 64, 96}. Euclidean distance on context-standardised values.
usage: python pat329/parrot.py --ctx 512 --H 128"""
import argparse, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--ctx', type=int, default=512); ap.add_argument('--H', type=int, default=128); ap.add_argument('--bs', type=int, default=64)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
DD = ROOT / 'pat321' / ('data' if a.ctx == 512 else f'data_ctx{a.ctx}_path')
os.environ['P321_OUT'] = str(DD); os.environ['P321_CTX'] = str(a.ctx)
sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np
import torch
import fc_corr as C  # noqa: E402
FC = C.FC
dev = torch.device('cuda'); H = a.H
DS = (8, 16, 32, 64, 96)


@torch.no_grad()
def predict(name):
    d = torch.load(DD / f'{name}.pt'); S = d['S'].to(dev)
    out = {f'{v}_{D}': [] for v in ('ctx', 'full') for D in DS}; out['y'] = []
    for i in range(0, len(d['rows']), a.bs):
        rows, t = d['rows'][i:i + a.bs].to(dev), d['t'][i:i + a.bs].to(dev)
        B = len(rows); ar = torch.arange(B, device=dev)
        X = S[rows]
        ctx = X.gather(1, t[:, None] - a.ctx + torch.arange(a.ctx, device=dev)[None])
        cm, cs = ctx.mean(1), ctx.std(1, unbiased=False).clamp_min(1e-10)
        Z = (X - cm[:, None]) / cs[:, None]
        out['y'].append(Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)].cpu())
        f = ((d['f'][i:i + a.bs, :H].to(dev) - cm[:, None]) / cs[:, None])                 # fallback: backbone
        for D in DS:
            q = Z[ar[:, None], t[:, None] + torch.arange(-D, 0, device=dev)]               # [B, D]
            # candidate match ends s: window Z[s - D : s], continuation starts at s
            Nmax = int(t.max())
            s = torch.arange(D, Nmax, device=dev)                                         # [N]
            sc = s[None].expand(B, -1).clamp(max=(t - 1)[:, None])
            win = Z[ar[:, None, None], sc[..., None] + torch.arange(-D, 0, device=dev)]   # [B, N, D]
            dist = (win - q[:, None]).square().sum(-1)
            for v in ('ctx', 'full'):
                if v == 'ctx':
                    ok = (s[None] - D >= (t - a.ctx)[:, None]) & (s[None] <= (t - 1)[:, None])
                else:
                    ok = s[None] + H <= t[:, None]
                dm = dist.masked_fill(~ok, float('inf'))
                best = dm.argmin(1); has = torch.isfinite(dm.min(1).values)
                sb = s[best]
                h = torch.arange(H, device=dev)[None]
                if v == 'ctx':
                    per = (t - sb).clamp_min(1)[:, None]
                    idx = sb[:, None] + (h % per)
                else:
                    idx = sb[:, None] + h
                pr = Z[ar[:, None], idx.clamp(max=X.shape[1] - 1)]
                out[f'{v}_{D}'].append(torch.where(has[:, None], pr, f).cpu())
    return d, {k: torch.cat(v) for k, v in out.items()}


dv, pv = predict('val')
vs = {k: float(FC.smape_traj(v, pv['y'], dv['ids'], dv['inst']).median()) for k, v in pv.items() if k != 'y'}
bc = min((k for k in vs if k.startswith('ctx')), key=vs.get); bf = min((k for k in vs if k.startswith('full')), key=vs.get)
dt, pt = predict('test')
sm = {'parrot_ctx': FC.smape_traj(pt[bc], pt['y'], dt['ids'], dt['inst']).numpy(),
      'parrot_full': FC.smape_traj(pt[bf], pt['y'], dt['ids'], dt['inst']).numpy()}
# S-map family on the same windows (pat329 outputs; same trajectory order)
if a.ctx == 512:
    p = torch.load(ROOT / 'pat329' / 'out' / f'inst_H{H}.pt'); sm.update({k: p[k].numpy() for k in ('smap_fc', 'smap_classic', 'r_only', 'panda')})
rng = np.random.default_rng(0)
res = dict(ctx=a.ctx, H=H, D_ctx=bc, D_full=bf, val=vs, median={k: float(np.median(v)) for k, v in sm.items()}, cmp={})
if a.ctx == 512:
    for x in ('smap_fc', 'smap_classic', 'r_only'):
        for y in ('parrot_ctx', 'parrot_full'):
            res['cmp'][f'{x}-{y}'] = C.boot(sm[x], sm[y], rng)
    res['cmp']['parrot_ctx-panda'] = C.boot(sm['parrot_ctx'], sm['panda'], rng)
print(f'[test] ctx {a.ctx} H{H} ({bc}, {bf}) ' + ' '.join(f'{k} {v:.3f}' for k, v in res['median'].items()), flush=True)
for k, (d0, lo, hi, fr) in res['cmp'].items():
    print(f'   {k:26s} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] frac {fr:.3f}', flush=True)
torch.save({k: torch.from_numpy(v) for k, v in sm.items() if k.startswith('parrot')}, ROOT / 'pat329' / 'out' / f'parrot_ctx{a.ctx}_H{H}.pt')
json.dump(res, open(ROOT / 'pat329' / 'out' / f'parrot_ctx{a.ctx}_H{H}.json', 'w'), indent=1)
