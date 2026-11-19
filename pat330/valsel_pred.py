"""PAT-330 step 1: per-trajectory sMAPE / MSE on val AND test for one scorer config (env: P321_OUT, P330_BLOCKS, P330_VALONLY),
methods r_all, r_top1, smap_{lambda} for lambda in LAMS, LQ 96, ctx from P321_CTX, bs 64.
usage: P330_BLOCKS=6 P321_OUT=pat330/blk python pat330/valsel_pred.py --tag blk"""
import argparse, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327')); sys.path.insert(0, str(ROOT / 'pat329'))
import torch
import fc_corr as C  # noqa: E402
from mse_eval import per_traj  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); assert FC.LQ == 96


@torch.no_grad()
def run(name, H):
    fcm = C.load_fc(H, dev); d, se = FC.load_set(name, dev)
    acc = {}
    for i in range(0, len(d['rows']), 64):
        idx = torch.arange(i, min(i + 64, len(d['rows'])))
        rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
        y = FC.target(se, rows, t, b, H)
        f = (d['f'][idx, :H].to(dev) - b['cm'][:, None]) / b['cs'][:, None]; has = b['any'][:, None]
        r0 = (w[..., None] * R).sum(1)
        P = {'r_all': r0, 'r_top1': R.gather(1, w.argmax(1)[:, None, None].expand(-1, 1, H))[:, 0]}
        for l, c in zip(C.LAMS, C.smap_corr(w, qp, cp, R, C.LAMS)): P[f'smap_{l}'] = r0 + c
        for m, p in P.items():
            p = torch.where(has, p, f); a_ = acc.setdefault(m, dict(smape=[], mse=[]))
            a_['smape'].append(((p - y).abs() / (p.abs() + y.abs() + 1e-10)).mean(1).cpu()); a_['mse'].append((p - y).square().mean(1).cpu())
    return {m: {k: per_traj(torch.cat(v), d['ids'], d['inst']) for k, v in x.items()} for m, x in acc.items()}


import os
HS = [int(h) for h in os.environ.get('P330_HS', '128,256,512').split(',')]
out = {H: {s: run(s, H) for s in ('val', 'test')} for H in HS}
torch.save(out, ROOT / 'pat330' / 'out' / f'valsel_{a.tag}.pt')
for H in out:
    print(f'[{a.tag}] H{H} val ' + ' '.join(f'{m} {100 * float(v["smape"].median()):.2f}' for m, v in out[H]['val'].items()), flush=True)
