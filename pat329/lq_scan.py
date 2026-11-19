"""PAT-329 LQ scan (ctx 512): FreshCast scorer retrained with query window LQ (env P321_LQ, P321_OUT = pat329/lq{LQ}, whose
train/val/test.pt link to pat321/data); r-only, smap-fc (lambda on val), smap-classic (theta, lambda on val) on the official
windows, paired vs LQ 96 smap-fc (pat327) and parrot-full (pat329/out/parrot_ctx512).
usage: P321_LQ=32 P321_OUT=pat329/lq32 python pat329/lq_scan.py --H 128"""
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--H', type=int, default=128); ap.add_argument('--bs', type=int, default=24)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat329')); sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np
import torch
import fc_corr as C  # noqa: E402
import smap_classic as P  # noqa: E402
FC = C.FC
LQ = FC.LQ; assert LQ == int(os.environ['P321_LQ']) and len(C.SUB) >= 1
dev = torch.device('cuda'); H = a.H
dv, pv = C.predict('val', H, dev, [])
vf = {l: float(FC.smape_traj(pv[f'smap{l}'], pv['y'], dv['ids'], dv['inst']).median()) for l in C.LAMS}; lf = min(vf, key=vf.get)
dt, pt = C.predict('test', H, dev, [])
sm = {'r_only': FC.smape_traj(pt['fc'], pt['y'], dt['ids'], dt['inst']).numpy(), 'smap_fc': FC.smape_traj(pt[f'smap{lf}'], pt['y'], dt['ids'], dt['inst']).numpy()}
dv2, pv2 = P.predict('val', H, dev, a.bs)
vc = {k: float(FC.smape_traj(v, pv2['y'], dv2['ids'], dv2['inst']).median()) for k, v in pv2.items() if k.startswith('smap')}; bc = min(vc, key=vc.get)
dt2, pt2 = P.predict('test', H, dev, a.bs)
sm['smap_classic'] = FC.smape_traj(pt2[bc], pt2['y'], dt2['ids'], dt2['inst']).numpy()
sm['smap_fc_96'] = torch.load(ROOT / 'pat327' / 'out' / f'inst_H{H}_s99.pt')['smap'].numpy()
sm['r_only_96'] = torch.load(ROOT / 'pat327' / 'out' / f'inst_H{H}_s99.pt')['fc'].numpy()
sm['parrot_full'] = torch.load(ROOT / 'pat329' / 'out' / f'parrot_ctx512_H{H}.pt')['parrot_full'].numpy()
assert len({len(v) for v in sm.values()}) == 1
rng = np.random.default_rng(0)
res = dict(LQ=LQ, H=H, lam_fc=lf, classic=bc, n_sub=len(C.SUB), median={k: float(np.median(v)) for k, v in sm.items()}, cmp={})
for x, y in [('smap_fc', 'smap_fc_96'), ('r_only', 'r_only_96'), ('smap_fc', 'parrot_full'), ('smap_classic', 'parrot_full'), ('r_only', 'parrot_full')]:
    res['cmp'][f'{x}-{y}'] = C.boot(sm[x], sm[y], rng)
print(f'[test] LQ {LQ} H{H} (lambda {lf}, {bc}, {len(C.SUB)} coords) ' + ' '.join(f'{k} {v:.3f}' for k, v in res['median'].items()), flush=True)
for k, (d0, lo, hi, fr) in res['cmp'].items():
    print(f'   {k:26s} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] frac {fr:.3f}', flush=True)
json.dump(res, open(ROOT / 'pat329' / 'out' / f'lq{LQ}_H{H}.json', 'w'), indent=1)
