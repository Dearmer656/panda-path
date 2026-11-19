"""PAT-329 context scan: r-only, smap-fc, smap-classic, simplex at backbone context C (1024 / 2048) on the PAT-325 windows,
vs the RoPE and PaTH backbones on the same windows (paired per trajectory).
usage: python pat329/ctx_scan.py --ctx 1024 --H 128   (env is set here before the pat321 / pat327 / pat329 imports)"""
import argparse, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--ctx', type=int, required=True); ap.add_argument('--H', type=int, default=128); ap.add_argument('--bs', type=int, default=24)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
RD, PD = ROOT / 'pat321' / f'data_ctx{a.ctx}_rope', ROOT / 'pat321' / f'data_ctx{a.ctx}_path'
os.environ['P321_OUT'] = str(PD); os.environ['P321_CTX'] = str(a.ctx)
sys.path.insert(0, str(ROOT / 'pat329')); sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np
import torch
import fc_corr as C  # noqa: E402
import smap_classic as P  # noqa: E402
FC = C.FC
assert FC.CTX == a.ctx and Path(os.environ['P321_OUT']) == PD
dev = torch.device('cuda'); H = a.H

# FreshCast r-only and smap-fc (lambda on val)
dv, pv = C.predict('val', H, dev, [])
vf = {l: float(FC.smape_traj(pv[f'smap{l}'], pv['y'], dv['ids'], dv['inst']).median()) for l in C.LAMS}
lf = min(vf, key=vf.get)
dt, pt = C.predict('test', H, dev, [])
sm = {'r_only': FC.smape_traj(pt['fc'], pt['y'], dt['ids'], dt['inst']).numpy(),
      'smap_fc': FC.smape_traj(pt[f'smap{lf}'], pt['y'], dt['ids'], dt['inst']).numpy()}
# classical S-map / simplex (theta, lambda on val)
dv2, pv2 = P.predict('val', H, dev, a.bs)
vc = {k: float(FC.smape_traj(v, pv2['y'], dv2['ids'], dv2['inst']).median()) for k, v in pv2.items() if k.startswith('smap')}
bc = min(vc, key=vc.get)
dt2, pt2 = P.predict('test', H, dev, a.bs)
sm['smap_classic'] = FC.smape_traj(pt2[bc], pt2['y'], dt2['ids'], dt2['inst']).numpy()
sm['simplex'] = FC.smape_traj(pt2['simplex'], pt2['y'], dt2['ids'], dt2['inst']).numpy()
assert torch.allclose(pt['y'], pt2['y'], atol=1e-4), 'targets differ between pipelines'
for name, dd in (('rope', RD), ('path', PD)):
    sm[name] = torch.load(dd / f'inst_fc_H{H}_s99.pt')['base'].numpy()
assert len({len(v) for v in sm.values()}) == 1
chk = float(np.median(torch.load(PD / f'inst_fc_H{H}_s99.pt')['r'].numpy()))
print(f'[check] ctx {a.ctx} H{H}: r-only {np.median(sm["r_only"]):.4f} vs PAT-325 {chk:.4f}; smap-fc lambda {lf}; classic {bc}', flush=True)
rng = np.random.default_rng(0)
res = dict(ctx=a.ctx, H=H, lam_fc=lf, classic=bc, median={k: float(np.median(v)) for k, v in sm.items()}, cmp={})
for x in ('smap_classic', 'smap_fc', 'r_only', 'simplex'):
    for y in ('path', 'rope'):
        res['cmp'][f'{x}-{y}'] = C.boot(sm[x], sm[y], rng)
res['cmp']['smap_classic-smap_fc'] = C.boot(sm['smap_classic'], sm['smap_fc'], rng)
print(f'[test] ctx {a.ctx} H{H} ' + ' '.join(f'{k} {v:.3f}' for k, v in res['median'].items()), flush=True)
for k, (d0, lo, hi, fr) in res['cmp'].items():
    print(f'   {k:22s} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] frac {fr:.3f}', flush=True)
(ROOT / 'pat329' / 'out').mkdir(exist_ok=True)
json.dump(res, open(ROOT / 'pat329' / 'out' / f'ctx{a.ctx}_H{H}.json', 'w'), indent=1)
