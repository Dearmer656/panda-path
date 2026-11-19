"""PAT-329 MSE: paired bootstrap of per-trajectory MSE, best smap config / smap-fc / r-only vs parrot-full."""
import sys, glob, numpy as np, torch
sys.path.insert(0, 'pat327')
from fc_corr import boot
BEST = {128: 'smap_0.01@LQ32', 256: 'smap_0.01@LQ64', 512: 'smap_0.1@LQ64'}
for H in (128, 256, 512):
    r = {}
    for f in glob.glob(f'pat329/out/mse_lq*_H{H}.pt'): r.update(torch.load(f))
    print(f'H{H} median MSE: ' + ' '.join(f'{k} {float(v["mse"].median()):.4f}' for k, v in sorted(r.items())))
    p = r['parrot_full@LQ96']['mse'].numpy(); rng = np.random.default_rng(0)
    for k in (BEST[H], [k for k in r if k.startswith('smap') and k.endswith('LQ96')][0], 'r_all@LQ96', 'r_top1@LQ96', 'panda@LQ96'):
        d0, lo, hi, fr = boot(r[k]['mse'].numpy(), p, rng)
        print(f'   {k:18s} - parrot_full  {d0:+.4f} [{lo:+.4f}, {hi:+.4f}] frac {fr:.3f}')
