"""PAT-329 top-k selection: per H pick the (LQ, method, k, lambda) with the lowest val median; paired bootstrap vs parrot-full."""
import json, sys, numpy as np, torch
sys.path.insert(0, 'pat327')
from fc_corr import boot
for H in (128, 256, 512):
    cands = []
    for LQ in (32, 64, 96):
        j = json.load(open(f'pat329/out/topk_lq{LQ}_H{H}.json'))
        cands += [(v, LQ, k) for k, v in j['val'].items()]
    v, LQ, k = min(cands)
    x = torch.load(f'pat329/out/topk_lq{LQ}_H{H}.pt')[k].numpy()
    p = torch.load(f'pat329/out/parrot_ctx512_H{H}.pt')['parrot_full'].numpy()
    d0, lo, hi, fr = boot(x, p, np.random.default_rng(0))
    allk = {kk: json.load(open(f'pat329/out/topk_lq{LQ}_H{H}.json'))['test'][kk] for kk in ('r_k1', 'r_k5', 'r_k20', 'r_k400', 'smap_k20_l0.01', 'smap_k400_l0.01')}
    print(f'H{H}: selected LQ {LQ} {k} (val {v:.3f}) test {np.median(x):.3f} vs parrot-full {np.median(p):.3f}: {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] frac {fr:.3f} | same LQ test: ' + ' '.join(f'{a} {b:.2f}' for a, b in allk.items()))
