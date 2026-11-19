"""Paired bootstrap: long-ctx MSE (retrieval vs PaTH / RoPE) and stride (denser vs 8) sMAPE / MSE."""
import sys, numpy as np, torch
sys.path.insert(0, 'pat327')
from fc_corr import boot
rng = np.random.default_rng(0)
for c in (1024, 2048):
    for H in (128, 256, 512):
        a = torch.load(f'pat329/out/mse_ctx{c}_lq96_H{H}.pt'); r = torch.load(f'pat329/out/mse_ctx{c}_panda_lq96_H{H}.pt')['panda@LQ96']['mse'].numpy()
        pth = a['panda@LQ96']['mse'].numpy(); sm = [k for k in a if k.startswith('smap')][0]
        for k in ('r_all@LQ96', sm):
            x = a[k]['mse'].numpy()
            d1 = boot(x, pth, rng); d2 = boot(x, r, rng)
            print(f'ctx{c} H{H} {k:15s} MSE {np.median(x):.4f} | - PaTH {d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}] | - RoPE {d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}]')
for q in (64, 96):
    for H in (128, 256, 512):
        s = torch.load(f'pat329/out/stride_lq{q}_H{H}.pt')
        for st in (1, 2, 4):
            ds = boot(s[st]['smape'].numpy(), s[8]['smape'].numpy(), rng); dm = boot(s[st]['mse'].numpy(), s[8]['mse'].numpy(), rng)
            print(f'LQ{q} H{H} stride {st} - stride 8: sMAPE {100 * ds[0]:+.3f} [{100 * ds[1]:+.3f}, {100 * ds[2]:+.3f}] | MSE {dm[0]:+.4f} [{dm[1]:+.4f}, {dm[2]:+.4f}]')
