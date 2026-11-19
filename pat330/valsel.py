"""PAT-330 step 1: per H pick (config, method, lambda) by val median sMAPE over configs {orig, vonly, blk, blkv} x
{r_all, r_top1, smap_lambda}; report its test sMAPE / MSE vs parrot-full (D_FULL chosen on val in PAT-329).
Criterion (fixed before results): the val-selected method beats parrot-full at H if the paired test median sMAPE difference
has a 95% CI upper end < 0.
usage: python pat330/valsel.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np, torch
from fc_corr import boot  # noqa: E402
O = ROOT / 'pat330' / 'out'
R = {c: torch.load(O / f'valsel_{c}.pt') for c in ('orig', 'vonly', 'blk', 'blkv')}
for H in (128, 256, 512):
    rng = np.random.default_rng(0)
    val = {(c, m): float(R[c][H]['val'][m]['smape'].median()) for c in R for m in R[c][H]['val']}
    best = min(val, key=val.get)
    par = torch.load(ROOT / 'pat329' / 'out' / f'mse_lq96_H{H}.pt')['parrot_full@LQ96']
    te = R[best[0]][H]['test'][best[1]]
    print(f'== H{H}: val-selected {best} val {100 * val[best]:.2f}; top-5 val: ' +
          ', '.join(f'{k[0]}/{k[1]} {100 * v:.2f}' for k, v in sorted(val.items(), key=lambda x: x[1])[:5]), flush=True)
    for met, sc in (('smape', 100), ('mse', 1)):
        x, y = sc * te[met].numpy(), sc * par[met].numpy(); d0, lo, hi, fr = boot(x, y, rng)
        print(f'   test {met:5s} {np.median(x):7.3f} vs parrot-full {np.median(y):7.3f}: {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] frac {fr:.3f}', flush=True)
