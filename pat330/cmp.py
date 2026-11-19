"""PAT-330 paired bootstrap: segment-descriptor scorers vs the same method without them (ctx 512, LQ 96, test).
blk = blocks + value/shape entries vs original (mse_lq96); blkv = blocks + value only vs value-only (mse_vonly_lq96).
Criterion (fixed before results): segments help if the median sMAPE difference vs the same method without them has a
95% CI below 0 (MSE reported alongside); beats parrot-full if CI upper end <= 0.
usage: python pat330/cmp.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np, torch
from fc_corr import boot  # noqa: E402
O = ROOT / 'pat329' / 'out'
for H in (128, 256, 512):
    rng = np.random.default_rng(0)
    for tag, ref in (('_blk', ''), ('_blkv', '_vonly')):
        new, base = torch.load(O / f'mse{tag}_lq96_H{H}.pt'), torch.load(O / f'mse{ref}_lq96_H{H}.pt')
        print(f'== H{H} {tag} vs {ref or "original"}', flush=True)
        for m in [k for k in new if k in base and not k.startswith(('panda', 'parrot'))]:
            for met, sc in (('smape', 100), ('mse', 1)):
                x, y = sc * new[m][met].numpy(), sc * base[m][met].numpy()
                d0, lo, hi, _ = boot(x, y, rng)
                p = sc * base['parrot_full@LQ96'][met].numpy(); e0, elo, ehi, _ = boot(x, p, rng)
                print(f'   {m:14s} {met:5s} {np.median(y):7.3f} -> {np.median(x):7.3f}  d {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]'
                      f'  vs parrot {e0:+.3f} [{elo:+.3f}, {ehi:+.3f}]', flush=True)
