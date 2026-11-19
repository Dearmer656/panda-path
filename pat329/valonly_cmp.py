"""PAT-329 value-only readouts (drop the shape readouts): paired bootstrap vs the original 400-entry FreshCast, ctx 512 LQ 96.
vmask = original scorer, shape entries masked at inference; vonly = scorer retrained with value entries only.
Criterion (fixed before results): value-only helps if the median sMAPE difference vs the same method with shape entries
has a 95% CI below 0; it closes the L512 gap if r_top1 or smap is not worse than parrot_full (CI upper end <= 0).
usage: python pat329/valonly_cmp.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np, torch
from fc_corr import boot  # noqa: E402
O = ROOT / 'pat329' / 'out'
for H in (128, 256, 512):
    base, rng = torch.load(O / f'mse_lq96_H{H}.pt'), np.random.default_rng(0)
    for tag in ('_vmask', '_vonly'):
        new = torch.load(O / f'mse{tag}_lq96_H{H}.pt')
        print(f'== H{H} {tag}', flush=True)
        for m in [k for k in new if k in base and not k.startswith(('panda', 'parrot'))]:
            for met, sc in (('smape', 100), ('mse', 1)):
                x, y = sc * new[m][met].numpy(), sc * base[m][met].numpy()
                d0, lo, hi, fr = boot(x, y, rng)
                p = sc * base['parrot_full@LQ96'][met].numpy(); e0, elo, ehi, _ = boot(x, p, rng)
                print(f'   {m:14s} {met:5s} {np.median(y):7.3f} -> {np.median(x):7.3f}  d {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]'
                      f'  vs parrot {e0:+.3f} [{elo:+.3f}, {ehi:+.3f}]', flush=True)
