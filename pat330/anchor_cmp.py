"""PAT-330 step 2 comparison: multi-anchor methods (val-selected per H among anc_mean / anc_top1 / anc_smap_l) vs
parrot-full and vs the step-1 val-selected method. Criterion (fixed before results): CI upper end < 0.
usage: python pat330/anchor_cmp.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np, torch
from fc_corr import boot  # noqa: E402
O = ROOT / 'pat330' / 'out'
STEP1 = {128: ('blk', 'smap_0.01'), 256: ('orig', 'r_top1'), 512: ('blk', 'r_top1')}
for H in (128, 256, 512):
    rng = np.random.default_rng(0); A = torch.load(O / f'anchor_H{H}.pt', map_location='cpu')
    val = {m: float(v['smape'].median()) for m, v in A['val'].items()}; best = min(val, key=val.get); te = A['test'][best]
    refs = {'parrot_full': torch.load(ROOT / 'pat329' / 'out' / f'mse_lq96_H{H}.pt')['parrot_full@LQ96'],
            f'step1 {STEP1[H][0]}/{STEP1[H][1]}': torch.load(O / f'valsel_{STEP1[H][0]}.pt')[H]['test'][STEP1[H][1]]}
    print(f'== H{H}: val-selected {best} (val {100 * val[best]:.2f}) test sMAPE {100 * float(te["smape"].median()):.2f} MSE {float(te["mse"].median()):.3f}; b = {[round(x, 2) for x in A["state"]["b"].tolist()]} tau {float(A["state"]["lt"].exp()):.4f}', flush=True)
    for n, r in refs.items():
        out = []
        for met, sc in (('smape', 100), ('mse', 1)):
            d0, lo, hi, _ = boot(sc * te[met].numpy(), sc * r[met].numpy(), rng); out.append(f'{met} {sc * float(r[met].median()):.3f} d {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        print(f'   vs {n:24s} ' + ' | '.join(out), flush=True)
