"""PAT-334 stride ablation table. Per setting and H, arm = headline-mode M2 (val-chosen) of each stride run (all with a 512
window before every anchor). Criteria (fixed before results):
  S1 resolution : s16g - s1g sMAPE CI > 0 at all H in both settings.
  reported      : monotone medians over s = 1, 2, 4, 8, 16; grid vs random subset (s g - s r); s16g - backbone f; MSE."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np, torch
import fc_corr as C
O = ROOT / 'pat334' / 'out'
for tag in ('ctx512_full', 'ctx2048_ctx'):
    for H in (128, 256, 512):
        R = {}
        for s in (1, 2, 4, 8, 16):
            for sub in ('g', 'r'):
                fp = O / f'mh_{tag}_s{s}{sub}_a512_H{H}.pt'
                if fp.exists(): d = torch.load(fp, weights_only=False); R[f's{s}{sub}'] = d['A'][f'{d["hm"]}_M2']; R['f'] = d['A']['f']
        print(f'== {tag} H{H}')
        print('   ' + '  '.join(f'{k} {100 * float(v["smape"].median()):.2f}/{float(v["mse"].median()):.3f}' for k, v in R.items()))
        rng = np.random.default_rng(0)
        pairs = [('s16g', 's1g'), ('s2g', 's1g'), ('s4g', 's2g'), ('s8g', 's4g'), ('s16g', 's8g')] + [(f's{s}g', f's{s}r') for s in (2, 4, 8, 16)] + [('s16g', 'f'), ('s1g', 'f')]
        for x, ref in pairs:
            if x in R and ref in R:
                o = [f'{met} %+.3f [%+.3f, %+.3f]' % C.boot(sc * R[x][met].numpy(), sc * R[ref][met].numpy(), rng)[:3] for met, sc in (('smape', 100), ('mse', 1))]
                print(f'   {x:5s} - {ref:5s} ' + ' | '.join(o))
