"""PAT-334 (b): signal on the dense stride-1 grid (mh.py ctx512 full) vs hidden keys on the stride-16 grid (mh_hid), test,
paired bootstrap seed 0 on per-trajectory medians. Both arms use the same test queries (pat321/data)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np, torch
import fc_corr as C
O = ROOT / 'pat334' / 'out'
for H in (128, 256, 512):
    S = torch.load(O / f'mh_ctx512_full_H{H}.pt', weights_only=False)['A']; Hh = torch.load(O / f'mh_hid_H{H}.pt', weights_only=False)['A']
    assert float((S['f']['smape'] - Hh['f']['smape']).abs().max()) < 1e-2  # same queries, float noise only
    rng = np.random.default_rng(0); print(f'== H{H}')
    for m in 'cj':
        for x in (f'{m}_whole', f'{m}_M2'):
            for ref in (f'hid_whole_{m}', f'hid_seg16_{m}'):
                o = []
                for met, sc in (('smape', 100), ('mse', 1)):
                    d0, lo, hi, _ = C.boot(sc * S[x][met].numpy(), sc * Hh[ref][met].numpy(), rng); o.append(f'{met} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
                print(f'   sig {x:9s} - {ref:13s} ' + ' | '.join(o), flush=True)
