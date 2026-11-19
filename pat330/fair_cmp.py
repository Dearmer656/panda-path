"""PAT-330 step 3: retrieval restricted to the backbone context [t - ctx, t) (candidate window, its future and history
statistics), blk scorer retrained per ctx. Per (ctx, H): method picked on val (r_all / r_top1 / smap_lambda), test vs the
backbones (RoPE; PaTH at ctx 1024/2048) and parrot-ctx (1-NN inside ctx, periodic copy).
Criterion (fixed before results): restricted retrieval beats a backbone at (ctx, H) if the paired test median sMAPE
difference has 95% CI upper end < 0; MSE reported alongside. ctx 512 / H512 impossible (96 + 512 > 512) -> not run.
usage: python pat330/fair_cmp.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np, torch
from fc_corr import boot  # noqa: E402
O, O9 = ROOT / 'pat330' / 'out', ROOT / 'pat329' / 'out'
for C in (512, 1024, 2048):
    R = torch.load(O / f'valsel_fair{C}.pt')
    for H in R:
        rng = np.random.default_rng(0)
        val = {m: float(v['smape'].median()) for m, v in R[H]['val'].items()}; best = min(val, key=val.get); te = R[H]['test'][best]
        if C == 512:
            b = torch.load(O9 / f'mse_lq96_H{H}.pt'); refs = {'RoPE': b['panda@LQ96'], 'parrot_ctx': b['parrot_ctx@LQ96'], 'parrot_full': b['parrot_full@LQ96']}
        else:
            b = torch.load(O9 / f'mse_ctx{C}_lq96_H{H}.pt'); r = torch.load(O9 / f'mse_ctx{C}_panda_lq96_H{H}.pt')
            refs = {'PaTH': b['panda@LQ96'], 'RoPE': r['panda@LQ96'], 'parrot_ctx': b['parrot_ctx@LQ96'], 'parrot_full': b['parrot_full@LQ96']}
        print(f'== ctx {C} H{H}: val-selected {best} (val {100 * val[best]:.2f}); test sMAPE {100 * float(te["smape"].median()):.2f} MSE {float(te["mse"].median()):.3f}', flush=True)
        for n, ref in refs.items():
            out = []
            for met, sc in (('smape', 100), ('mse', 1)):
                d0, lo, hi, _ = boot(sc * te[met].numpy(), sc * ref[met].numpy(), rng)
                out.append(f'{met} {sc * float(ref[met].median()):.3f} d {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
            print(f'   vs {n:11s} ' + ' | '.join(out), flush=True)
