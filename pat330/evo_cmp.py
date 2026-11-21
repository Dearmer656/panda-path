"""PAT-330 evo: A-D table. Strongest baseline among A / B / C chosen by val median sMAPE; D arms vs their base and vs the
strongest; per-trajectory medians, paired bootstrap seed 0. usage: python pat330/evo_cmp.py"""
from pathlib import Path
import numpy as np, torch
O = Path(__file__).resolve().parent / 'out'


def boot(x, y, rng, B=2000):
    idx = rng.integers(0, len(x), (B, len(x))); ds = np.median(x[idx], 1) - np.median(y[idx], 1)
    return float(np.median(x) - np.median(y)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


for H in (128, 256, 512):
    Pp = torch.load(O / f'evo_pool_H{H}.pt', weights_only=False); Pb = torch.load(O / f'evo_blk_H{H}.pt', weights_only=False)
    assert torch.equal(Pp['t'], Pb['t']) and torch.equal(Pp['rows'], Pb['rows']), 'test sets differ'
    T = Pp['T'] | Pb['T']; V = Pp['V'] | Pb['V']
    best = min(('A', 'B', 'C'), key=lambda a: float(V[a]['smape'].median()))
    print(f'== H{H}: strongest baseline on val = {best} (k {Pp["k"]}, C = {Pb["method"]}, gamma {Pb["gamma"]})')
    for a in ('f', 'A', 'B', 'Bl', 'C', 'D_B', 'Dl', 'D_C'):
        print(f'   {a:4s} test sMAPE {100 * float(T[a]["smape"].median()):6.2f}  MSE {float(T[a]["mse"].median()):.4f}   val sMAPE {100 * float(V[a]["smape"].median()):6.2f}')
    rng = np.random.default_rng(0)
    for x, ref in (('D_B', 'B'), ('Dl', 'Bl'), ('D_C', 'C'), ('D_B', best), ('D_C', best), ('B', 'A'), ('C', 'A')):
        o = [f'{met} %+.3f [%+.3f, %+.3f]' % boot(sc * T[x][met].numpy(), sc * T[ref][met].numpy(), rng) for met, sc in (('smape', 100), ('mse', 1))]
        print(f'   {x:4s} - {ref:4s} ' + ' | '.join(o))
