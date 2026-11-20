"""PAT-333: fair long-ctx table. ens (joint-state top-k, val-chosen) vs every backbone variant at the same origins
(PaTH / RoPE truncated to the last 512 steps, PaTH on the full ctx = length extrapolation) and vs constant fusion.
Per-trajectory medians, paired bootstrap seed 0. usage: python pat333/vsbb.py 2048"""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'pat327'))


def boot(x, y, rng, B=2000):
    n = len(x); idx = rng.integers(0, n, (B, n)); ds = np.median(x[idx], 1) - np.median(y[idx], 1)
    return float(np.median(x) - np.median(y)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


C = int(sys.argv[1]); TR = torch.load(ROOT / 'pat330' / 'out' / f'trunc_ctx{C}.pt', weights_only=False)
for H in (128, 256, 512):
    K = torch.load(ROOT / 'pat333' / 'out' / f'topk_ctx{C}_ctx_H{H}.pt', weights_only=False)
    Sg = torch.load(ROOT / 'pat333' / 'out' / f'sig_ctx{C}_ctx_H{H}.pt', weights_only=False)
    A = {'ens': K['T']['ens'], 'joint': K['T']['joint'], 'fuse_const': Sg['T']['fuse_const']} | {n: TR[H][n] for n in ('PaTH_trunc512', 'RoPE_trunc512', 'PaTH_full')}
    print(f'== ctx {C} H{H} (ens = {K["best"]})')
    for m, e in A.items(): print(f'   {m:14s} sMAPE {100 * float(e["smape"].median()):6.2f}  MSE {float(e["mse"].median()):.4f}')
    rng = np.random.default_rng(0)
    for ref in ('PaTH_trunc512', 'RoPE_trunc512', 'PaTH_full', 'fuse_const', 'joint'):
        o = [f'{met} {"%+.3f [%+.3f, %+.3f]" % boot(sc * A["ens"][met].numpy(), sc * A[ref][met].numpy(), rng)}' for met, sc in (('smape', 100), ('mse', 1))]
        print(f'   ens - {ref:14s} ' + ' | '.join(o))
