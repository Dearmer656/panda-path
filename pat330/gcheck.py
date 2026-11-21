"""PAT-330 Task 1 check: g == (log(d6+eps) - log(d1+eps))/5 numerically on real val candidates (pool mode). usage: P321_OUT=pat321/data python pat330/gcheck.py"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch, evo as E
d, se = E.FC.load_set('val', E.dev)
for H in E.HS:
    b = E.pool_batch(se, d, torch.arange(64), H, with_f=False); ok = b['ok']; ld = torch.log(b['dj'] + E.EPS)
    tel = (ld[..., -1] - ld[..., 0]) / 5; diff = (b['g'] - tel)[ok].abs()
    print(f'H{H}: {int(ok.sum())} candidates, max |g - telescoped| = {diff.max().item():.3e}, std(g) = {b["g"][ok].std().item():.3f}')
