"""PAT-325: paired comparison of FreshCast on the PaTH backbone (pat321/data_pathbb) vs the RoPE backbone (pat321/data,
PAT-321 job 37197). Per-trajectory sMAPE vectors (official aggregation, same 9347 trajectories in the same order);
paired bootstrap (2000 resamples of trajectories, seed 0) of the median difference.
  K1  PaTH+fc < PaTH        at L256 and L512, CI < 0
  K2  PaTH+fc < RoPE+fc     CI < 0 at >= 2 of L128/256/512
usage: python pat321/cmp_pathbb.py [arm=fc]"""
import sys
import numpy as np
import torch

ARM = sys.argv[1] if len(sys.argv) > 1 else 'fc'
rng = np.random.default_rng(0)


def load(d, H):
    x = torch.load(f'pat321/{d}/inst_{ARM}_H{H}_s99.pt')
    return {k: x[k].numpy() for k in ('base', 'comb', 'r')}


def boot(a, b, B=2000):
    n = len(a); d0 = np.median(a) - np.median(b)
    idx = rng.integers(0, n, (B, n))
    ds = np.median(a[idx], 1) - np.median(b[idx], 1)
    return d0, np.percentile(ds, 2.5), np.percentile(ds, 97.5), float((a < b).mean())


k1, k2 = [], []
print(f'arm {ARM}   (sMAPE median over trajectories; delta = first - second, 95% paired bootstrap CI; frac = share of trajectories where first < second)')
for H in (128, 256, 512):
    P, R = load('data_pathbb', H), load('data', H)
    assert len(P['base']) == len(R['base'])
    print(f'L{H}: RoPE {np.median(R["base"]):.2f}  RoPE+fc {np.median(R["comb"]):.2f}  PaTH {np.median(P["base"]):.2f}  PaTH+fc {np.median(P["comb"]):.2f}  '
          f'| r-only RoPE {np.median(R["r"]):.2f} PaTH {np.median(P["r"]):.2f}')
    for name, a, b in [('PaTH+fc - PaTH   ', P['comb'], P['base']), ('PaTH+fc - RoPE+fc', P['comb'], R['comb']), ('PaTH - RoPE      ', P['base'], R['base']),
                       ('RoPE+fc - RoPE   ', R['comb'], R['base'])]:
        d, lo, hi, fr = boot(a, b)
        print(f'   {name} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] frac {fr:.3f}')
        if name.startswith('PaTH+fc - PaTH ') and H in (256, 512): k1.append(hi < 0)
        if name.startswith('PaTH+fc - RoPE+fc'): k2.append(hi < 0)
print(f'K1 (FreshCast helps PaTH at L256 & L512): {all(k1)} {k1}')
print(f'K2 (PaTH+fc < RoPE+fc at >= 2/3 L): {sum(k2) >= 2} {k2}')
