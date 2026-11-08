"""PAT-318: summarize official evaluate.py outputs like Panda Tables 8/9 (median [P25, P75] over all held-out test samples).
Usage: python -m pat318.agg pat318/eval/released [pat318/eval/other ...]
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = {'smape': {128: 27.6, 256: 36.7, 512: 46.3}, 'mae': {128: 0.35, 256: 0.49, 512: 0.65}}


def load(d, L, metric):
    df = pd.read_csv(Path(d) / f'metrics_pred{L}.csv')
    return np.concatenate([np.asarray(ast.literal_eval(x), dtype=float) for x in df[metric]])


def summary(d):
    out = {}
    for m in ('smape', 'mae', 'mse'):
        for L in (128, 256, 512):
            v = load(d, L, m)
            out[m, L] = (np.median(v), np.percentile(v, 25), np.percentile(v, 75), np.mean(v), len(v))
    return out


if __name__ == '__main__':
    for d in sys.argv[1:]:
        print(f'== {d}')
        s = summary(d)
        for (m, L), (med, p25, p75, mean, n) in s.items():
            ref = PAPER.get(m, {}).get(L)
            print(f'{m:6s} L{L:<4d} median {med:8.4f} [{p25:.4f}, {p75:.4f}] mean {mean:9.4f} n={n}' + (f' | paper median {ref}' if ref else ''))
