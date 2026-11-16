"""PAT-318: paired comparison of temporal position arms (PatchTST-univariate, seed 99). Medians over the 9347 official
test trajectories, fraction where arm < ref, bootstrap 95% CI of median(arm) - median(ref).
usage: python pat318/cmp_pos.py ARM1,ARM2,... REF1,REF2 [metrics]"""
import sys; sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from pat318.agg import load
arms = sys.argv[1].split(','); refs = sys.argv[2].split(','); mets = sys.argv[3].split(',') if len(sys.argv) > 3 else ['smape', 'mse']
rng = np.random.default_rng(0)
for ctx in (512, 1024, 2048):
    for m in mets:
        V = {a: {L: load(f'pat318/eval/p318ut_{a}_s99_st100000_ls99_ctx{ctx}', L, m) for L in (128, 256, 512)}
             for a in arms if Path(f'pat318/eval/p318ut_{a}_s99_st100000_ls99_ctx{ctx}').exists()}
        for L in (128, 256, 512):
            row = f'ctx{ctx} {m} L{L}: ' + ' '.join(f'{a} {np.median(V[a][L]):.3f}' for a in V)
            for ref in refs:
                r = V[ref][L]; I = [rng.integers(0, len(r), len(r)) for _ in range(300)]
                for a in V:
                    if a == ref: continue
                    x = V[a][L]; bs = [np.median(x[i]) - np.median(r[i]) for i in I]
                    row += f' | {a}<{ref} {np.mean(x < r):.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]'
            print(row, flush=True)
