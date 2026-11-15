import sys; sys.path.insert(0,'.')
import numpy as np
from pat318.agg import load
arms=['rope','path','rope_path']; rng=np.random.default_rng(0)
for ctx in (512,1024,2048):
    for m in ('mse',):
        V={a:{L:load(f'pat318/eval/p318ut_{a}_s99_st100000_ls99_ctx{ctx}',L,m) for L in (128,256,512)} for a in arms}
        for L in (128,256,512):
            r=V['rope'][L]; row=f'ctx{ctx} {m} L{L}: '+' '.join(f'{a} {np.median(V[a][L]):.3f}' for a in arms)
            for a in ('path','rope_path'):
                x=V[a][L]; assert len(x)==len(r)
                d=x-r; bs=[np.median(x[i])-np.median(r[i]) for i in [rng.integers(0,len(r),len(r)) for _ in range(300)]]
                row+=f' | {a}<rope {np.mean(x<r):.3f} dmed CI[{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]'
            print(row, f'n={len(r)}')
