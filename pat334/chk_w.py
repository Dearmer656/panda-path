"""PAT-334 D check: why val crps is identical for k = 5/10/20 at ctx2048 H128. Records weight concentration per k."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ens as E
E.KK = (1, 5, 10, 20); R = {}
_c = E.crps
def crps(x, w, y):
    n = x.shape[1]
    if n in (3 * 1, 3 * 5, 3 * 10, 3 * 20, 1, 5, 10, 20) and n != 100:
        R.setdefault(n, []).append(torch.stack([w.max(1).values, 1 / w.square().sum(1)], 1).cpu())
    return _c(x, w, y)
E.crps = crps
pk = torch.load(E.O / f'mh_{E.TAG}_H{os.environ["P334_H"]}.pt', weights_only=False); key = pk['pick'][f'{pk["hm"]}_M2'][0]; cfg = E.parse(key)
print('M2', key, 'heads', E.M.FAM[cfg[1]], flush=True)
for name in ('val', 'test'):
    R.clear(); V = E.run_set(name, int(os.environ['P334_H']), cfg)
    for k in E.KK:
        c = V[f'ens{k}']['crps']; print(f'{name} k{k:2d} crps med {float(c.median()):.6f} mean {float(c.mean()):.6f}', flush=True)
    for n, v in sorted(R.items()):
        v = torch.cat(v); q = torch.tensor([.1, .5, .9])
        print(f'{name} n={n:3d} max-w q10/50/90 {v[:,0].quantile(q).tolist()} | eff-n q10/50/90 {v[:,1].quantile(q).tolist()} | frac max-w>0.99 {float((v[:,0]>0.99).float().mean()):.3f}', flush=True)
