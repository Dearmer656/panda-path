"""PAT-334: collect test medians of every arm/metric for ctx2048_ctx and ctx512_full (reads saved per-trajectory results)."""
import torch
from pathlib import Path
O = Path('pat334/out'); P9 = Path('pat329/out')
def med(x): return float(torch.as_tensor(x).median())
for tag, p9 in (('ctx2048_ctx', 'mse_ctx2048_lq96_H{}.pt'), ('ctx512_full', 'mse_lq96_H{}.pt')):
    for H in (128, 256, 512):
        r = {}
        mh = torch.load(O / f'mh_{tag}_H{H}.pt', weights_only=False); A = mh['A']; hm = mh['hm']
        b9 = torch.load(P9 / p9.format(H), weights_only=False)
        pt = {'f': A['f'], 'parrot_ctx': b9['parrot_ctx@LQ96'], 'parrot_full': b9['parrot_full@LQ96'], 'M2': A[f'{hm}_M2']}
        for nm, f in (('R', f're_{tag}_H{H}.pt'), ('Bq', f'blend_{tag}_H{H}.pt')):
            if (O / f).exists(): pt[nm] = torch.load(O / f, weights_only=False)['A'][nm]
        for k, v in pt.items(): r.setdefault(k, {}).update(smape=100 * med(v['smape']), mse=med(v['mse']))
        for sfx, lab in (('', ''), ('_fix', '_fix')):
            e = torch.load(O / f'ens_{tag}_H{H}{sfx}.pt', weights_only=False)['A']
            for k, v in e.items():
                kk = k + lab if k == 'ens' else k
                if lab and k != 'ens': continue
                r.setdefault(kk, {}).update(crps=med(v['crps']), spec=med(v['spec']), vpt=med(v['vpt']))
        print(f'## {tag} H{H} M2={mh["pick"][f"{hm}_M2"][0]}')
        for k, v in r.items(): print(f'{k:12s} ' + ' '.join(f'{m}={v[m]:.4f}' if m in v else f'{m}=-' for m in ('smape', 'mse', 'crps', 'spec', 'vpt')))
