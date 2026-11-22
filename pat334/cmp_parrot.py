"""PAT-334 vs context parroting (Zhang & Gilpin 2505.11349; PAT-329 implementation, D chosen on val). Test, per-trajectory
medians, paired bootstrap seed 0. ctx512 full history: vs parrot_full (whole own history). ctx2048 fair: vs parrot_ctx
(paper version, inside ctx) and parrot_full (whole history; more information than our fair arms)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np, torch
import fc_corr as C
O, O9 = ROOT / 'pat334' / 'out', ROOT / 'pat329' / 'out'
for tag, pf in (('ctx512_full', 'mse_lq96_H{H}.pt'), ('ctx2048_ctx', 'mse_ctx2048_lq96_H{H}.pt')):
    for H in (128, 256, 512):
        b = torch.load(O9 / pf.format(H=H), weights_only=False); P = {'parrot_ctx': b['parrot_ctx@LQ96'], 'parrot_full': b['parrot_full@LQ96'], 'bb': b['panda@LQ96']}
        g = torch.load(O / f'mh_{tag}_H{H}.pt', weights_only=False); s1 = torch.load(O / f'mh_{tag}_s1g_a512_H{H}.pt', weights_only=False)
        A = {'M2': g['A'][f'{g["hm"]}_M2'], 'whole': g['A'][f'{g["hm"]}_whole'], 's1_a512': s1['A'][f'{s1["hm"]}_M2'], 'f': g['A']['f']}
        if tag == 'ctx512_full': A['rr'] = torch.load(O / f'rr_H{H}.pt', weights_only=False)['A']['c_rr']
        dd = float((A['f']['smape'] - P['bb']['smape']).abs().max()) if tag == 'ctx512_full' else float('nan')
        print(f'== {tag} H{H}  (f vs pat329 backbone max |diff| {dd:.2e})')
        print('   ' + '  '.join(f'{k} {100 * float(v["smape"].median()):.2f}/{float(v["mse"].median()):.3f}' for k, v in {**A, **P}.items()))
        rng = np.random.default_rng(0)
        refs = ('parrot_full',) if tag == 'ctx512_full' else ('parrot_ctx', 'parrot_full')
        for x in A:
            if x == 'f': continue
            for r in refs:
                o = [f'{met} %+.3f [%+.3f, %+.3f]' % C.boot(sc * A[x][met].numpy(), sc * P[r][met].numpy(), rng)[:3] for met, sc in (('smape', 100), ('mse', 1))]
                print(f'   {x:8s} - {r:11s} ' + ' | '.join(o))
