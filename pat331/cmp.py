"""PAT-331 pilot comparison (ctx 2048 origins, H128, test, per-trajectory median, paired bootstrap seed 0).
Arms: bb = truncated RoPE backbone (last 512 steps); mem (LB chosen on val MSE); mem_rand = same model, memory from another
query (fixed permutation); retr = ctx-restricted retrieval (fair2048 scorer, smap_0.01 as val-selected in PAT-330; fallback bb);
fuse = a bb + (1 - a) retr, a on a 0.1 grid chosen on val median MSE.
GO (fixed before results): mem beats bb AND fuse on sMAPE and MSE (95% CI excludes 0), and mem_rand loses >= half of the
sMAPE gain of mem over bb.
usage: P321_LQ=96 P321_CTX=2048 P330_BLOCKS=6 P330_HIST=ctx P321_OUT=pat330/fair2048 python pat331/cmp.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327')); sys.path.insert(0, str(ROOT / 'pat329')); sys.path.insert(0, str(ROOT / 'pat331'))
import numpy as np, torch
import fc_corr as C  # noqa: E402
from mse_eval import per_traj  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); H = 128; O = ROOT / 'pat331' / 'out'


@torch.no_grad()
def retrieval(name, bb_norm):
    fcm = C.load_fc(H, dev); d, se = FC.load_set(name, dev); out = []
    for i in range(0, len(d['rows']), 64):
        idx = torch.arange(i, min(i + 64, len(d['rows'])))
        rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
        p = (w[..., None] * R).sum(1) + C.smap_corr(w, qp, cp, R, [0.01])[0]
        out.append(torch.where(b['any'][:, None], p, bb_norm[idx].to(dev)).cpu())
    return torch.cat(out)


def norm(pred_raw, d):
    S, rows, t = d['S'], d['rows'], d['t']
    Wc = S[rows[:, None], t[:, None] - FC.CTX + torch.arange(FC.CTX)[None]]; cm, cs = Wc.mean(1), Wc.std(1, unbiased=False).clamp_min(1e-10)
    return (pred_raw - cm[:, None]) / cs[:, None], (S[rows[:, None], t[:, None] + torch.arange(H)[None]] - cm[:, None]) / cs[:, None]


def errs(p, y, d):
    return {'smape': per_traj(((p - y).abs() / (p.abs() + y.abs() + 1e-10)).mean(1), d['ids'], d['inst']),
            'mse': per_traj((p - y).square().mean(1), d['ids'], d['inst'])}


P = {lb: torch.load(O / f'pred_lb{lb}.pt') for lb in (6, 8)}
E = {}
for name in ('val', 'test'):
    d = torch.load(FC.OUT / f'{name}.pt'); bb, y = norm(P[6][name]['bb'], d)
    A = {'bb': bb, 'retr': retrieval(name, bb)}
    for lb in (6, 8):
        A[f'mem{lb}'] = norm(P[lb][name]['mem'], d)[0]; A[f'mem{lb}_rand'] = norm(P[lb][name]['mem_rand'], d)[0]
    for al in np.round(np.arange(0, 1.01, 0.1), 1): A[f'fuse{al}'] = al * A['bb'] + (1 - al) * A['retr']
    E[name] = {k: errs(v, y, d) for k, v in A.items()}
vm = {k: float(v['mse'].median()) for k, v in E['val'].items()}
lb = min((6, 8), key=lambda l: vm[f'mem{l}']); fa = min([k for k in vm if k.startswith('fuse')], key=vm.get)
print(f'val MSE: mem6 {vm["mem6"]:.4f} mem8 {vm["mem8"]:.4f} -> LB {lb}; fusion {fa} ({vm[fa]:.4f}); bb {vm["bb"]:.4f} retr {vm["retr"]:.4f}', flush=True)
T = E['test']; rng = np.random.default_rng(0)
for k in ['bb', 'retr', fa, f'mem{lb}', f'mem{lb}_rand', f'mem{14 - lb}']:
    print(f'   test {k:12s} sMAPE {100 * float(T[k]["smape"].median()):6.2f}  MSE {float(T[k]["mse"].median()):.4f}', flush=True)
for x, ref in [(f'mem{lb}', 'bb'), (f'mem{lb}', fa), (f'mem{lb}', 'retr'), (f'mem{lb}_rand', 'bb'), (f'mem{lb}', f'mem{lb}_rand')]:
    o = []
    for met, sc in (('smape', 100), ('mse', 1)):
        d0, lo, hi, _ = C.boot(sc * T[x][met].numpy(), sc * T[ref][met].numpy(), rng); o.append(f'{met} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
    print(f'   {x:12s} - {ref:12s} ' + ' | '.join(o), flush=True)
torch.save(E, O / 'cmp.pt')
