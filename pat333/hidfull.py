"""PAT-333 E1' clean thesis test (ctx 512, whole history, test). sig.py E1 used stored patch states whose context differs from the
query's (mid-window patches); here every candidate end e gets the SAME treatment as the query: the frozen PaTH Panda encodes
the 512-step window ending at e, key = last-patch state ('last') or mean over patches ('mean', what the head pools), full
512-d, cosine or Euclid. Candidates: e = t - 16k, e >= 512, e + H <= t; readout = copy Z[e : e + H] for every key.
sig = Euclid on the signal window (D = D_FULL[H]) on the same candidates. Per-channel and joint (summed over channels).
Thesis supported (fixed before results) if sig beats every hidden key on sMAPE at all H (CI < 0), per-channel and joint.
usage: P321_LQ=96 P321_OUT=pat321/data python pat333/hidfull.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in ('pat327', 'pat329', 'pat330'): sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(ROOT))
import numpy as np, torch
import torch.nn.functional as F
import fc_corr as C  # noqa: E402
from mse_eval import per_traj, D_FULL  # noqa: E402
from trunc import CK  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); assert FC.CTX == 512
HS = (128, 256, 512); KEYS = ['sig', 'last_cos', 'last_eu', 'mean_cos', 'mean_eu']


@torch.no_grad()
def encode(p, X, ends):
    """X [C, T] raw; ends [n] -> last-patch and mean states [n, C, 512]."""
    L, M = [], []
    for i in range(0, len(ends), 128):
        e = ends[i:i + 128]
        win = X[:, e[:, None] - 512 + torch.arange(512, device=dev)].permute(1, 2, 0)       # [b, 512, C]
        h = p.model.model(past_values=win).last_hidden_state                               # [b, C, NPW, d]
        L.append(h[:, :, -1]); M.append(h.mean(2))
    return torch.cat(L), torch.cat(M)


@torch.no_grad()
def main():
    d = torch.load(FC.OUT / 'test.pt'); S = d['S'].to(dev)
    p = FC.load_backbone(ROOT / CK['PaTH'], dev)
    E = {H: {f'{k}{j}': dict(smape=[], mse=[]) for k in KEYS for j in ('', '_joint')} for H in HS}; order = []
    for i in range(d['n_inst']):
        q = torch.nonzero(d['ids'] == i)[:, 0]; order.append(q)
        rows = d['rows'][q].to(dev); t = int(d['t'][q[0]]); X = S[rows]; nc = len(q)
        w = X[:, t - 512:t]; cm, cs = w.mean(1), w.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        ends = t - 16 * torch.arange(8, (t - 512) // 16 + 1, device=dev)                       # e <= t - 128, e >= 512
        Lc, Mc = encode(p, X, ends); Lq, Mq = encode(p, X, torch.tensor([t], device=dev))
        dk = {'last_cos': -F.cosine_similarity(Lc, Lq, dim=-1), 'last_eu': (Lc - Lq).square().sum(-1),
              'mean_cos': -F.cosine_similarity(Mc, Mq, dim=-1), 'mean_eu': (Mc - Mq).square().sum(-1)}        # [n, C]
        for H in HS:
            D = D_FULL[H]; ok = ends + H <= t
            y = Z[:, t:t + H]
            qs = Z[:, t - D:t]; Wc = Z[:, ends[:, None] - D + torch.arange(D, device=dev)]                 # [C, n, D]
            dd = dict(dk, sig=(Wc - qs[:, None]).square().mean(-1).T)
            for k in KEYS:
                v = dd[k].masked_fill(~ok[:, None], float('inf'))                                          # [n, C]
                for j, sel in (('', v.argmin(0)), ('_joint', None)):
                    if sel is None:
                        vj = (v / v[ok].abs().mean(0, keepdim=True).clamp_min(1e-12)).sum(1) if k != 'sig' else v.sum(1)
                        sel = vj.argmin().expand(nc)
                    pr = torch.stack([Z[c, ends[sel[c]]:ends[sel[c]] + H] for c in range(nc)])
                    E[H][k + j]['smape'].append(((pr - y).abs() / (pr.abs() + y.abs() + 1e-10)).mean(1).cpu())
                    E[H][k + j]['mse'].append((pr - y).square().mean(1).cpu())
    o = torch.cat(order); inv = torch.empty_like(o); inv[o] = torch.arange(len(o))
    for H in HS:
        T = {m: {k: per_traj(torch.cat(v)[inv], d['ids'], d['inst']) for k, v in e.items()} for m, e in E[H].items()}
        print(f'== H{H}', flush=True)
        for m in T: print(f'   test {m:15s} sMAPE {100 * float(T[m]["smape"].median()):6.2f}  MSE {float(T[m]["mse"].median()):.4f}', flush=True)
        rng = np.random.default_rng(0)
        for j in ('', '_joint'):
            for k in KEYS[1:]:
                o2 = []
                for met, sc in (('smape', 100), ('mse', 1)):
                    d0, lo, hi, _ = C.boot(sc * T['sig' + j][met].numpy(), sc * T[k + j][met].numpy(), rng); o2.append(f'{met} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
                print(f'   sig{j:6s} - {k + j:15s} ' + ' | '.join(o2), flush=True)
        torch.save(T, ROOT / 'pat333' / 'out' / f'hidfull_H{H}.pt')


if __name__ == '__main__':
    main()
