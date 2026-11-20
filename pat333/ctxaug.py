"""PAT-333 E4 retrieval-augmented context for the frozen Panda PaTH backbone (ctx 512 data, whole history, test, H 128 / 256).
The joint-state analog s (one per instance, D = D_FULL[H], channels summed, s + H <= t) is found in signal space; the
backbone then sees context = [S[s + H - 256 : s + H] (analog window + its continuation) ; S[t - 256 : t]] (512 steps).
aug_rand: same layout, analog time drawn uniformly from the valid positions (seed 0). f = the official 512-step forecast.
GO (fixed before results): aug beats f AND joint on sMAPE and MSE (CI excludes 0) and aug beats aug_rand on sMAPE.
usage: P321_LQ=96 P321_OUT=pat321/data python pat333/ctxaug.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in ('pat327', 'pat329', 'pat330'): sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(ROOT))
import numpy as np, torch
import fc_corr as C  # noqa: E402
from mse_eval import per_traj, D_FULL  # noqa: E402
from trunc import CK  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); assert FC.CTX == 512; HALF = 256


@torch.no_grad()
def joint_s(S, rows, t, H, ids):
    """per query: joint analog end s (same for all channels of an instance) and a random valid s."""
    X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev); D = D_FULL[H]
    win = X[ar[:, None], t[:, None] - 512 + torch.arange(512, device=dev)]
    cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
    s = torch.arange(D, T + 1, device=dev); q = Z[ar[:, None], t[:, None] + torch.arange(-D, 0, device=dev)]
    dist = (Z.unfold(1, D, 1) - q[:, None]).square().mean(-1)
    ok = (s[None] + H <= t[:, None]) & (s[None] + H - HALF >= 0) & (s[None] >= D)
    loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
    J = torch.zeros(ni, len(s), device=dev).index_add_(0, loc, dist.masked_fill(~ok, 0.0))
    rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar); okr = ok[rep]
    sj = s[J.masked_fill(~okr, float('inf')).argmin(1)]
    g = torch.Generator(device='cpu').manual_seed(0)
    sr = torch.stack([s[torch.nonzero(okr[i])[:, 0]][torch.randint(int(okr[i].sum()), (1,), generator=g).item()] for i in range(ni)])
    return sj[loc], sr[loc], Z, cm, cs


@torch.no_grad()
def main():
    d = torch.load(FC.OUT / 'test.pt'); S = d['S'].to(dev)
    p = FC.load_backbone(ROOT / CK['PaTH'], dev)
    for H in (128, 256):
        E = {m: dict(smape=[], mse=[]) for m in ('f', 'joint', 'aug', 'aug_rand')}; order = []
        for i in range(d['n_inst']):
            q = torch.nonzero(d['ids'] == i)[:, 0]; order.append(q)
            rows, t, ids = d['rows'][q].to(dev), d['t'][q].to(dev), d['ids'][q].to(dev)
            sj, sr, Z, cm, cs = joint_s(S, rows, t, H, ids); tt = int(t[0])
            y = Z[:, tt:tt + H]
            P = {'f': (d['f'][q, :H].to(dev) - cm[:, None]) / cs[:, None], 'joint': Z[:, int(sj[0]):int(sj[0]) + H]}
            ctxs = []
            for sa in (int(sj[0]), int(sr[0])):
                ctxs.append(torch.cat([S[rows, sa + H - HALF:sa + H], S[rows, tt - HALF:tt]], 1).T)
            pr = p.predict(context=torch.stack(ctxs), prediction_length=H, limit_prediction_length=False, sliding_context=True, verbose=False)
            pr = pr.reshape(2, H, len(q)).transpose(1, 2).float()
            P['aug'], P['aug_rand'] = (pr[0] - cm[:, None]) / cs[:, None], (pr[1] - cm[:, None]) / cs[:, None]
            for m, v in P.items():
                E[m]['smape'].append(((v - y).abs() / (v.abs() + y.abs() + 1e-10)).mean(1).cpu()); E[m]['mse'].append((v - y).square().mean(1).cpu())
        o = torch.cat(order); inv = torch.empty_like(o); inv[o] = torch.arange(len(o))
        T = {m: {k: per_traj(torch.cat(v)[inv], d['ids'], d['inst']) for k, v in e.items()} for m, e in E.items()}
        print(f'== H{H}', flush=True)
        for m in T: print(f'   test {m:9s} sMAPE {100 * float(T[m]["smape"].median()):6.2f}  MSE {float(T[m]["mse"].median()):.4f}', flush=True)
        rng = np.random.default_rng(0)
        for x, ref in (('aug', 'f'), ('aug', 'joint'), ('aug', 'aug_rand')):
            o2 = []
            for met, sc in (('smape', 100), ('mse', 1)):
                d0, lo, hi, _ = C.boot(sc * T[x][met].numpy(), sc * T[ref][met].numpy(), rng); o2.append(f'{met} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
            print(f'   {x:9s} - {ref:9s} ' + ' | '.join(o2), flush=True)
        torch.save(T, ROOT / 'pat333' / 'out' / f'ctxaug_H{H}.pt')


if __name__ == '__main__':
    main()
