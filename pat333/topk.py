"""PAT-333 E5 joint-state top-k analog ensemble (same settings as sig.py). Analogs: greedy argmin of the joint distance J with
an exclusion radius R = 32 steps around each picked end (distinct analogs); prediction = mean of their copied futures
(uniform 'u') or weighted by exp(-(J - J_min) / J_min) ('w'); k in {1, 3, 5, 10, 20}. k and weighting chosen on val median MSE
-> 'ens'. Then the soft gate of sig.py with r = ens (kappa on val MSE) -> 'gate_ens'.
GO (fixed before results): ens beats f on sMAPE AND MSE (CI excludes 0) at all H; gate_ens reported against f, ens, joint.
usage: as sig.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import sig as G  # noqa: E402
from sig import FC, dev, D_FULL, chunks, copy, okmask, errs, cmp, backbone, O, TAG  # noqa: E402
KS, R = (1, 3, 5, 10, 20), 32


@torch.no_grad()
def run_set(name, H):
    d = torch.load(FC.OUT / f'{name}.pt'); S = d['S'].to(dev); fall = backbone(name, d)
    P = {}; Y, DN = [], []
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        Y.append(Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)].cpu())
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        nch = torch.zeros(ni, device=dev).index_add_(0, loc, torch.ones(B, device=dev))
        D = D_FULL[H]; s = torch.arange(D, T + 1, device=dev)
        q = Z[ar[:, None], t[:, None] + torch.arange(-D, 0, device=dev)]
        dist = (Z.unfold(1, D, 1) - q[:, None]).square().mean(-1); ok = okmask(s, t, D, H)
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)
        J = torch.zeros(ni, len(s), device=dev).index_add_(0, loc, dist.masked_fill(~ok, 0.0)).masked_fill(~ok[rep], float('inf'))
        picks, vals = [], []
        for _ in range(max(KS)):
            v, j = J.min(1); picks.append(j); vals.append(v)
            J = J.masked_fill((s[None] - s[j][:, None]).abs() <= R, float('inf'))
        js, vs = torch.stack(picks, 1), torch.stack(vals, 1)                                  # [ni, K]
        fut = torch.stack([copy(Z, s[js[:, k]][loc], H) for k in range(max(KS))], 1)         # [B, K, H]
        okk = torch.isfinite(vs)[loc]; jm = vs[:, :1]
        wexp = torch.exp(-(vs - jm) / jm.clamp_min(1e-8)).masked_fill(~torch.isfinite(vs), 0.0)[loc]
        DN.append(torch.where(okk[:, 0], (vs[:, 0] / nch)[loc], torch.full((B,), float('inf'), device=dev)).cpu())
        A = {'f': f}
        for k in KS:
            for wn, w in (('u', okk[:, :k].float()), ('w', wexp[:, :k])):
                p = (w[..., None] * fut[:, :k]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
                A[f'k{k}{wn}'] = torch.where(okk[:, :1], p, f)
        for m, p in A.items(): P.setdefault(m, []).append(p.cpu())
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    return d, {m: torch.cat(v)[inv] for m, v in P.items()}, torch.cat(Y)[inv], torch.cat(DN)[inv]


def run(H):
    dv, Pv, Yv, DNv = run_set('val', H); dt, Pt, Yt, DNt = run_set('test', H)
    vm = {m: float(errs(p, Yv, dv)['mse'].median()) for m, p in Pv.items() if m != 'f'}; best = min(vm, key=vm.get)
    Pv['ens'], Pt['ens'] = Pv[best], Pt[best]; Pv['joint'], Pt['joint'] = Pv['k1u'], Pt['k1u']
    g = list(np.logspace(-3, 1, 25))
    sc = []
    for kap in g:
        a = torch.exp(-DNv / kap)[:, None]; sc.append(float(errs(a * Pv['ens'] + (1 - a) * Pv['f'], Yv, dv)['mse'].median()))
    kap = g[int(np.argmin(sc))]; a = torch.exp(-DNt / kap)[:, None]; Pt['gate_ens'] = a * Pt['ens'] + (1 - a) * Pt['f']
    T = {m: errs(p, Yt, dt) for m, p in Pt.items()}
    print(f'== {TAG} H{H}: val-chosen ens = {best} (val MSE {vm[best]:.4f}); gate kappa {kap:.4g} (val {min(sc):.4f})', flush=True)
    for m in T: print(f'   test {m:9s} sMAPE {100 * float(T[m]["smape"].median()):6.2f}  MSE {float(T[m]["mse"].median()):.4f}', flush=True)
    rng = np.random.default_rng(0)
    for x, ref in (('ens', 'f'), ('ens', 'joint'), ('gate_ens', 'f'), ('gate_ens', 'ens'), ('gate_ens', 'joint')):
        print(f'   {x:9s} - {ref:9s} ' + cmp(T, x, ref, rng), flush=True)
    torch.save({'T': T, 'best': best, 'kappa': kap}, O / f'topk_{TAG}_H{H}.pt')


if __name__ == '__main__':
    for H in (128, 256, 512): run(H)
