"""PAT-334 (a): dense signal retrieval + hidden-state rerank (ctx 512, whole own history, val + test, seed 99 backbone).
Stage 1: whole 96-step window, every anchor a in [512, t - H] (stride 1; a >= 512 so each anchor has its own 512 window),
  per-channel ('c') or joint ('j', per-channel median-normalised distances summed over channels) mean-square distance,
  greedy exclusion radius 32, pool of N = 50 anchors.
Stage 2: the frozen PaTH Panda encodes the 512 window ending at each pooled anchor; hidden key = last-layer last-patch state
  (cos: 1 - cos, eu: squared Euclid; 'j' = per-channel median-normalised, summed).
  score = log(ds / med ds) + lam * log(dh / med dh) over the pool, lam in {0, .25, .5, 1, 2, 4, inf (hidden only)};
  top k in {1, 3, 5, 10, 20}, weights u (uniform) / w (exp(-(s - s1))); readout copies Z[a, a + H).
Selection on val median sMAPE. rr = best lam > 0; base = lam = 0 (same pool, signal only).
Criteria (fixed before results):
  R1 rerank helps : rr - base sMAPE CI < 0 at all H, in the headline mode (lower val sMAPE of rr).
  reported        : rr - mh.py ctx512 M2 / whole (same mode, stride 1, a >= 96), MSE for all.
usage: P334_SHARD=val:0:1 | test:i:n | merge:nv:nt  P321_LQ=96 P321_OUT=pat321/data python pat334/rr.py"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in ('pat327', 'pat329', 'pat330', 'pat333'): sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(ROOT))
import numpy as np, torch
import torch.nn.functional as F
import fc_corr as C  # noqa: E402
from mse_eval import per_traj  # noqa: E402
from trunc import CK  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); assert FC.CTX == 512
HS, KS, N, R, L = (128, 256, 512), (1, 3, 5, 10, 20), 50, 32, 96
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, float('inf'))
O = ROOT / 'pat334' / 'out'


@torch.no_grad()
def encode(p, X, ends):
    """X [C, T]; ends [n] -> last-patch last-layer states [n, C, d]."""
    out = []
    for i in range(0, len(ends), 128):
        e = ends[i:i + 128]
        win = X[:, e[:, None] - 512 + torch.arange(512, device=dev)].permute(1, 2, 0)
        out.append(p.model.model(past_values=win).last_hidden_state[:, :, -1])
    return torch.cat(out)


def nms(D, a):
    """D [C, n] (inf = invalid) -> anchors [C, N], distances [C, N]."""
    js, vs = [], []
    for _ in range(N):
        v, j = D.min(1); js.append(a[j]); vs.append(v)
        D = D.masked_fill((a[None] - a[j][:, None]).abs() <= R, float('inf'))
    return torch.stack(js, 1), torch.stack(vs, 1)


def lnorm(x, val):
    """log(x / median over valid pool entries), per row; x [C, N]."""
    m = torch.nanmedian(x.masked_fill(~val, float('nan')), 1, keepdim=True).values.abs().clamp_min(1e-12)
    return torch.log(x.clamp_min(1e-12) / m)


@torch.no_grad()
def run_set(name, p, shard):
    d = torch.load(FC.OUT / f'{name}.pt'); S = d['S'].to(dev)
    E = {H: [] for H in HS}; KEYS = {}; order = []
    for i in range(shard[0], d['n_inst'], shard[1]):
        q = torch.nonzero(d['ids'] == i)[:, 0]; order.append(q)
        rows = d['rows'][q].to(dev); t = int(d['t'][q[0]]); X = S[rows]; nc = len(q); ar = torch.arange(nc, device=dev)
        w = X[:, t - 512:t]; cm, cs = w.mean(1), w.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        a = torch.arange(512, t + 1, device=dev)
        Wq = Z[:, t - L:t]; ds_all = (Z.unfold(1, L, 1)[:, a - L] - Wq[:, None]).square().mean(-1)          # [C, n]
        pools = {}
        for H in HS:
            ok = a + H <= t
            if not bool(ok.any()): continue
            dc = ds_all.masked_fill(~ok[None], float('inf'))
            pools[(H, 'c')] = nms(dc, a)
            med = ds_all[:, ok].median(1, keepdim=True).values.abs().clamp_min(1e-12)
            dj = (ds_all / med).masked_fill(~ok[None], float('inf')).sum(0, keepdim=True)
            js, vs = nms(dj, a); pools[(H, 'j')] = (js.expand(nc, -1), vs.expand(nc, -1))
        if pools:
            allj = torch.cat([v[0].reshape(-1) for v in pools.values()]); allj = allj[(allj >= 512) & (allj <= t)]
            U = torch.unique(allj); Hc = encode(p, X, U); Hq = encode(p, X, torch.tensor([t], device=dev))[0]   # [u, C, d], [C, d]
            hcos = 1 - F.cosine_similarity(Hc, Hq[None], dim=-1); heu = (Hc - Hq[None]).square().sum(-1)           # [u, C]
        for H in HS:
            f = (d['f'][q, :H].to(dev) - cm[:, None]) / cs[:, None]; y = Z[:, t:t + H]; loc = {}

            def put(k, pr):
                loc[k] = torch.stack([((pr - y).abs() / (pr.abs() + y.abs() + 1e-10)).mean(1), (pr - y).square().mean(1)])
            put('f', f)
            for mode in 'cj':
                if (H, mode) not in pools:
                    for hk in ('cos', 'eu'):
                        for lam in LAMS:
                            for wn in 'uw':
                                for k in KS: put(f'{mode}|{hk}|{lam}|{wn}{k}', f)
                    continue
                js, vs = pools[(H, mode)]; val = torch.isfinite(vs)
                jj = js.clamp(512, t - H); ui = torch.searchsorted(U, jj).clamp(max=len(U) - 1)                                         # [C, N]
                fu = Z[ar[:, None, None], jj[..., None] + torch.arange(H, device=dev)].clamp(-1000, 1000)           # [C, N, H]
                if mode == 'c': ls = lnorm(vs, val)
                else: ls = lnorm(vs[:1], val[:1]).expand(nc, -1)
                for hk, hm in (('cos', hcos), ('eu', heu)):
                    dh = hm[ui, ar[:, None]]                                                                         # [C, N]
                    if mode == 'c': lh = lnorm(dh, val)
                    else:
                        mh = torch.nanmedian(dh.masked_fill(~val, float('nan')), 1, keepdim=True).values.abs().clamp_min(1e-12)
                        sj = (dh / mh).sum(0, keepdim=True); lh = lnorm(sj, val[:1]).expand(nc, -1)
                    for lam in LAMS:
                        sc = lh if lam == float('inf') else ls + lam * lh
                        sc = sc.masked_fill(~val, float('inf')); ss, o = sc.sort(1); fo = fu[ar[:, None], o]; vo = torch.isfinite(ss)
                        for wn in 'uw':
                            ww = vo.float() if wn == 'u' else torch.exp(-(ss - ss[:, :1])).masked_fill(~vo, 0.0)
                            for k in KS:
                                wk = ww[:, :k]; pr = (wk[..., None] * fo[:, :k]).sum(1) / wk.sum(1, keepdim=True).clamp_min(1e-12)
                                put(f'{mode}|{hk}|{lam}|{wn}{k}', pr)
            ks = list(loc); assert KEYS.setdefault(H, ks) == ks
            E[H].append(torch.stack([loc[k] for k in ks], 0).cpu())                                                 # [K, 2, nc]
    return {'order': torch.cat(order).cpu(), 'E': {H: {k: (x[j, 0], x[j, 1]) for j, k in enumerate(KEYS[H])}
                                                   for H in HS for x in [torch.cat(E[H], 2)]}}


def merged(name, n):
    d = torch.load(FC.OUT / f'{name}.pt'); Rr = [torch.load(O / f'rr_raw_{name}_{i}of{n}.pt') for i in range(n)]
    o = torch.cat([r['order'] for r in Rr]); inv = torch.empty_like(o); inv[o] = torch.arange(len(o))
    return {H: {k: {'smape': per_traj(torch.cat([r['E'][H][k][0] for r in Rr])[inv], d['ids'], d['inst']),
                    'mse': per_traj(torch.cat([r['E'][H][k][1] for r in Rr])[inv], d['ids'], d['inst'])}
                for k in Rr[0]['E'][H]} for H in HS}


def med(x): return 100 * float(x['smape'].median())


def report(V, T):
    for H in HS:
        A, pick = {'f': T[H]['f']}, {}
        for mode in 'cj':
            for arm, sel in (('base', lambda k: k.split('|')[2] == '0.0'), ('rr', lambda k: k.split('|')[2] != '0.0')):
                cs = [k for k in V[H] if k.startswith(f'{mode}|') and sel(k)]; b = min(cs, key=lambda k: med(V[H][k]))
                A[f'{mode}_{arm}'] = T[H][b]; pick[f'{mode}_{arm}'] = (b, med(V[H][b]))
        hm = min('cj', key=lambda m: pick[f'{m}_rr'][1])
        M = torch.load(O / f'mh_ctx512_full_H{H}.pt', weights_only=False)['A']
        assert float((M['f']['smape'] - A['f']['smape']).abs().max()) < 1e-2
        for m in 'cj': A[f'{m}_M2'] = M[f'{m}_M2']; A[f'{m}_whole'] = M[f'{m}_whole']
        print(f'== H{H}: headline mode = {hm}', flush=True)
        for k, (b, v) in pick.items(): print(f'   pick {k:8s} {b:22s} val sMAPE {v:6.2f}', flush=True)
        for k, v in A.items(): print(f'   test {k:8s} sMAPE {med(v):6.2f}  MSE {float(v["mse"].median()):.4f}', flush=True)
        rng = np.random.default_rng(0)
        pairs = [(f'{m}_rr', f'{m}_{r}') for m in (hm, 'cj'.replace(hm, '')) for r in ('base', 'M2', 'whole')] + [(f'{hm}_rr', 'f')]
        for x, ref in pairs:
            o2 = []
            for met, sc in (('smape', 100), ('mse', 1)):
                d0, lo, hi, _ = C.boot(sc * A[x][met].numpy(), sc * A[ref][met].numpy(), rng); o2.append(f'{met} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
            print(f'   {x:8s} - {ref:8s} ' + ' | '.join(o2), flush=True)
        torch.save({'A': A, 'pick': pick, 'hm': hm}, O / f'rr_H{H}.pt')


def main():
    O.mkdir(exist_ok=True); sh = os.environ['P334_SHARD']
    if sh.startswith('merge'):
        _, nv, nt = sh.split(':'); report(merged('val', int(nv)), merged('test', int(nt))); return
    name, i, n = sh.split(':'); p = FC.load_backbone(ROOT / CK['PaTH'], dev)
    torch.save(run_set(name, p, (int(i), int(n))), O / f'rr_raw_{name}_{i}of{n}.pt'); print('shard done', sh, flush=True)


if __name__ == '__main__':
    main()
