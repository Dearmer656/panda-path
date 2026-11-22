"""PAT-334 horizon-segment selection: the H-step future is split into NB = 4 equal segments; each segment takes its own
retrieval config (family, mode, k, weights, head mixing) chosen on val by that segment's error, and the forecast is the
concatenation of the per-segment predictions. Settings and head families as mh.py (G1), plus family 'all' (the 14 heads of
scale + seg16 + seg32 pooled). Head mixing per query: uniform ('uni', static) or dynamic by each head's current match quality
('best' = min dn; softmax(-dn / tau)), dn = head top-1 distance / its median candidate distance.
Errors per trajectory are means over queries, so a stitched forecast's per-trajectory error is the mean of its segments'
per-trajectory errors (exact; no re-run needed).
Arms (test): M2 = G1 headline (one config for the whole horizon, chosen on val full-horizon sMAPE, as mh.py)
             P1 = per-segment selection among static configs (tau = 'uni' or single-head families)
             P2 = per-segment selection among all configs (static + dynamic head mixing)
Criteria (fixed before results):
  Q1 segment selection : P2 - M2 sMAPE CI < 0 at all H, in both settings.
  reported             : P2 - P1 (dynamic contribution), P2 - parroting (ctx2048: parrot_ctx paper version; ctx512:
                         parrot_full), MSE for all, chosen config per segment.
usage: same env as mh.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import mh as M
from mh import G, FC, dev, chunks, backbone, TAG, MODE, head, futs, KS, TAUS
from mse_eval import per_traj  # noqa: E402
O = ROOT / 'pat334' / 'out'; HS = (128, 256, 512); NB = 4
FAM = dict(M.FAM); FAM['all'] = M.FAM['scale'] + M.FAM['seg16'] + M.FAM['seg32']
M2F = ('seg16', 'seg32', 'scale')


@torch.no_grad()
def run_set(name, H):
    d = torch.load(FC.OUT / f'{name}.pt'); S = d['S'].to(dev); fall = backbone(name, d); E = {}; hb = H // NB
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        y = Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)]
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)

        def put(k, p):
            s = (p - y).abs().div(p.abs() + y.abs() + 1e-10).reshape(B, NB, hb).mean(-1); m = (p - y).square().reshape(B, NB, hb).mean(-1)
            E.setdefault(k, ([], []))[0].append(s.cpu()); E[k][1].append(m.cpu())
        put('f', f)
        for mode in ('c', 'j'):
            for fam, hs in FAM.items():
                HD = []
                for (l, o) in hs:
                    js, vs, dn = head(Z, t, H, l, o, fam == 'sharedlog', mode, loc, rep, ni)
                    val = torch.isfinite(vs); fu = futs(Z, js, H); v1 = vs[:, :1]
                    ww = torch.exp(-(vs - v1) / v1.abs().clamp_min(1e-8)).masked_fill(~val, 0.0)
                    HD.append((fu, val, ww, torch.where(val[:, 0], dn, torch.full_like(dn, float('inf')))))
                for wn in (('u',) if fam == 'sharedlog' else ('u', 'w')):
                    for k in KS:
                        ph, has = [], []
                        for fu, val, ww, dn in HD:
                            w = val[:, :k].float() if wn == 'u' else ww[:, :k]
                            ph.append((w[..., None] * fu[:, :k]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)); has.append(val[:, 0])
                        ph, has = torch.stack(ph, 1), torch.stack(has, 1); dns = torch.stack([h[3] for h in HD], 1); anyh = has.any(1)
                        for tau in (TAUS if len(HD) > 1 else ('uni',)):
                            if tau == 'uni': a = has.float()
                            elif tau == 'best': a = torch.nn.functional.one_hot(dns.argmin(1), len(HD)).float() * has.float()
                            else: a = torch.softmax((-dns / tau).masked_fill(~has, float('-inf')), 1).nan_to_num(0.0)
                            p = (a[..., None] * ph).sum(1) / a.sum(1, keepdim=True).clamp_min(1e-12)
                            put(f'{mode}_{fam}|{wn}{k}|{tau}', torch.where(anyh[:, None], p, f))
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    # per config: per-trajectory per-segment errors [n_traj, NB]
    return {k: {met: torch.stack([per_traj(torch.cat(v)[inv][:, b], d['ids'], d['inst']) for b in range(NB)], 1) for met, v in zip(('smape', 'mse'), sm)}
            for k, sm in E.items()}


def full(x): return {met: x[met].mean(1) for met in ('smape', 'mse')}


def run(H):
    V = run_set('val', H); T = run_set('test', H)
    keys = [k for k in V if k != 'f']
    vf = {k: 100 * float(V[k]['smape'].mean(1).median()) for k in keys}
    pickM = {}
    for mode in 'cj':
        fb = {fm: min((k for k in keys if k.startswith(f'{mode}_{fm}|')), key=vf.get) for fm in M2F}
        pickM[mode] = min(fb.values(), key=vf.get)
    hm = min('cj', key=lambda m: vf[pickM[m]]); kM2 = pickM[hm]
    static = [k for k in keys if k.endswith('|uni')]
    A = {'f': full(T['f']), 'M2': full(T[kM2])}; picks = {}
    for arm, pool in (('P1', static), ('P2', keys)):
        ch = [min(pool, key=lambda k: float(V[k]['smape'][:, b].median())) for b in range(NB)]; picks[arm] = ch
        A[arm] = {met: torch.stack([T[ch[b]][met][:, b] for b in range(NB)], 1).mean(1) for met in ('smape', 'mse')}
    P9 = ROOT / 'pat329' / 'out' / (f'mse_lq96_H{H}.pt' if MODE == 'full' else f'mse_ctx{G.CTX}_lq96_H{H}.pt')
    b = torch.load(P9, weights_only=False); A['parrot_ctx'] = b['parrot_ctx@LQ96']; A['parrot_full'] = b['parrot_full@LQ96']
    for k, v in A.items(): assert len(v['smape']) == len(A['f']['smape']), k
    print(f'== {TAG} seg H{H}: M2 = {kM2} (val {vf[kM2]:.2f})', flush=True)
    for arm, ch in picks.items(): print(f'   {arm} per segment: ' + ' | '.join(ch), flush=True)
    print('   test ' + '  '.join(f'{k} {100 * float(v["smape"].median()):.2f}/{float(v["mse"].median()):.3f}' for k, v in A.items()), flush=True)
    print('   test per-segment sMAPE  M2 ' + ' '.join(f'{100 * float(T[kM2]["smape"][:, b].median()):.2f}' for b in range(NB))
          + ' | P2 ' + ' '.join(f'{100 * float(T[picks["P2"][b]]["smape"][:, b].median()):.2f}' for b in range(NB)), flush=True)
    rng = np.random.default_rng(0)
    for x, ref in (('P2', 'M2'), ('P1', 'M2'), ('P2', 'P1'), ('P2', 'parrot_ctx'), ('P2', 'parrot_full'), ('M2', 'parrot_ctx'), ('P2', 'f')):
        o = [f'{met} %+.3f [%+.3f, %+.3f]' % G.C.boot(sc * A[x][met].numpy(), sc * A[ref][met].numpy(), rng)[:3] for met, sc in (('smape', 100), ('mse', 1))]
        print(f'   {x:3s} - {ref:11s} ' + ' | '.join(o), flush=True)
    torch.save({'A': A, 'picks': picks, 'M2': kM2}, O / f'seg_{TAG}_H{H}.pt')


if __name__ == '__main__':
    import os
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else HS): run(H)
