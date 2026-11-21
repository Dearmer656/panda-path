"""PAT-334 multihead segment retrieval on the observed signal (dysts, frozen Panda, seed 99; errors in ctx-window-normalised
units; per-trajectory median; paired bootstrap seed 0). Same settings as pat333/sig.py (P333_HIST=full: ctx 512, whole own
history; P333_HIST=ctx: ctx 2048 fair, window and future inside [t - CTX, t), backbone = PaTH truncated to the last 512).

A head = (length l, offset o): its query is Z[t - o - l, t - o); a candidate window ending at u maps to the anchor a = u + o
(the time that plays the role of t); readout copies Z[a, a + H). Each head picks its own top-k anchors (greedy, exclusion
radius R = 32 around picked anchors), constraint a + H <= t (ctx mode also: window start >= t - CTX).
  M0 whole     : one head (96, 0)                                     (whole-window retrieval)
  M1 sharedlog : one shared anchor, score = mean_j log(d_j + 1e-4) over the 6 x 16 segments of the 96 window
  M2 families  : seg16 = 6 heads (16, 16 j), j = 0..5 ; seg32 = 3 heads (32, 32 j) ; scale = 5 heads (l, 0), l = 16..96
  head mixing  : uniform / best head (min dn) / softmax(-dn / tau), dn = head top-1 distance / its median candidate distance
  per-channel ('c') and joint ('j', one anchor per instance, distances summed over channels).
Every arm: k in {1, 3, 5, 10, 20}, weights u (uniform) / w (exp(-(v - v1) / v1)), mixing; all chosen on val median sMAPE.
M2 = best of seg16 / seg32 / scale on val; mode (c / j) for the headline = the one with lower val sMAPE of M2.
Criteria (fixed before results):
  G1 multihead : in BOTH settings, M2 beats M1 and M0 of the same mode on sMAPE at all H (CI < 0); MSE reported.
  Reported, no gate: ctx 512 M2 vs PAT-330 Bl / C (evo files); ctx 2048 M2 vs PAT-333 E5 ens and truncated PaTH.
usage: P321_LQ=96 P321_OUT=pat321/data P333_HIST=full python pat334/mh.py
       P321_LQ=96 P321_CTX=2048 P321_OUT=pat321/data_ctx2048_path P333_HIST=ctx python pat334/mh.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import sig as G  # noqa: E402
from sig import FC, dev, chunks, backbone, TAG, MODE  # noqa: E402
from mse_eval import per_traj  # noqa: E402
O = ROOT / 'pat334' / 'out'; HS = (128, 256, 512)
# stride ablation (defaults = original G1 run): anchors restricted to a = t - STRIDE m ('g') or a pseudo-random 1/STRIDE subset
# ('r'); AMIN = 1 additionally requires a full 512 window before the anchor (full: a >= 512; ctx: a - 512 >= t - CTX)
import os
STRIDE, SUB, AMIN = int(os.environ.get('P334_STRIDE', 1)), os.environ.get('P334_SUB', 'g'), os.environ.get('P334_AMIN', '0') == '1'
SFX = '' if (STRIDE, AMIN) == (1, False) else f'_s{STRIDE}{SUB}{"_a512" if AMIN else ""}'
KS, KM, R, EPS, CLIP = (1, 3, 5, 10, 20), 20, 32, 1e-4, 1000.0
TAUS = ('uni', 'best', 0.03, 0.1, 0.3, 1.0)
FAM = {'whole': [(96, 0)], 'sharedlog': [(96, 0)], 'seg16': [(16, 16 * j) for j in range(6)],
       'seg32': [(32, 32 * j) for j in range(3)], 'scale': [(l, 0) for l in (16, 32, 48, 64, 96)]}
M2F = ('seg16', 'seg32', 'scale')


def cand(Z, t, H, l, o, shared):
    B, T = Z.shape; ar = torch.arange(B, device=dev)
    a = torch.arange(l + o, T + 1, device=dev)
    q = Z[ar[:, None], t[:, None] - o - l + torch.arange(l, device=dev)]
    W = Z.unfold(1, l, 1)[:, :T - l - o + 1]
    if shared: dist = torch.log((W - q[:, None]).square().reshape(B, W.shape[1], 6, 16).mean(-1) + EPS).mean(-1)
    else: dist = (W - q[:, None]).square().mean(-1)
    ok = a[None] + H <= t[:, None]
    if STRIDE > 1:
        if SUB == 'g': ok &= (t[:, None] - a[None]) % STRIDE == 0
        else: ok &= ((a[None] * 1000003 + t[:, None] * 7919) % 104729) % STRIDE == 0
    if AMIN: ok &= (a[None] >= 512) if MODE == 'full' else (a[None] - 512 >= (t - G.CTX)[:, None])
    if MODE == 'ctx': ok &= (a - o - l)[None] >= (t - G.CTX)[:, None]
    return a, dist, ok


def nms(D, a):
    js, vs = [], []
    for _ in range(KM):
        v, j = D.min(1); js.append(a[j]); vs.append(v)
        D = D.masked_fill((a[None] - a[j][:, None]).abs() <= R, float('inf'))
    return torch.stack(js, 1), torch.stack(vs, 1)


def head(Z, t, H, l, o, shared, mode, loc, rep, ni):
    a, dist, ok = cand(Z, t, H, l, o, shared)
    if mode == 'j':
        okI = ok[rep]
        dist = torch.zeros(ni, len(a), device=dev).index_add_(0, loc, dist.masked_fill(~ok, 0.0)); ok = okI
    Dm = dist.masked_fill(~ok, float('inf')); js, vs = nms(Dm, a)
    med = torch.nanmedian(dist.masked_fill(~ok, float('nan')), 1).values
    dn = vs[:, 0] / med.abs().clamp_min(1e-12)
    if mode == 'j': js, vs, dn = js[loc], vs[loc], dn[loc]
    return js, vs, dn


def futs(Z, js, H):
    B, T = Z.shape
    idx = (js[..., None] + torch.arange(H, device=dev)).clamp(max=T - 1)
    return Z.gather(1, idx.reshape(B, -1)).reshape(B, KM, H).clamp(-CLIP, CLIP)


def qerr(p, y):
    return (p - y).abs().div(p.abs() + y.abs() + 1e-10).mean(1), (p - y).square().mean(1)


@torch.no_grad()
def run_set(name, H):
    d = torch.load(FC.OUT / f'{name}.pt'); S = d['S'].to(dev); fall = backbone(name, d)
    E = {}
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
            s, m = qerr(p, y); E.setdefault(k, ([], []))[0].append(s.cpu()); E[k][1].append(m.cpu())
        put('f', f)
        for mode in ('c', 'j'):
            for fam, hs in FAM.items():
                HD = []
                for (l, o) in hs:
                    js, vs, dn = head(Z, t, H, l, o, fam == 'sharedlog', mode, loc, rep, ni)
                    val = torch.isfinite(vs); fu = futs(Z, js, H)
                    v1 = vs[:, :1]
                    ww = torch.exp(-(vs - v1) / v1.abs().clamp_min(1e-8)).masked_fill(~val, 0.0)
                    HD.append((fu, val, ww, torch.where(val[:, 0], dn, torch.full_like(dn, float('inf')))))
                wts = ('u',) if fam == 'sharedlog' else ('u', 'w')
                for wn in wts:
                    for k in KS:
                        ph, has = [], []
                        for fu, val, ww, dn in HD:
                            w = val[:, :k].float() if wn == 'u' else ww[:, :k]
                            ph.append((w[..., None] * fu[:, :k]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)); has.append(val[:, 0])
                        ph, has = torch.stack(ph, 1), torch.stack(has, 1); dns = torch.stack([h[3] for h in HD], 1)
                        anyh = has.any(1)
                        for tau in (TAUS if len(HD) > 1 else ('uni',)):
                            if tau == 'uni': a = has.float()
                            elif tau == 'best': a = torch.nn.functional.one_hot(dns.argmin(1), len(HD)).float() * has.float()
                            else: a = torch.softmax((-dns / tau).masked_fill(~has, float('-inf')), 1).nan_to_num(0.0)
                            p = (a[..., None] * ph).sum(1) / a.sum(1, keepdim=True).clamp_min(1e-12)
                            put(f'{mode}_{fam}|{wn}{k}|{tau}', torch.where(anyh[:, None], p, f))
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    T = {k: {'smape': per_traj(torch.cat(s)[inv], d['ids'], d['inst']), 'mse': per_traj(torch.cat(m)[inv], d['ids'], d['inst'])}
         for k, (s, m) in E.items()}
    return T


def med(x): return 100 * float(x['smape'].median())


def run(H):
    V = run_set('val', H); T = run_set('test', H)
    A, pick = {'f': T['f']}, {}
    for mode in ('c', 'j'):
        for fam in FAM:
            cs = [k for k in V if k.startswith(f'{mode}_{fam}|')]; b = min(cs, key=lambda k: med(V[k]))
            A[f'{mode}_{fam}'] = T[b]; pick[f'{mode}_{fam}'] = (b, med(V[b]))
        b = min(M2F, key=lambda fm: pick[f'{mode}_{fm}'][1]); A[f'{mode}_M2'] = A[f'{mode}_{b}']; pick[f'{mode}_M2'] = pick[f'{mode}_{b}']
    hm = min('cj', key=lambda m: pick[f'{m}_M2'][1])
    print(f'== {TAG}{SFX} H{H}: headline mode = {hm}', flush=True)
    for k, (b, v) in pick.items(): print(f'   pick {k:13s} {b:28s} val sMAPE {v:6.2f}', flush=True)
    ext = {}
    if SFX: pass
    elif MODE == 'full' and G.CTX == 512:
        Pp = torch.load(ROOT / 'pat330' / 'out' / f'evo_pool_H{H}.pt', weights_only=False)['T']; Pb = torch.load(ROOT / 'pat330' / 'out' / f'evo_blk_H{H}.pt', weights_only=False)['T']
        ext = {'p330_A': Pp['A'], 'p330_Bl': Pp['Bl'], 'p330_C': Pb['C']}
    elif MODE == 'ctx':
        ext = {'p333_ens': torch.load(ROOT / 'pat333' / 'out' / f'topk_{TAG}_H{H}.pt', weights_only=False)['T']['ens']}
    for k, v in ext.items(): assert len(v['smape']) == len(A['f']['smape']), k
    A.update(ext)
    for k, v in A.items(): print(f'   test {k:13s} sMAPE {med(v):6.2f}  MSE {float(v["mse"].median()):.4f}', flush=True)
    rng = np.random.default_rng(0)
    pairs = [(f'{m}_M2', f'{m}_{r}') for m in (hm, 'cj'.replace(hm, '')) for r in ('sharedlog', 'whole')]
    pairs += [(f'{hm}_{fm}', f'{hm}_whole') for fm in M2F] + [(f'{hm}_M2', 'f')] + [(f'{hm}_M2', k) for k in ext]
    for x, ref in pairs:
        o = []
        for met, sc in (('smape', 100), ('mse', 1)):
            d0, lo, hi, _ = G.C.boot(sc * A[x][met].numpy(), sc * A[ref][met].numpy(), rng); o.append(f'{met} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        print(f'   {x:13s} - {ref:13s} ' + ' | '.join(o), flush=True)
    torch.save({'A': A, 'pick': pick, 'hm': hm}, O / f'mh_{TAG}{SFX}_H{H}.pt')


if __name__ == '__main__':
    O.mkdir(exist_ok=True)
    for H in HS: run(H)
