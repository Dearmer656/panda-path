"""PAT-334 backbone + dense retrieval blend. Retrieval r = the G1 headline M2 config of mh.py (same setting, stride 1, its val-chosen
key read from out/mh_{TAG}_H{H}.pt); backbone f = seed-99 PaTH Panda. Forecast p = alpha * f + (1 - alpha) * r, the H-step future
split into NB = 4 equal segments. alpha (backbone weight):
  Bg : one constant per H, grid 0, .05, ..., 1
  Bs : one constant per horizon segment (same grid)
  Bq : per horizon segment, gated by match quality, alpha = sigmoid(a0 + b * log dn), dn = best head's top-1 distance / its median
       candidate distance (min over the M2 heads), a0 in -4..4 step .5, b in -3..3 step .5 (+ the Bs constants)
Selection on val by J = med sMAPE / med sMAPE_f + med MSE / med MSE_f (both metrics count equally, relative to the backbone).
Stitched per-trajectory error = mean of its segments' per-trajectory errors (exact; per_traj is linear, segments equal length).
Criteria (fixed before results), headline arm Bq, headline setting ctx2048 (P333_HIST=ctx):
  C1 : Bq - f          sMAPE CI < 0 and MSE CI < 0 at all H
  C2 : Bq - parrot_ctx sMAPE CI < 0 and MSE CI < 0 at all H
  reported: Bg / Bs, Bq - M2, Bq - parrot_full, ctx512 full setting, chosen alpha / gate per segment.
usage: same env as mh.py (+ P334_H)"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import mh as M
from mh import G, FC, dev, chunks, backbone, TAG, MODE, head, futs
from mse_eval import per_traj  # noqa: E402
O = ROOT / 'pat334' / 'out'; HS = (128, 256, 512); NB = 4
AL = [i / 20 for i in range(21)]
GT = [(a0, b) for a0 in np.arange(-4, 4.01, .5) for b in np.arange(-3, 3.01, .5)]
CFG = [('c', x) for x in AL] + [('g', x) for x in GT]


def parse(key):
    mf, wn_k, tau = key.split('|'); mode, fam = mf.split('_', 1)
    return mode, fam, wn_k[0], int(wn_k[1:]), (tau if tau in ('uni', 'best') else float(tau))


@torch.no_grad()
def run_set(name, H, key):
    mode, fam, wn, k, tau = parse(key); hs = M.FAM[fam]
    d = torch.load(FC.OUT / f'{name}.pt'); S = d['S'].to(dev); fall = backbone(name, d); hb = H // NB; Es, Em = [], []
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        y = Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)]
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)
        ph, has, dns = [], [], []
        for (l, o) in hs:
            js, vs, dn = head(Z, t, H, l, o, fam == 'sharedlog', mode, loc, rep, ni)
            val = torch.isfinite(vs); fu = futs(Z, js, H); v1 = vs[:, :1]
            w = val[:, :k].float() if wn == 'u' else torch.exp(-(vs - v1) / v1.abs().clamp_min(1e-8)).masked_fill(~val, 0.0)[:, :k]
            ph.append((w[..., None] * fu[:, :k]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)); has.append(val[:, 0])
            dns.append(torch.where(val[:, 0], dn, torch.full_like(dn, float('inf'))))
        ph, has, dns = torch.stack(ph, 1), torch.stack(has, 1), torch.stack(dns, 1); anyh = has.any(1)
        if tau == 'uni': a = has.float()
        elif tau == 'best': a = torch.nn.functional.one_hot(dns.argmin(1), len(hs)).float() * has.float()
        else: a = torch.softmax((-dns / tau).masked_fill(~has, float('-inf')), 1).nan_to_num(0.0)
        r = torch.where(anyh[:, None], (a[..., None] * ph).sum(1) / a.sum(1, keepdim=True).clamp_min(1e-12), f)
        g = torch.log(dns.min(1).values.clamp_min(1e-8)); g = torch.where(torch.isfinite(g), g, torch.zeros_like(g)).clamp(-20, 5)
        alpha = torch.stack([torch.full_like(g, x) if c == 'c' else torch.sigmoid(x[0] + x[1] * g) for c, x in CFG], 1)   # [B, C]
        s_, m_ = [], []
        for i in range(0, len(CFG), 32):
            al = alpha[:, i:i + 32, None]; p = al * f[:, None] + (1 - al) * r[:, None]; yy = y[:, None]
            s_.append((p - yy).abs().div(p.abs() + yy.abs() + 1e-10).reshape(B, -1, NB, hb).mean(-1)); m_.append((p - yy).square().reshape(B, -1, NB, hb).mean(-1))
        Es.append(torch.cat(s_, 1).cpu()); Em.append(torch.cat(m_, 1).cpu())
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    out = {}
    for met, E in (('smape', Es), ('mse', Em)):
        x = torch.cat(E)[inv]                                                                                      # [n, C, NB]
        out[met] = torch.stack([torch.stack([per_traj(x[:, c, b], d['ids'], d['inst']) for b in range(NB)], 1) for c in range(len(CFG))], 1)  # [ntraj, C, NB]
    return out


def run(H):
    pk = torch.load(O / f'mh_{TAG}_H{H}.pt', weights_only=False); key = pk['pick'][f'{pk["hm"]}_M2'][0]
    V = run_set('val', H, key); T = run_set('test', H, key)
    i0, i1 = CFG.index(('c', 1.0)), CFG.index(('c', 0.0))                                                       # f, r
    ref = {met: float(V[met][:, i0].mean(1).median()) for met in ('smape', 'mse')}
    def J(x): return sum(float(x[met].median()) / ref[met] for met in ('smape', 'mse'))        # x[met] [ntraj]
    def pickseg(pool, b): return min(pool, key=lambda c: J({met: V[met][:, c, b] for met in V}))
    def stitch(D, ch): return {met: torch.stack([D[met][:, ch[b], b] for b in range(NB)], 1).mean(1) for met in D}
    cidx = [i for i, (c, _) in enumerate(CFG) if c == 'c']; allidx = list(range(len(CFG)))
    picks = {'f': [i0] * NB, 'r': [i1] * NB}
    cg = min(cidx, key=lambda c: J({met: V[met][:, c].mean(1) for met in V})); picks['Bg'] = [cg] * NB
    picks['Bs'] = [pickseg(cidx, b) for b in range(NB)]; picks['Bq'] = [pickseg(allidx, b) for b in range(NB)]
    A = {arm: stitch(T, ch) for arm, ch in picks.items()}
    P9 = ROOT / 'pat329' / 'out' / (f'mse_lq96_H{H}.pt' if MODE == 'full' else f'mse_ctx{G.CTX}_lq96_H{H}.pt')
    b9 = torch.load(P9, weights_only=False); A['parrot_ctx'] = b9['parrot_ctx@LQ96']; A['parrot_full'] = b9['parrot_full@LQ96']
    Mh = torch.load(O / f'mh_{TAG}_H{H}.pt', weights_only=False)['A']
    assert float((Mh[f'{pk["hm"]}_M2']['smape'] - A['r']['smape']).abs().max()) < 1e-2, 'M2 reconstruction mismatch'
    for kk, v in A.items(): assert len(v['smape']) == len(A['f']['smape']), kk
    print(f'== {TAG} blend H{H}: M2 = {key}', flush=True)
    for arm in ('Bg', 'Bs', 'Bq'): print(f'   {arm} per segment: ' + ' | '.join(str(CFG[c]) for c in picks[arm]), flush=True)
    print('   test ' + '  '.join(f'{kk} {100 * float(v["smape"].median()):.2f}/{float(v["mse"].median()):.3f}' for kk, v in A.items()), flush=True)
    rng = np.random.default_rng(0)
    for x, rf in (('Bq', 'f'), ('Bq', 'parrot_ctx'), ('Bq', 'r'), ('Bq', 'parrot_full'), ('Bg', 'f'), ('Bs', 'f'), ('Bq', 'Bs'), ('r', 'f')):
        o = [f'{met} %+.3f [%+.3f, %+.3f]' % G.C.boot(sc * A[x][met].numpy(), sc * A[rf][met].numpy(), rng)[:3] for met, sc in (('smape', 100), ('mse', 1))]
        print(f'   {x:3s} - {rf:11s} ' + ' | '.join(o), flush=True)
    torch.save({'A': A, 'picks': {a: [CFG[c] for c in ch] for a, ch in picks.items()}, 'M2': key}, O / f'blend_{TAG}_H{H}.pt')


if __name__ == '__main__':
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else HS): run(H)
