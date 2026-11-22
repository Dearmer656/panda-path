"""PAT-334 B: re-anchored retrieval. The H-step future is produced S steps at a time; before each segment the query window
(last l steps of [real history up to t] + [forecast copied so far]) is matched again against the real history with the same
retrieval config as the setting's G1 headline M2 (key read from out/mh_{TAG}_H{H}.pt), and the next S real steps after the
match are copied. Candidates: readout fully real (a + S <= t); ctx mode also window start >= t - CTX (as mh.py).
Continuity across segment joins is part of the match cost (the new analog's past must match the copied segment).
  R1 greedy : W = 1, rank-0 analogs at every segment
  R2 beam   : W = 5; alternatives at each segment = ranks r..r+k-1 (r < W) of every head; path score = sum over segments of the
              head-mixing-weighted mean log dn (dn = rank-r distance / median candidate distance); best path kept
Grid (selected once on val by median sMAPE): S in {32, 64} x {R1, R2}. Sanity arm S = H, W = 1 must reproduce M2.
Criteria (fixed before results), headline setting ctx2048:
  B1 : R - M2         sMAPE CI < 0 at all H
  B2 : R - parrot_ctx sMAPE CI < 0 and MSE CI < 0 at all H
  reported: ctx512 full, R - parrot_full, R - f, R1 / R2 arms, chosen S.
usage: same env as mh.py + P334_H"""
import os, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import mh as M
from mh import G, FC, dev, chunks, backbone, TAG, MODE
from mse_eval import per_traj  # noqa: E402
O = ROOT / 'pat334' / 'out'; HS = (128, 256, 512); R_EX, CLIP, LQ = 32, 1000.0, 96


def parse(key):
    mf, wn_k, tau = key.split('|'); mode, fam = mf.split('_', 1)
    return mode, fam, wn_k[0], int(wn_k[1:]), (tau if tau in ('uni', 'best') else float(tau))


def nms(D, a, n):
    js, vs = [], []
    for _ in range(n):
        v, j = D.min(1); js.append(a[j]); vs.append(v)
        D = D.masked_fill((a[None] - a[j][:, None]).abs() <= R_EX, float('inf'))
    return torch.stack(js, 1), torch.stack(vs, 1)


def head(Z, Wall, seq, t, S, l, o, mode, loc, rep, ni, n):
    """Wall = Z.unfold(1, l, 1); seq [B, L] = query sequence ending at the current segment start."""
    B, T = Z.shape; L = seq.shape[1]
    a = torch.arange(l + o, T + 1, device=dev); W = Wall[:, :T - l - o + 1]
    q = seq[:, L - o - l:L - o]
    dist = (W - q[:, None]).square().mean(-1)
    ok = a[None] + S <= t[:, None]
    if MODE == 'ctx': ok &= (a - o - l)[None] >= (t - G.CTX)[:, None]
    if mode == 'j':
        okI = ok[rep]; dist = torch.zeros(ni, len(a), device=dev).index_add_(0, loc, dist.masked_fill(~ok, 0.0)); ok = okI
    js, vs = nms(dist.masked_fill(~ok, float('inf')), a, n)
    med = torch.nanmedian(dist.masked_fill(~ok, float('nan')), 1).values.abs().clamp_min(1e-12)
    if mode == 'j': js, vs, med = js[loc], vs[loc], med[loc]
    return js, vs, med


@torch.no_grad()
def forecast(Z, t, f, H, S, Wb, cfg, loc, rep, ni, Walls):
    mode, fam, wn, k, tau = cfg; hs = M.FAM[fam]; B, T = Z.shape; ar = torch.arange(B, device=dev)
    hist = Z[ar[:, None], t[:, None] - LQ + torch.arange(LQ, device=dev)]
    P = torch.zeros(B, 1, 0, device=dev); cum = torch.zeros(B, 1, device=dev)
    for s in range(math.ceil(H / S)):
        cP, cS = [], []
        for b in range(P.shape[1]):
            seq = torch.cat([hist, P[:, b]], 1); HD = []
            for hi, (l, o) in enumerate(hs):
                js, vs, med = head(Z, Walls[hi], seq, t, S, l, o, mode, loc, rep, ni, k + Wb - 1); HD.append((js, vs, med))
            for r in range(Wb):
                ph, has, dns = [], [], []
                for js, vs, med in HD:
                    jr, vr = js[:, r:r + k], vs[:, r:r + k]; val = torch.isfinite(vr)
                    idx = (jr[..., None] + torch.arange(S, device=dev)).clamp(max=T - 1)
                    fu = Z.gather(1, idx.reshape(B, -1)).reshape(B, k, S).clamp(-CLIP, CLIP)
                    w = val.float() if wn == 'u' else torch.exp(-(vr - vr[:, :1]) / vr[:, :1].abs().clamp_min(1e-8)).masked_fill(~val, 0.0)
                    ph.append((w[..., None] * fu).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)); has.append(val[:, 0])
                    dns.append(torch.where(val[:, 0], vr[:, 0] / med, torch.full_like(med, float('inf'))))
                ph, has, dns = torch.stack(ph, 1), torch.stack(has, 1), torch.stack(dns, 1); anyh = has.any(1)
                if tau == 'uni': a = has.float()
                elif tau == 'best': a = torch.nn.functional.one_hot(dns.argmin(1), len(hs)).float() * has.float()
                else: a = torch.softmax((-dns / tau).masked_fill(~has, float('-inf')), 1).nan_to_num(0.0)
                pr = (a[..., None] * ph).sum(1) / a.sum(1, keepdim=True).clamp_min(1e-12)
                pr = torch.where(anyh[:, None], pr, f[:, s * S:s * S + S] if s * S + S <= H else torch.cat([f[:, s * S:], f[:, -1:].expand(B, s * S + S - H)], 1))
                lg = torch.log(dns.clamp_min(1e-12)).masked_fill(~has, 0.0)
                sc = torch.where(anyh, (a * lg).sum(1) / a.sum(1).clamp_min(1e-12), torch.zeros_like(cum[:, 0]))
                cP.append(torch.cat([P[:, b], pr], 1)); cS.append(cum[:, b] + sc)
        cP, cS = torch.stack(cP, 1), torch.stack(cS, 1)
        if mode == 'j':                                                       # keep beams shared by all channels of an instance
            cS = torch.zeros(ni, cS.shape[1], device=dev).index_add_(0, loc, cS)[loc]
        keep = cS.topk(min(Wb, cS.shape[1]), 1, largest=False).indices
        P = cP[ar[:, None], keep]; cum = cS.gather(1, keep)
    return P[:, 0, :H]


@torch.no_grad()
def run_set(name, H, cfg, grid):
    d = torch.load(FC.OUT / f'{name}.pt'); S_ = d['S'].to(dev); fall = backbone(name, d); E = {g: ([], []) for g in grid}
    hs = M.FAM[cfg[1]]
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S_[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        y = Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)]
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)
        Walls = [Z.unfold(1, l, 1) for (l, o) in hs]
        for g in grid:
            p = forecast(Z, t, f, H, g[0], g[1], cfg, loc, rep, ni, Walls)
            E[g][0].append(((p - y).abs() / (p.abs() + y.abs() + 1e-10)).mean(1).cpu()); E[g][1].append((p - y).square().mean(1).cpu())
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    return {g: {'smape': per_traj(torch.cat(s)[inv], d['ids'], d['inst']), 'mse': per_traj(torch.cat(m)[inv], d['ids'], d['inst'])}
            for g, (s, m) in E.items()}


def med(x): return 100 * float(x['smape'].median())


def run(H):
    pk = torch.load(O / f'mh_{TAG}_H{H}.pt', weights_only=False); hm = pk['hm']; key = pk['pick'][f'{hm}_M2'][0]; cfg = parse(key)
    assert cfg[1] != 'sharedlog'
    grid = [(32, 1), (64, 1), (32, 5), (64, 5), (H, 1)]
    V = run_set('val', H, cfg, grid); T = run_set('test', H, cfg, grid)
    Mh = pk['A']; dM = float((Mh[f'{hm}_M2']['smape'] - T[(H, 1)]['smape']).abs().max())
    print(f'== {TAG} reanchor H{H}: M2 = {key}; sanity |S=H,W=1 - M2| max per-traj sMAPE diff {dM:.2e}', flush=True)
    assert dM < 1e-2, 'S = H, W = 1 does not reproduce M2'
    sel = [g for g in grid if g[0] < H]
    for g in grid: print(f'   S{g[0]:<4d} W{g[1]}  val {med(V[g]):6.2f}  test {med(T[g]):6.2f} / {float(T[g]["mse"].median()):.4f}', flush=True)
    gR = min(sel, key=lambda g: med(V[g])); gR1 = min([g for g in sel if g[1] == 1], key=lambda g: med(V[g])); gR2 = min([g for g in sel if g[1] > 1], key=lambda g: med(V[g]))
    A = {'f': Mh['f'], 'M2': Mh[f'{hm}_M2'], 'R': T[gR], 'R1': T[gR1], 'R2': T[gR2]}
    P9 = ROOT / 'pat329' / 'out' / (f'mse_lq96_H{H}.pt' if MODE == 'full' else f'mse_ctx{G.CTX}_lq96_H{H}.pt')
    b9 = torch.load(P9, weights_only=False); A['parrot_ctx'] = b9['parrot_ctx@LQ96']; A['parrot_full'] = b9['parrot_full@LQ96']
    for kk, v in A.items(): assert len(v['smape']) == len(A['f']['smape']), kk
    print(f'   chosen R = S{gR[0]} W{gR[1]} | R1 = S{gR1[0]} | R2 = S{gR2[0]}', flush=True)
    print('   test ' + '  '.join(f'{kk} {med(v):.2f}/{float(v["mse"].median()):.3f}' for kk, v in A.items()), flush=True)
    rng = np.random.default_rng(0)
    for x, rf in (('R', 'M2'), ('R', 'parrot_ctx'), ('R', 'parrot_full'), ('R', 'f'), ('R1', 'M2'), ('R2', 'M2'), ('R2', 'R1')):
        o = [f'{met} %+.3f [%+.3f, %+.3f]' % G.C.boot(sc * A[x][met].numpy(), sc * A[rf][met].numpy(), rng)[:3] for met, sc in (('smape', 100), ('mse', 1))]
        print(f'   {x:3s} - {rf:11s} ' + ' | '.join(o), flush=True)
    torch.save({'A': A, 'grid': {str(g): T[g] for g in grid}, 'val': {str(g): med(V[g]) for g in grid}, 'R': gR, 'M2': key}, O / f're_{TAG}_H{H}.pt')


if __name__ == '__main__':
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else HS): run(H)
