"""PAT-334 D: analog ensemble as a probabilistic forecast (training-free). Samples = the futures of the top-k analogs of every
head of the setting's M2 config (key from out/mh_{TAG}_H{H}.pt; NMS radius 32; candidates a + H <= t, ctx mode window start
>= t - CTX); sample weight = M2 head-mixing weight x within-head weight (u / w as M2). k in {5, 10, 20} chosen once on val CRPS.
Arms (test, values in context-standardised units):
  ens    analog ensemble (k chosen on val)
  clim   control: 100 equally weighted futures at uniformly random valid anchors (same constraints, no matching)
  f      Panda backbone (deterministic: CRPS = MAE)
  chr    Chronos-T5-base, 20 samples, last 512 context steps (pat334/chronos_run.py; test only)
Metrics per query, then per-trajectory means (per_traj):
  crps   weighted sample CRPS averaged over the H steps (sorted-sample identity)
  spec   power-spectrum Hellinger distance between each sample and the truth (DC removed, normalised), sample-weighted mean
  vpt    steps until the instance's channel-mean per-step sMAPE (Gilpin scale, 200 |p - y| / (|p| + |y|)) of the weighted-mean
         forecast first exceeds 30 (H if never)
Criteria (fixed before results), headline ctx2048:
  D1 : ens - chr  crps CI < 0 at all H        D2 : ens - f  crps CI < 0 at all H
  D3 : ens - f and ens - chr  spec CI < 0 at all H
  reported: clim, vpt, ctx512 full.
usage: same env as mh.py + P334_H"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import mh as M
from mh import G, FC, dev, chunks, backbone, TAG, MODE, head, futs, cand
from mse_eval import per_traj  # noqa: E402
O = ROOT / 'pat334' / 'out'; HS = (128, 256, 512); KK = (5, 10, 20); NCL = 100


def parse(key):
    mf, wn_k, tau = key.split('|'); mode, fam = mf.split('_', 1)
    return mode, fam, wn_k[0], int(wn_k[1:]), (tau if tau in ('uni', 'best') else float(tau))


def crps(x, w, y):
    """x [B, n, H] samples, w [B, n] weights (sum 1), y [B, H] -> [B] mean CRPS over H."""
    xs, o = x.sort(1); ws = w[:, :, None].expand_as(x).gather(1, o); F = ws.cumsum(1)
    t1 = (ws * (xs - y[:, None]).abs()).sum(1)
    t2 = (ws * xs * (2 * F - ws - 1)).sum(1)                       # = 0.5 sum_ij w_i w_j |x_i - x_j|
    return (t1 - t2).mean(1)


def spec(x, y):
    """x [B, n, H], y [B, H] -> Hellinger distance between power spectra [B, n]."""
    def ps(z):
        p = torch.fft.rfft(z - z.mean(-1, keepdim=True), dim=-1).abs().square()[..., 1:]
        s = p.sum(-1, keepdim=True); return torch.where(s > 1e-12, p / s.clamp_min(1e-12), torch.full_like(p, 1.0 / p.shape[-1]))
    return (1 - (ps(x) * ps(y)[:, None]).sqrt().sum(-1)).clamp_min(0).sqrt()


def vpt(p, y, loc, ni):
    e = 200 * (p - y).abs() / (p.abs() + y.abs() + 1e-10)                                        # [B, H]
    cnt = torch.zeros(ni, device=dev).index_add_(0, loc, torch.ones(len(loc), device=dev))
    ei = torch.zeros(ni, e.shape[1], device=dev).index_add_(0, loc, e) / cnt[:, None]
    bad = ei > 30; first = torch.where(bad.any(1), bad.float().argmax(1), torch.full((ni,), e.shape[1], device=dev, dtype=torch.long))
    return first.float()[loc]


@torch.no_grad()
def run_set(name, H, cfg, chr_s=None):
    mode, fam, wn, _, tau = cfg; hs = M.FAM[fam]
    d = torch.load(FC.OUT / f'{name}.pt'); S = d['S'].to(dev); fall = backbone(name, d); E = {}
    gen = torch.Generator(device=dev).manual_seed(0)
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        y = Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)]
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)

        def put(k, x, w):                                                                         # x [B, n, H], w [B, n]
            pm = (w[..., None] * x).sum(1)
            vals = (crps(x, w, y), (spec(x, y) * w).sum(1), vpt(pm, y, loc, ni))
            for nm, v in zip(('crps', 'spec', 'vpt'), vals): E.setdefault(k, {}).setdefault(nm, []).append(v.cpu())
        one = torch.ones(B, 1, device=dev)
        put('f', f[:, None], one)
        HD = []
        for (l, o) in hs:
            js, vs, dn = head(Z, t, H, l, o, False, mode, loc, rep, ni)
            val = torch.isfinite(vs); fu = futs(Z, js, H); v1 = vs[:, :1]
            ww = val.float() if wn == 'u' else torch.exp(-(vs - v1) / v1.abs().clamp_min(1e-8)).masked_fill(~val, 0.0)
            HD.append((fu, val, ww, torch.where(val[:, 0], dn, torch.full_like(dn, float('inf')))))
        has = torch.stack([h[1][:, 0] for h in HD], 1); dns = torch.stack([h[3] for h in HD], 1); anyh = has.any(1)
        if tau == 'uni': a = has.float()
        elif tau == 'best': a = torch.nn.functional.one_hot(dns.argmin(1), len(HD)).float() * has.float()
        else: a = torch.softmax((-dns / tau).masked_fill(~has, float('-inf')), 1).nan_to_num(0.0)
        a = a / a.sum(1, keepdim=True).clamp_min(1e-12)
        for k in KK:
            xs = torch.cat([h[0][:, :k] for h in HD], 1)
            ws = torch.cat([a[:, i:i + 1] * h[2][:, :k] / h[2][:, :k].sum(1, keepdim=True).clamp_min(1e-12) for i, h in enumerate(HD)], 1)
            xs = torch.where(anyh[:, None, None], xs, f[:, None]); ws = torch.where(anyh[:, None], ws, torch.full_like(ws, 1.0 / ws.shape[1]))
            put(f'ens{k}', xs, ws / ws.sum(1, keepdim=True))
        # climatology control: random valid anchors of the longest head
        l, o = max(hs); aa, _, ok = cand(Z, t, H, l, o, False)
        okf = ok.float(); anyc = okf.sum(1) > 0
        pick = torch.multinomial(torch.where(anyc[:, None], okf, torch.ones_like(okf)), NCL, replacement=True, generator=gen)
        xc = futs_n(Z, aa[pick], H); xc = torch.where(anyc[:, None, None], xc, f[:, None])
        put('clim', xc, torch.full((B, NCL), 1.0 / NCL, device=dev))
        if chr_s is not None:
            xr = (chr_s[idx].to(dev).float()[..., :H] - cm[:, None, None]) / cs[:, None, None]
            put('chr', xr.clamp(-M.CLIP, M.CLIP), torch.full((B, xr.shape[1]), 1.0 / xr.shape[1], device=dev))
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    return {k: {nm: per_traj(torch.cat(v)[inv], d['ids'], d['inst']) for nm, v in m.items()} for k, m in E.items()}


def futs_n(Z, js, H):
    B, T = Z.shape; n = js.shape[1]
    idx = (js[..., None] + torch.arange(H, device=dev)).clamp(max=T - 1)
    return Z.gather(1, idx.reshape(B, -1)).reshape(B, n, H).clamp(-M.CLIP, M.CLIP)


def run(H):
    pk = torch.load(O / f'mh_{TAG}_H{H}.pt', weights_only=False); hm = pk['hm']; key = pk['pick'][f'{hm}_M2'][0]; cfg = parse(key)
    assert cfg[1] != 'sharedlog'
    cf = O / f'chronos_{TAG}_test.pt'; chr_s = torch.load(cf)['samples'] if cf.exists() else None
    V = run_set('val', H, cfg); T = run_set('test', H, cfg, chr_s)
    kb = min(KK, key=lambda k: float(V[f'ens{k}']['crps'].median()))
    A = {'ens': T[f'ens{kb}'], 'clim': T['clim'], 'f': T['f']}
    if 'chr' in T: A['chr'] = T['chr']
    print(f'== {TAG} ens H{H}: M2 = {key}; k = {kb} (val crps ' + ' '.join(f'{k}:{float(V[f"ens{k}"]["crps"].median()):.4f}' for k in KK) + ')', flush=True)
    for kk, v in A.items(): print(f'   test {kk:5s} crps {float(v["crps"].median()):.4f}  spec {float(v["spec"].median()):.4f}  vpt {float(v["vpt"].median()):6.1f}', flush=True)
    rng = np.random.default_rng(0); refs = [r for r in ('chr', 'f', 'clim') if r in A]
    for r in refs:
        o = [f'{nm} %+.4f [%+.4f, %+.4f]' % G.C.boot(A['ens'][nm].numpy(), A[r][nm].numpy(), rng)[:3] for nm in ('crps', 'spec', 'vpt')]
        print(f'   ens - {r:5s} ' + ' | '.join(o), flush=True)
    torch.save({'A': A, 'k': kb, 'M2': key}, O / f'ens_{TAG}_H{H}.pt')


if __name__ == '__main__':
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else HS): run(H)
