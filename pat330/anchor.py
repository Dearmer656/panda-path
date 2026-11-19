"""PAT-330 step 2: the user's multi-anchor design (ctx 512, LQ 96; values context-standardised, raw level kept).
Query window [t-96, t) = 6 segments L_1..L_6 of 16 steps. Anchor i: offset o_i = 16 (6 - i) (steps from the end of L_i to t).
  retrieval   top-K positions s by d_i(s) = mean (Z[s-16:s] - L_i)^2, with s >= 16 and s + o_i + H <= t (no leakage)
  validation  e(s) = mean (Z[s:s+o_i] - Z[t-o_i:t])^2 for i < 6 (later segments); anchor 6 has none -> e = d_6
  reference   Z[s + o_i : s + o_i + H] (aligned to t)
  score       b_i - e / tau ; w = softmax over the pooled 6K candidates; b_1..b_6, log tau (7 parameters)
Fit on train queries with the FreshCast objective (clipped normalised MSE of the weighted mean, eps_q as PAT-321),
early stop on val loss. Outputs: weighted mean (anc_mean), top-1 (anc_top1), S-map on the pooled candidates (anc_smap_l,
lambda on val). Per-trajectory val / test sMAPE and MSE saved for step-1-style selection.
usage: P321_LQ=96 P321_OUT=pat321/data python pat330/anchor.py --H 512"""
import argparse, math, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--H', type=int, default=512); ap.add_argument('--K', type=int, default=50)
ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327')); sys.path.insert(0, str(ROOT / 'pat329'))
import numpy as np, torch
import torch.nn as nn
import fc_corr as C  # noqa: E402
from mse_eval import per_traj  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); H, K, SEG, NA = a.H, a.K, 16, 6
assert FC.LQ == 96 and FC.CTX == 512
OFF = [SEG * (NA - i) for i in range(1, NA + 1)]                    # o_1..o_6 = 80, 64, 48, 32, 16, 0
torch.manual_seed(99)


@torch.no_grad()
def cands(se, rows, t, cm, cs):
    """pooled candidates: anchor id [B, 6K], validation error e [B, 6K], aligned position p = s + o_i [B, 6K], valid."""
    Z = (se.S[rows] - cm[:, None]) / cs[:, None]                                   # [B, T]
    B, T = Z.shape; ar = torch.arange(B, device=dev)
    U = Z.unfold(1, SEG, 1)                                                         # [B, T-15, 16], window ending at s = j + 16
    s_all = torch.arange(SEG, T + 1, device=dev)
    A, E, P, V = [], [], [], []
    for i, o in enumerate(OFF):
        st = t - 96 + SEG * i
        q = Z[ar[:, None], st[:, None] + torch.arange(SEG, device=dev)]           # L_{i+1}
        d = (U - q[:, None]).square().mean(-1)
        ok = s_all[None] + o + H <= t[:, None]
        d = d.masked_fill(~ok, math.inf)
        ok &= s_all[None] + o >= 96                                                 # aligned 96-window exists (S-map coords)
        d = d.masked_fill(~ok, math.inf)
        dk, jk = d.topk(K, 1, largest=False); s = s_all[jk]; v = torch.isfinite(dk)
        s = torch.where(v, s, SEG)
        if o > 0:
            cv = Z[ar[:, None, None], s[..., None] + torch.arange(o, device=dev)]
            qv = Z[ar[:, None], t[:, None] - o + torch.arange(o, device=dev)]
            e = (cv - qv[:, None]).square().mean(-1)
        else:
            e = dk
        A.append(torch.full_like(s, i)); E.append(torch.where(v, e, 0.)); P.append(s + o); V.append(v)
    return Z, torch.cat(A, 1), torch.cat(E, 1), torch.cat(P, 1), torch.cat(V, 1)


class AnchorScore(nn.Module):
    def __init__(self):
        super().__init__(); self.b = nn.Parameter(torch.zeros(NA)); self.lt = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(self, A, E, V):
        sc = self.b[A] - E / self.lt.exp()
        return torch.softmax(sc.masked_fill(~V, -math.inf), -1)


def batch(se, d, idx):
    rows, t = d['rows'][idx].to(dev), d['t'][idx].to(dev)
    X = se.S[rows]; ar = torch.arange(len(rows), device=dev)
    win = X[ar[:, None], t[:, None] - FC.CTX + torch.arange(FC.CTX, device=dev)]
    cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10)
    Z, A, E, P, V = cands(se, rows, t, cm, cs)
    R = Z[ar[:, None, None], P[..., None] + torch.arange(H, device=dev)]          # references [B, 6K, H]
    y = Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)]
    f = (d['f'][idx, :H].to(dev) - cm[:, None]) / cs[:, None] if 'f' in d else None
    return dict(rows=rows, t=t, Z=Z, A=A, E=E, P=P, V=V, R=R.clamp(-FC.CLIP, FC.CLIP), y=y, f=f, has=V.any(1))


def eps_of(bt):
    """eps_q: MSE of the untrained rule (mean of the 20 best anchor-6 candidates = 16-step 1-NN copies)."""
    un = bt['R'][:, (NA - 1) * K:(NA - 1) * K + 20].mean(1)
    return (un - bt['y']).square().mean(1).clamp_min(1e-3)


def loss_of(model, bt):
    w = model(bt['A'], bt['E'], bt['V']); r = (w[..., None] * bt['R']).sum(1)
    l = ((r - bt['y']).square().mean(1) / eps_of(bt)).clamp_max(FC.C_MAX)
    return l[bt['has']].mean()


def check():
    d, se = FC.load_set('val', dev)
    bt = batch(se, d, torch.arange(0, 256))
    v = bt['V']; P = bt['P']; t = bt['t'][:, None].expand_as(P)
    assert ((P + H <= t) | ~v).all(), 'leakage'
    i = int(v[:, 0].nonzero()[0]); j = 0; s = int(P[i, j]) - OFF[0]; tt = int(bt['t'][i]); Z = bt['Z'][i]
    e = float((Z[s:s + OFF[0]] - Z[tt - OFF[0]:tt]).square().mean())
    assert abs(e - float(bt['E'][i, j])) < 1e-4 * (1 + e), (e, float(bt['E'][i, j]))
    print(f'check ok (leakage, validation error), valid fraction {float(v.float().mean()):.3f}', flush=True)


def fit():
    d, se = FC.load_set('train', dev); dv, sv = FC.load_set('val', dev)
    sel = torch.randperm(len(d['rows']), generator=torch.Generator().manual_seed(99))[:32000]
    model = AnchorScore().to(dev); opt = torch.optim.Adam(model.parameters(), lr=0.02)
    best = (math.inf, None)
    for ep in range(a.epochs):
        t0 = time.time(); tot = 0.0; n = 0
        for i in range(0, len(sel), 256):
            bt = batch(se, d, sel[i:i + 256])
            if not bt['has'].any(): continue
            l = loss_of(model, bt); opt.zero_grad(); l.backward(); opt.step(); tot += float(l); n += 1
        with torch.no_grad():
            vl = [float(loss_of(model, batch(sv, dv, torch.arange(i, min(i + 256, len(dv['rows'])))))) for i in range(0, len(dv['rows']), 256)]
        vl = float(np.nanmean(vl))
        print(f'[anchor] H{H} ep {ep} train {tot / n:.4f} val {vl:.4f} b {[round(x, 2) for x in model.b.tolist()]} tau {model.lt.exp().item():.4f} {time.time() - t0:.0f}s', flush=True)
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1]); return model


@torch.no_grad()
def predict(model, name):
    d, se = FC.load_set(name, dev); acc = {}
    for i in range(0, len(d['rows']), 64):
        bt = batch(se, d, torch.arange(i, min(i + 64, len(d['rows']))))
        w = model(bt['A'], bt['E'], bt['V']); w = torch.nan_to_num(w, nan=0.0); R = bt['R']
        r0 = (w[..., None] * R).sum(1)
        Pm = {'anc_mean': r0, 'anc_top1': R.gather(1, w.argmax(1)[:, None, None].expand(-1, 1, H))[:, 0]}
        ar = torch.arange(len(R), device=dev)
        qp = bt['Z'][ar[:, None], bt['t'][:, None] + torch.arange(-96, 0, device=dev)]
        cp = bt['Z'][ar[:, None, None], bt['P'][..., None] + torch.arange(-96, 0, device=dev)]
        for l, c in zip(C.LAMS, C.smap_corr(w, qp.clamp(-FC.CLIP, FC.CLIP), cp.clamp(-FC.CLIP, FC.CLIP), R, C.LAMS)):
            Pm[f'anc_smap_{l}'] = r0 + c
        for m, p in Pm.items():
            p = torch.where(bt['has'][:, None], p, bt['f']); x = acc.setdefault(m, dict(smape=[], mse=[]))
            x['smape'].append(((p - bt['y']).abs() / (p.abs() + bt['y'].abs() + 1e-10)).mean(1).cpu())
            x['mse'].append((p - bt['y']).square().mean(1).cpu())
    return {m: {k: per_traj(torch.cat(v), d['ids'], d['inst']) for k, v in x.items()} for m, x in acc.items()}


if __name__ == '__main__':
    if a.check: check(); sys.exit()
    model = fit()
    out = {'state': model.state_dict(), 'val': predict(model, 'val'), 'test': predict(model, 'test')}
    torch.save(out, ROOT / 'pat330' / 'out' / f'anchor_H{H}.pt')
    for s in ('val', 'test'):
        print(f'[anchor] H{H} {s} ' + ' '.join(f'{m} {100 * float(v["smape"].median()):.2f}/{float(v["mse"].median()):.3f}' for m, v in out[s].items()), flush=True)
