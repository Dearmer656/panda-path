"""PAT-332 Wasserstein retrieval vs Euclid (ctx 512, LQ 96, whole own history, test, top-1 copy; fallback = backbone f).
Hypothesis (user): a segment's energy state drives the next segment; analogs need not match point by point, a shifted match
may be the most similar. Z = series standardised by the ctx window; windows of 96 ending at s, s >= 96, s + H <= t.
  eu_parrot  Euclid, D = D_FULL[H] (parrot-full, reference)        eu96  Euclid, D = 96
  wval       W1 between the value distributions of the two windows (sorted values, mean |diff|): order-free energy state
  wtime      W1 over time between the normalised energy profiles Z^2 / sum Z^2 (in steps / 96) + |log E_c / E_q|
             (E = mean Z^2; the second term keeps the total energy, the first tolerates time shifts); raw copy from s
  wtime_al   wtime candidate, readout shifted by the energy-centroid offset (candidate's pattern later by d -> copy from s + d)
  eu_wval / eu_wtime  Euclid D = 96 top-200 re-ranked by wval / wtime
Criterion (fixed before results): a W arm supports the hypothesis at H if its paired median sMAPE difference vs eu_parrot has
95% CI upper end < 0.
usage: P321_LQ=96 P321_OUT=pat321/data python pat332/wret.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327')); sys.path.insert(0, str(ROOT / 'pat329'))
import numpy as np, torch
import fc_corr as C  # noqa: E402
from mse_eval import per_traj, D_FULL  # noqa: E402
from why512 import eu_best  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); assert FC.LQ == 96 and FC.CTX == 512
L, K = 96, 200
ARMS = ['eu_parrot', 'eu96', 'wval', 'wtime', 'wtime_al', 'eu_wval', 'eu_wtime']


def dists(Z, t, H):
    """per query: s [T'] window end positions, valid [B, T'], euclid / wval / wtime distances [B, T'], centroid offset [B, T']."""
    B, T = Z.shape; ar = torch.arange(B, device=dev)
    U = Z.unfold(1, L, 1); s = torch.arange(L, T + 1, device=dev)                        # window ending at s: Z[s-96:s]
    q = Z[ar[:, None], t[:, None] + torch.arange(-L, 0, device=dev)]
    ok = (s[None] + H <= t[:, None])
    eu = (U - q[:, None]).square().mean(-1)
    wv = (U.sort(-1).values - q.sort(-1).values[:, None]).abs().mean(-1)
    eU, eq = U.square(), q.square(); EU, Eq = eU.mean(-1).clamp_min(1e-8), eq.mean(-1).clamp_min(1e-8)
    pU, pq = eU / (EU[..., None] * L), eq / (Eq[:, None] * L)
    wt = (pU.cumsum(-1) - pq.cumsum(-1)[:, None]).abs().sum(-1) / L + (EU / Eq[:, None]).log().abs()
    ar_l = torch.arange(L, device=dev, dtype=Z.dtype)
    off = (pU * ar_l).sum(-1) - (pq * ar_l).sum(-1)[:, None]                                 # candidate centroid - query centroid
    return s, ok, eu, wv, wt, off


def pick(d, ok, s):
    dm = d.masked_fill(~ok, float('inf')); j = dm.argmin(1)
    return s[j], torch.isfinite(dm.min(1).values), j


def copy(Z, start, H):
    ar = torch.arange(len(Z), device=dev)
    return Z[ar[:, None], start.clamp(0, Z.shape[1] - H)[:, None] + torch.arange(H, device=dev)].clamp(-FC.CLIP, FC.CLIP)


@torch.no_grad()
def run(H):
    d, se = FC.load_set('test', dev); acc = {m: dict(smape=[], mse=[]) for m in ARMS}; shift = []
    for i in range(0, len(d['rows']), 64):
        idx = torch.arange(i, min(i + 64, len(d['rows'])))
        rows, t = d['rows'][idx].to(dev), d['t'][idx].to(dev)
        X = se.S[rows]; ar = torch.arange(len(rows), device=dev)
        win = X[ar[:, None], t[:, None] - FC.CTX + torch.arange(FC.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10)
        Z = (X - cm[:, None]) / cs[:, None]
        y = Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)]
        f = (d['f'][idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        s, ok, eu, wv, wt, off = dists(Z, t, H)
        P = {}
        sA, okA = eu_best(Z, t, D_FULL[H], H, False); P['eu_parrot'] = (copy(Z, sA, H), okA)
        for name, dd in (('eu96', eu), ('wval', wv), ('wtime', wt)):
            sb, okb, _ = pick(dd, ok, s); P[name] = (copy(Z, sb, H), okb)
        _, okb, j = pick(wt, ok, s); dl = off.gather(1, j[:, None])[:, 0].round().long().clamp(-48, 48)
        st = s[j] + dl; oka = okb & (st + H <= t) & (st >= 0)
        P['wtime_al'] = (torch.where(oka[:, None], copy(Z, st, H), copy(Z, s[j], H)), okb); shift.append(dl[okb].float().abs().cpu())
        top = eu.masked_fill(~ok, float('inf')).topk(K, 1, largest=False).indices; vt = ok.gather(1, top)
        for name, dd in (('eu_wval', wv), ('eu_wtime', wt)):
            r = dd.gather(1, top).masked_fill(~vt, float('inf')); jj = top.gather(1, r.argmin(1)[:, None])[:, 0]
            P[name] = (copy(Z, s[jj], H), torch.isfinite(r.min(1).values))
        for m, (p, okm) in P.items():
            p = torch.where(okm[:, None], p, f)
            acc[m]['smape'].append(((p - y).abs() / (p.abs() + y.abs() + 1e-10)).mean(1).cpu()); acc[m]['mse'].append((p - y).square().mean(1).cpu())
    tr = {m: {k: per_traj(torch.cat(v), d['ids'], d['inst']) for k, v in acc[m].items()} for m in ARMS}
    print(f'== H{H} (parrot D = {D_FULL[H]}); wtime_al median |shift| {float(torch.cat(shift).median()):.0f} steps', flush=True)
    rng = np.random.default_rng(0)
    for m in ARMS:
        o = [f'sMAPE {100 * float(tr[m]["smape"].median()):6.2f}  MSE {float(tr[m]["mse"].median()):.3f}']
        if m != 'eu_parrot':
            for met, sc in (('smape', 100), ('mse', 1)):
                d0, lo, hi, _ = C.boot(sc * tr[m][met].numpy(), sc * tr['eu_parrot'][met].numpy(), rng); o.append(f'{met} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        print(f'   {m:10s} ' + ' | '.join(o), flush=True)
    torch.save(tr, ROOT / 'pat332' / 'out' / f'wret_H{H}.pt')


if __name__ == '__main__':
    (ROOT / 'pat332' / 'out').mkdir(exist_ok=True)
    for H in (128, 256, 512): run(H)
