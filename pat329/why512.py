"""PAT-329 ablation ladder parrot-full -> FreshCast top-1 (ctx 512, LQ 96, test, all top-1 copies; fallback = backbone f).
  A eu_parrot  Euclid, D = D_FULL[H] (= parrot-full), all positions, raw copy
  B eu96       Euclid, D = 96                                  (A->B: window length)
  C euc96      Euclid on mean-centred windows, D = 96          (B->C: drop level)
  D cos96      Pearson cosine argmax (= FreshCast top shape cand), raw copy (C->D: drop amplitude)
  E cos96_sh   D with shape readout (rescaled to query level / sd)          (D->E: readout)
  F eu96_pool  Euclid D = 96 restricted to FreshCast's 200 cosine candidates (B->F: candidate pool)
  G fc_top1    FreshCast scorer argmax over 400 entries (its own readout)  (D/E->G: scorer)
  P eu_sh      A's analog with shape readout                               (A->P: readout at the parrot analog)
Diagnostics of the chosen analog s: lag t - s, level gap |mu_s - mu_t| / ctx_sd, |log sd_s / sd_t| (96-windows),
raw 96-window distance / ctx_sd^2, agreement |s - s_A| <= 8.
Criterion (fixed before results): a step explains the L512 gap if its paired median sMAPE difference has a 95% CI
excluding 0 and |delta| >= 0.5.
usage: P321_LQ=96 P321_OUT=pat321/data python pat329/why512.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327')); sys.path.insert(0, str(ROOT / 'pat329'))
import numpy as np, torch
import fc_corr as C  # noqa: E402
from mse_eval import per_traj, D_FULL  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); assert FC.LQ == 96 and FC.CTX == 512
ARMS = ['eu_parrot', 'eu96', 'euc96', 'cos96', 'cos96_sh', 'eu96_pool', 'fc_top1', 'eu_sh']
STEPS = [('eu96', 'eu_parrot'), ('euc96', 'eu96'), ('cos96', 'euc96'), ('cos96_sh', 'cos96'), ('eu96_pool', 'eu96'),
         ('fc_top1', 'cos96'), ('fc_top1', 'cos96_sh'), ('eu_sh', 'eu_parrot'), ('fc_top1', 'eu_parrot')]


def windows(Z, s, D):
    ar = torch.arange(len(Z), device=dev)
    return Z[ar[:, None, None], s[..., None] + torch.arange(-D, 0, device=dev)]          # [B, n, D]


@torch.no_grad()
def eu_best(Z, t, D, H, center, cand=None):
    """argmin Euclid distance of the D-window ending at s vs ending at t, s + H <= t; cand restricts s."""
    ar = torch.arange(len(Z), device=dev)
    q = Z[ar[:, None], t[:, None] + torch.arange(-D, 0, device=dev)]
    s = torch.arange(D, int(t.max()), device=dev)[None].expand(len(Z), -1) if cand is None else cand
    win = windows(Z, s.clamp(min=D, max=Z.shape[1]), D)
    if center: win = win - win.mean(-1, keepdim=True); q = q - q.mean(-1, keepdim=True)
    dist = (win - q[:, None]).square().sum(-1).masked_fill(~((s + H <= t[:, None]) & (s >= D)), float('inf'))
    best = dist.argmin(1)
    return s.gather(1, best[:, None])[:, 0], torch.isfinite(dist.min(1).values)


def read_at(se, rows, t, b, s, H, shape):
    S = se.S; kh = torch.arange(H, device=dev)
    s = s.clamp(0, FC.T_LEN - H)                       # invalid s (fallback f is used) kept in range
    cont = S[rows[:, None], s[:, None] + kh]
    if shape:                                          # 96-window stats exist only for s >= 96; else raw copy
        s9 = s.clamp(min=96)
        mc, sc = se.mu[rows, s9][:, None], se.sd[rows, s9].clamp_min(1e-10)[:, None]
        cont = torch.where((s >= 96)[:, None], se.mu[rows, t][:, None] + se.sd[rows, t][:, None] / sc * (cont - mc), cont)
    return ((cont - b['cm'][:, None]) / b['cs'][:, None]).clamp(-FC.CLIP, FC.CLIP)


@torch.no_grad()
def run(H):
    fcm = C.load_fc(H, dev); d, se = FC.load_set('test', dev)
    acc = {m: dict(smape=[], mse=[], lag=[], gap=[], amp=[], dist=[], agree=[]) for m in ARMS}; vfrac = []
    for i in range(0, len(d['rows']), 64):
        idx = torch.arange(i, min(i + 64, len(d['rows'])))
        rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
        y = FC.target(se, rows, t, b, H)
        f = (d['f'][idx, :H].to(dev) - b['cm'][:, None]) / b['cs'][:, None]
        Z = (se.S[rows] - b['cm'][:, None]) / b['cs'][:, None]
        N = b['s'].shape[1] // 2
        sA, okA = eu_best(Z, t, D_FULL[H], H, False)
        sB, okB = eu_best(Z, t, 96, H, False)
        sC, okC = eu_best(Z, t, 96, H, True)
        sF, okF = eu_best(Z, t, 96, H, False, cand=torch.where(b['valid'][:, :N], b['s'][:, :N], -10 ** 6))
        sD, okD = b['s'][:, 0], b['valid'][:, 0]
        g = w.argmax(1); sG = b['s'].gather(1, g[:, None])[:, 0]; okG = b['any']
        vfrac.append(b['val'].gather(1, g[:, None])[:, 0][okG].float().cpu())
        pick = {'eu_parrot': (sA, okA, False), 'eu96': (sB, okB, False), 'euc96': (sC, okC, False), 'cos96': (sD, okD, False),
                'cos96_sh': (sD, okD, True), 'eu96_pool': (sF, okF, False), 'eu_sh': (sA, okA, True)}
        for m in ARMS:
            if m == 'fc_top1': s, ok = sG, okG; p = R.gather(1, g[:, None, None].expand(-1, 1, H))[:, 0]
            else: s, ok, sh = pick[m]; p = read_at(se, rows, t, b, s, H, sh)
            p = torch.where(ok[:, None], p, f)
            a = acc[m]
            a['smape'].append(((p - y).abs() / (p.abs() + y.abs() + 1e-10)).mean(1).cpu())
            a['mse'].append((p - y).square().mean(1).cpu())
            sc = s.clamp(96, FC.T_LEN - H)
            a['lag'].append(torch.where(ok, (t - s).float(), float('nan')).cpu())
            a['gap'].append(torch.where(ok, (se.mu[rows, sc] - se.mu[rows, t]).abs() / b['cs'], float('nan')).cpu())
            a['amp'].append(torch.where(ok, (se.sd[rows, sc].clamp_min(1e-10) / se.sd[rows, t].clamp_min(1e-10)).log().abs(), float('nan')).cpu())
            wq = Z[torch.arange(len(Z), device=dev)[:, None], t[:, None] + torch.arange(-96, 0, device=dev)]
            a['dist'].append(torch.where(ok, (windows(Z, sc[:, None], 96)[:, 0] - wq).square().mean(-1), float('nan')).cpu())
            a['agree'].append(torch.where(ok & okA, ((s - sA).abs() <= 8).float(), float('nan')).cpu())
    tr = {m: {k: per_traj(torch.cat(acc[m][k]), d['ids'], d['inst']) for k in ('smape', 'mse')} for m in ARMS}
    print(f'== H{H} (parrot D = {D_FULL[H]}); fc_top1 picks value readout {100 * float(torch.cat(vfrac).mean()):.1f}%', flush=True)
    print(f'   {"arm":10s} sMAPE    MSE     lag   lvl_gap |log_sd| dist96 agree_A', flush=True)
    for m in ARMS:
        dg = {k: float(np.nanmedian(torch.cat(acc[m][k]).numpy())) for k in ('lag', 'gap', 'amp', 'dist')}
        ag = float(np.nanmean(torch.cat(acc[m]['agree']).numpy()))
        print(f'   {m:10s} {100 * float(tr[m]["smape"].median()):6.2f} {float(tr[m]["mse"].median()):6.3f} {dg["lag"]:7.0f} '
              f'{dg["gap"]:7.3f} {dg["amp"]:7.3f} {dg["dist"]:6.3f} {ag:6.3f}', flush=True)
    rng = np.random.default_rng(0)
    for x, yv in STEPS:
        d0, lo, hi, fr = C.boot(100 * tr[x]['smape'].numpy(), 100 * tr[yv]['smape'].numpy(), rng)
        print(f'   {x:10s} - {yv:10s} sMAPE {d0:+.2f} [{lo:+.2f}, {hi:+.2f}] frac {fr:.3f}', flush=True)
    torch.save({m: {k: torch.cat(v) for k, v in acc[m].items()} | tr[m] for m in ARMS}, ROOT / 'pat329' / 'out' / f'why_H{H}.pt')


if __name__ == '__main__':
    for H in (512, 128, 256): run(H)
