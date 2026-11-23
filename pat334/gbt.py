"""PAT-334 Gbt: in-context backtest gate between Panda f and retrieval (training-free, no parameters fitted anywhere).
For every test query (origin t) the trajectory's own past is replayed at NB = 4 backtest origins t'_b = t - 512 - 128 b
(gbt_bb.py; Panda context and backtest truth inside the allowed history; retrieval at t' uses candidates a + H <= t' and,
ctx mode, window start >= t - CTX). Backtest errors are averaged over valid origins and the instance's channels.
  point : p = alpha f + (1 - alpha) M2,  alpha = e_M2 / (e_f + e_M2)   (inverse-MSE weights; e = backtest MSE over H)
  prob  : mixture  alpha_p delta(f) + (1 - alpha_p) ens,  alpha_p = c_ens / (c_f + c_ens)   (c = backtest CRPS; f: MAE)
  no valid backtest origin -> alpha = alpha_p = 0.5.  M2 / ens configs (and ens k) = the setting's existing val choices.
Criteria (fixed before results), ctx2048, all H, paired bootstrap of medians over trajectories:
  G1 : Gbt - f and Gbt - M2 : sMAPE CI < 0 and MSE CI < 0
  G2 : Gbt - Bq (blend, match-quality gate)    : sMAPE CI < 0 and MSE CI < 0
  G3 : Gbt_prob - ens : CRPS CI < 0
Reported: hard selection (alpha = 1[e_f < e_M2]), share of queries with valid backtests, median alpha, ctx512 full.
usage: same env as mh.py + P334_H"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import mh as M
from mh import G, FC, dev, chunks, backbone, TAG, MODE, futs
from ens import crps
from reanchor import parse
from mse_eval import per_traj  # noqa: E402
O = ROOT / 'pat334' / 'out'; NB = 4


def head_bt(Z, to, t, H, l, o, mode, loc, rep, ni):
    a, dist, ok = M.cand(Z, to, H, l, o, False)
    if MODE == 'ctx': ok &= (a - o - l)[None] >= (t - G.CTX)[:, None]
    if mode == 'j':
        okI = ok[rep]; dist = torch.zeros(ni, len(a), device=dev).index_add_(0, loc, dist.masked_fill(~ok, 0.0)); ok = okI
    js, vs = M.nms(dist.masked_fill(~ok, float('inf')), a)
    med = torch.nanmedian(dist.masked_fill(~ok, float('nan')), 1).values
    dn = vs[:, 0] / med.abs().clamp_min(1e-12)
    if mode == 'j': js, vs, dn = js[loc], vs[loc], dn[loc]
    return js, vs, dn


def m2_ens(Z, to, t, H, cfg, kb, fb, loc, rep, ni):
    mode, fam, wn, k, tau = cfg; hs = M.FAM[fam]; ph, has, dns, HD = [], [], [], []
    for (l, o) in hs:
        js, vs, dn = head_bt(Z, to, t, H, l, o, mode, loc, rep, ni)
        val = torch.isfinite(vs); fu = futs(Z, js, H); v1 = vs[:, :1]
        wf = val.float() if wn == 'u' else torch.exp(-(vs - v1) / v1.abs().clamp_min(1e-8)).masked_fill(~val, 0.0)
        ph.append((wf[:, :k, None] * fu[:, :k]).sum(1) / wf[:, :k].sum(1, keepdim=True).clamp_min(1e-12)); has.append(val[:, 0])
        dns.append(torch.where(val[:, 0], dn, torch.full_like(dn, float('inf')))); HD.append((fu, wf))
    ph, has, dns = torch.stack(ph, 1), torch.stack(has, 1), torch.stack(dns, 1); anyh = has.any(1)
    if tau == 'uni': a = has.float()
    elif tau == 'best': a = torch.nn.functional.one_hot(dns.argmin(1), len(hs)).float() * has.float()
    else: a = torch.softmax((-dns / tau).masked_fill(~has, float('-inf')), 1).nan_to_num(0.0)
    a = a / a.sum(1, keepdim=True).clamp_min(1e-12)
    m2 = torch.where(anyh[:, None], (a[..., None] * ph).sum(1), fb)
    xs = torch.cat([fu[:, :kb] for fu, _ in HD], 1)
    ws = torch.cat([a[:, i:i + 1] * wf[:, :kb] / wf[:, :kb].sum(1, keepdim=True).clamp_min(1e-12) for i, (_, wf) in enumerate(HD)], 1)
    xs = torch.where(anyh[:, None, None], xs, fb[:, None]); ws = torch.where(anyh[:, None], ws, torch.full_like(ws, 1.0 / ws.shape[1]))
    return m2, xs, ws / ws.sum(1, keepdim=True)


def inst_mean(x, m, loc, ni):                                  # x, m [B, NB] -> instance mean over valid (b, channel), per row
    s = torch.zeros(ni, device=dev).index_add_(0, loc, (x * m).sum(1)); c = torch.zeros(ni, device=dev).index_add_(0, loc, m.sum(1))
    return (s / c.clamp_min(1e-12))[loc], (c > 0)[loc]


def qerr(p, y): return (p - y).abs().div(p.abs() + y.abs() + 1e-10).mean(1), (p - y).square().mean(1)


@torch.no_grad()
def run(H):
    pk = torch.load(O / f'mh_{TAG}_H{H}.pt', weights_only=False); hm = pk['hm']; key = pk['pick'][f'{hm}_M2'][0]; cfg = parse(key)
    en = torch.load(O / f'ens_{TAG}_H{H}.pt', weights_only=False); kb = en['k']
    BB = [torch.load(O / f'gbt_bb_{TAG}_{b}.pt') for b in range(NB)]
    d = torch.load(FC.OUT / 'test.pt'); S = d['S'].to(dev); fall = backbone('test', d)
    E = {k: [] for k in ('sm_f', 'ms_f', 'sm_M2', 'ms_M2', 'sm_G', 'ms_G', 'sm_Gh', 'ms_Gh', 'cr_f', 'cr_ens', 'cr_Gp', 'cr_Gph', 'al', 'alp', 'okq')}
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        hz = torch.arange(H, device=dev); y = Z[ar[:, None], t[:, None] + hz]
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)
        m2, xs, ws = m2_ens(Z, t, t, H, cfg, kb, f, loc, rep, ni)
        ef, er, cf, ce, mk = [], [], [], [], []
        for b in range(NB):
            ok = BB[b]['ok'][idx].to(dev); tp = torch.where(ok, BB[b]['tp'][idx].to(dev), t)
            fb = ((BB[b]['f'][idx, :H].to(dev).float() - cm[:, None]) / cs[:, None]).clamp(-M.CLIP, M.CLIP)
            yb = Z[ar[:, None], tp[:, None] + hz]
            rb, xb, wb = m2_ens(Z, tp, t, H, cfg, kb, fb, loc, rep, ni)
            ef.append((fb - yb).square().mean(1)); er.append((rb - yb).square().mean(1))
            cf.append((fb - yb).abs().mean(1)); ce.append(crps(xb, wb, yb)); mk.append(ok.float())
        mk = torch.stack(mk, 1)
        (ef, okq), (er, _), (cf, _), (ce, _) = [inst_mean(torch.stack(v, 1), mk, loc, ni) for v in (ef, er, cf, ce)]
        al = torch.where(okq, er / (ef + er).clamp_min(1e-12), torch.full_like(ef, 0.5)); alh = torch.where(okq, (ef < er).float(), torch.full_like(ef, 0.5))
        alp = torch.where(okq, ce / (cf + ce).clamp_min(1e-12), torch.full_like(ef, 0.5)); alph = torch.where(okq, (cf < ce).float(), torch.full_like(ef, 0.5))
        for nm, p in (('f', f), ('M2', m2), ('G', al[:, None] * f + (1 - al[:, None]) * m2), ('Gh', alh[:, None] * f + (1 - alh[:, None]) * m2)):
            s_, m_ = qerr(p, y); E[f'sm_{nm}'].append(s_.cpu()); E[f'ms_{nm}'].append(m_.cpu())
        xm = torch.cat([xs, f[:, None]], 1)
        E['cr_f'].append((f - y).abs().mean(1).cpu()); E['cr_ens'].append(crps(xs, ws, y).cpu())
        for nm, a_ in (('cr_Gp', alp), ('cr_Gph', alph)):
            E[nm].append(crps(xm, torch.cat([ws * (1 - a_[:, None]), a_[:, None]], 1), y).cpu())
        E['al'].append(al.cpu()); E['alp'].append(alp.cpu()); E['okq'].append(okq.float().cpu())
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    R = {k: torch.cat(v)[inv] for k, v in E.items()}; P = {k: per_traj(v, d['ids'], d['inst']) for k, v in R.items() if k not in ('al', 'alp', 'okq')}
    dM = float((P['sm_M2'] - pk['A'][f'{hm}_M2']['smape']).abs().max()); dE = float((P['cr_ens'] - en['A']['ens']['crps']).abs().max())
    print(f'== {TAG} Gbt H{H}: M2 = {key}, ens k = {kb}; sanity M2 {dM:.1e}, ens crps {dE:.1e}; valid backtest {float(R["okq"].mean()):.3f}; '
          f'alpha med {float(R["al"].median()):.3f} (share > .5 {float((R["al"] > .5).float().mean()):.3f}), alpha_p med {float(R["alp"].median()):.3f}', flush=True)
    assert dM < 1e-2 and dE < 1e-3
    bq = torch.load(O / f'blend_{TAG}_H{H}.pt', weights_only=False)['A']['Bq']
    A = {nm: {'smape': P[f'sm_{nm}'], 'mse': P[f'ms_{nm}']} for nm in ('f', 'M2', 'G', 'Gh')} | {'Bq': bq}
    print('   test ' + '  '.join(f'{k} {100 * float(v["smape"].median()):.2f}/{float(v["mse"].median()):.4f}' for k, v in A.items())
          + '  | CRPS ' + '  '.join(f'{k} {float(P[k].median()):.4f}' for k in ('cr_f', 'cr_ens', 'cr_Gp', 'cr_Gph')), flush=True)
    rng = np.random.default_rng(0)
    for x, rf in (('G', 'f'), ('G', 'M2'), ('G', 'Bq'), ('Gh', 'f'), ('Gh', 'M2'), ('Gh', 'Bq')):
        o = [f'{met} %+.3f [%+.3f, %+.3f]' % G.C.boot(sc * A[x][met].numpy(), sc * A[rf][met].numpy(), rng)[:3] for met, sc in (('smape', 100), ('mse', 1))]
        print(f'   {x:3s} - {rf:3s} ' + ' | '.join(o), flush=True)
    for x, rf in (('cr_Gp', 'cr_ens'), ('cr_Gp', 'cr_f'), ('cr_Gph', 'cr_ens')):
        print(f'   {x} - {rf}  %+.4f [%+.4f, %+.4f]' % G.C.boot(P[x].numpy(), P[rf].numpy(), rng)[:3], flush=True)
    torch.save({'A': A, 'P': P, 'al': R['al'], 'alp': R['alp'], 'okq': R['okq']}, O / f'gbt_{TAG}_H{H}.pt')


if __name__ == '__main__':
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else (128, 256, 512)): run(H)
