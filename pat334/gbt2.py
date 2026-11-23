"""PAT-334 Gbt + map (criteria fixed before results). Adds the classical S-map local-linear map (Farmer-Sidorowich / Sugihara;
PAT-329 classic: 12 delay coordinates = every 8th step of the last 96, w = exp(-8 d / dbar), ridge 0.01 x trace/m; library =
every 96-window of the allowed history with its next H steps observed before the origin; ctx mode window start >= t - CTX).
  point : candidates {f, M2, smap}, weights proportional to 1 / backtest MSE (4 backtest origins of gbt_bb.py)
  prob  : components {delta(f), mapped ens, raw ens}; mapped ens member j: y_j + (x_t - x_j) B, B = the S-map coefficient
          matrix at this origin, x_j = delay coordinates of the 96-window ending at the analog (uncorrected if a_j < 96);
          component weights proportional to 1 / backtest CRPS
Criteria, ctx2048, all H (paired bootstrap of medians over trajectories):
  M1 : Gbt3 - Gbt (2-way, gbt.py)       : sMAPE CI < 0 and MSE CI < 0
  M2 : Gbt3_prob - Gbt_prob (gbt.py)    : CRPS CI < 0
Reported: smap alone, mapped ens alone, Gbt3 - f / M2, ctx512 full. Sanity: 2-way weights here reproduce gbt.py.
usage: same env as mh.py + P334_H [+ P334_BB = RoPE | PaTH | bolt | dynamix for the plugin on another backbone]"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import mh as M
from mh import G, FC, dev, chunks, backbone, TAG, MODE, futs
from ens import crps
from reanchor import parse
from gbt import head_bt, inst_mean, qerr, NB
from mse_eval import per_traj  # noqa: E402
O = ROOT / 'pat334' / 'out'; LQ, TH, LAM = 96, 8.0, 0.01
BBN = os.environ.get('P334_BB')                               # other backbone (gbt_bb_{BBN}_{TAG}_{t,0..3}.pt); default = setting's Panda
PRE = f'gbt_bb_{BBN}_{TAG}' if BBN else f'gbt_bb_{TAG}'; OUTN = f'gbt2_{BBN}_{TAG}' if BBN else f'gbt2_{TAG}'
SUB = torch.arange(LQ - 1, -1, -8).flip(0).to(dev)                                    # 12 delay coordinates (offsets in window)


def coords(Z, end):                                                                   # end [B] or [B, n] (exclusive) -> [.., 12]
    idx = (end[..., None] - LQ + SUB).clamp(0, Z.shape[1] - 1)
    return Z.gather(1, idx.reshape(Z.shape[0], -1)).reshape(*idx.shape)


def smap(Z, to, t, H, fb):
    B, T = Z.shape; s = torch.arange(LQ, T - H + 1, device=dev)
    ok = (s[None] + H <= to[:, None])
    if MODE == 'ctx': ok &= (s - LQ)[None] >= (t - G.CTX)[:, None]
    Xl = Z[:, (s[:, None] - LQ + SUB)]                                                # [B, N, 12]
    Xq = coords(Z, to)
    dist = (Xl - Xq[:, None]).norm(dim=-1).masked_fill(~ok, float('inf'))
    dbar = (dist.masked_fill(~ok, 0).sum(1) / ok.sum(1).clamp_min(1))[:, None].clamp_min(1e-8)
    w = torch.exp(-TH * dist / dbar).masked_fill(~ok, 0); w = w / w.sum(1, keepdim=True).clamp_min(1e-30)
    R = Z[:, (s[:, None] + torch.arange(H, device=dev))].clamp(-M.CLIP, M.CLIP)      # [B, N, H]
    xb = (w[..., None] * Xl).sum(1); rb = (w[..., None] * R).sum(1)
    Xc = Xl - xb[:, None]; A = (Xc * w[..., None]).transpose(1, 2) @ Xc; XY = (Xc * w[..., None]).transpose(1, 2) @ (R - rb[:, None])
    tr = A.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-8) / 12
    Bm = torch.linalg.solve(A.double() + (LAM * tr)[:, None, None].double() * torch.eye(12, device=dev, dtype=torch.float64), XY.double()).float()
    has = ok.any(1)
    p = torch.where(has[:, None], rb + ((Xq - xb)[:, None] @ Bm)[:, 0], fb).clamp(-M.CLIP, M.CLIP)
    return p, Bm, has, Xq


def m2_ens_js(Z, to, t, H, cfg, kb, fb, loc, rep, ni):
    mode, fam, wn, k, tau = cfg; hs = M.FAM[fam]; ph, has, dns, HD = [], [], [], []
    for (l, o) in hs:
        js, vs, dn = head_bt(Z, to, t, H, l, o, mode, loc, rep, ni)
        val = torch.isfinite(vs); fu = futs(Z, js, H); v1 = vs[:, :1]
        wf = val.float() if wn == 'u' else torch.exp(-(vs - v1) / v1.abs().clamp_min(1e-8)).masked_fill(~val, 0.0)
        ph.append((wf[:, :k, None] * fu[:, :k]).sum(1) / wf[:, :k].sum(1, keepdim=True).clamp_min(1e-12)); has.append(val[:, 0])
        dns.append(torch.where(val[:, 0], dn, torch.full_like(dn, float('inf')))); HD.append((fu, wf, js))
    ph, has, dns = torch.stack(ph, 1), torch.stack(has, 1), torch.stack(dns, 1); anyh = has.any(1)
    if tau == 'uni': a = has.float()
    elif tau == 'best': a = torch.nn.functional.one_hot(dns.argmin(1), len(hs)).float() * has.float()
    else: a = torch.softmax((-dns / tau).masked_fill(~has, float('-inf')), 1).nan_to_num(0.0)
    a = a / a.sum(1, keepdim=True).clamp_min(1e-12)
    m2 = torch.where(anyh[:, None], (a[..., None] * ph).sum(1), fb)
    xs = torch.cat([fu[:, :kb] for fu, _, _ in HD], 1); jj = torch.cat([js[:, :kb] for _, _, js in HD], 1)
    ws = torch.cat([a[:, i:i + 1] * wf[:, :kb] / wf[:, :kb].sum(1, keepdim=True).clamp_min(1e-12) for i, (_, wf, _) in enumerate(HD)], 1)
    xs = torch.where(anyh[:, None, None], xs, fb[:, None]); ws = torch.where(anyh[:, None], ws, torch.full_like(ws, 1.0 / ws.shape[1]))
    return m2, xs, ws / ws.sum(1, keepdim=True), jj, anyh


def mapped(Z, xs, jj, anyh, Bm, Xq):
    xj = coords(Z, jj); okj = (jj >= LQ) & anyh[:, None]
    corr = ((Xq[:, None] - xj) @ Bm)                                                   # [B, n, H]
    return torch.where(okj[..., None], xs + corr, xs).clamp(-M.CLIP, M.CLIP)


def inv_w(es, okq):                                                                    # list of [B] errors -> weights [B, n]
    e = torch.stack(es, 1).clamp_min(1e-12); w = (1 / e) / (1 / e).sum(1, keepdim=True)
    return torch.where(okq[:, None], w, torch.full_like(w, 1.0 / w.shape[1]))


@torch.no_grad()
def run(H):
    pk = torch.load(O / f'mh_{TAG}_H{H}.pt', weights_only=False); hm = pk['hm']; key = pk['pick'][f'{hm}_M2'][0]; cfg = parse(key)
    en = torch.load(O / f'ens_{TAG}_H{H}.pt', weights_only=False); kb = en['k']; g2 = None if BBN else torch.load(O / f'gbt_{TAG}_H{H}.pt', weights_only=False)
    BB = [torch.load(O / f'{PRE}_{b}.pt') for b in range(NB)]
    d = torch.load(FC.OUT / 'test.pt'); S = d['S'].to(dev); fall = torch.load(O / f'{PRE}_t.pt')['f'].float() if BBN else backbone('test', d)
    names = ('f', 'M2', 'sm', 'G2', 'G3', 'E2', 'E3'); E = {f'{m}_{n}': [] for n in names for m in ('s', 'm')}
    for k_ in ('c_f', 'c_ens', 'c_map', 'c_G2p', 'c_G3p', 'c_E2p', 'c_E3p', 'w3', 'w3p', 'lbt', 'lte'): E[k_] = []
    for idx in chunks(d, 48):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        hz = torch.arange(H, device=dev); y = Z[ar[:, None], t[:, None] + hz]
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)
        m2, xs, ws, jj, anyh = m2_ens_js(Z, t, t, H, cfg, kb, f, loc, rep, ni)
        sm_, Bm, _, Xq = smap(Z, t, t, H, f); xmap = mapped(Z, xs, jj, anyh, Bm, Xq)
        BT = {k_: [] for k_ in ('ef', 'er', 'es', 'cf', 'ce', 'cm', 'mk')}
        for b in range(NB):
            ok = BB[b]['ok'][idx].to(dev); tp = torch.where(ok, BB[b]['tp'][idx].to(dev), t)
            fb = ((BB[b]['f'][idx, :H].to(dev).float() - cm[:, None]) / cs[:, None]).clamp(-M.CLIP, M.CLIP); yb = Z[ar[:, None], tp[:, None] + hz]
            rb, xb_, wb, jb, ab = m2_ens_js(Z, tp, t, H, cfg, kb, fb, loc, rep, ni)
            sb, Bb, _, Xqb = smap(Z, tp, t, H, fb); xmb = mapped(Z, xb_, jb, ab, Bb, Xqb)
            BT['ef'].append((fb - yb).square().mean(1)); BT['er'].append((rb - yb).square().mean(1)); BT['es'].append((sb - yb).square().mean(1))
            BT['cf'].append((fb - yb).abs().mean(1)); BT['ce'].append(crps(xb_, wb, yb)); BT['cm'].append(crps(xmb, wb, yb)); BT['mk'].append(ok.float())
        mk = torch.stack(BT['mk'], 1)
        agg = {k_: inst_mean(torch.stack(v, 1), mk, loc, ni)[0] for k_, v in BT.items() if k_ != 'mk'}
        okq = inst_mean(torch.stack(BT['ef'], 1), mk, loc, ni)[1]
        w2 = inv_w([agg['ef'], agg['er']], okq); w3 = inv_w([agg['ef'], agg['er'], agg['es']], okq)
        w2p = inv_w([agg['cf'], agg['ce']], okq); w3p = inv_w([agg['cf'], agg['cm'], agg['ce']], okq)
        P = {'f': f, 'M2': m2, 'sm': sm_, 'G2': w2[:, :1] * f + w2[:, 1:2] * m2, 'G3': w3[:, :1] * f + w3[:, 1:2] * m2 + w3[:, 2:3] * sm_, 'E2': (f + m2) / 2, 'E3': (f + m2 + sm_) / 3}
        for n, p in P.items():
            s_, m_ = qerr(p, y); E[f's_{n}'].append(s_.cpu()); E[f'm_{n}'].append(m_.cpu())
        E['c_f'].append((f - y).abs().mean(1).cpu()); E['c_ens'].append(crps(xs, ws, y).cpu()); E['c_map'].append(crps(xmap, ws, y).cpu())
        E['c_G2p'].append(crps(torch.cat([f[:, None], xs], 1), torch.cat([w2p[:, :1], ws * w2p[:, 1:2]], 1), y).cpu())
        E['c_G3p'].append(crps(torch.cat([f[:, None], xmap, xs], 1), torch.cat([w3p[:, :1], ws * w3p[:, 1:2], ws * w3p[:, 2:3]], 1), y).cpu())
        h2 = torch.full_like(w2p[:, :1], 0.5); h3 = torch.full_like(w2p[:, :1], 1 / 3)
        E['c_E2p'].append(crps(torch.cat([f[:, None], xs], 1), torch.cat([h2, ws * h2], 1), y).cpu())
        E['c_E3p'].append(crps(torch.cat([f[:, None], xmap, xs], 1), torch.cat([h3, ws * h3, ws * h3], 1), y).cpu())
        E['lbt'].append(torch.where(okq, (agg['ef'].clamp_min(1e-12) / agg['er'].clamp_min(1e-12)).log(), torch.nan)[loc].cpu())
        E['lte'].append(((f - y).square().mean(1).clamp_min(1e-12) / (m2 - y).square().mean(1).clamp_min(1e-12)).log().cpu())
        E['w3'].append(w3.cpu()); E['w3p'].append(w3p.cpu())
    order = torch.cat(chunks(d, 48)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    R = {k_: torch.cat(v)[inv] for k_, v in E.items()}
    Pt = {k_: per_traj(v, d['ids'], d['inst']) for k_, v in R.items() if not k_.startswith('w3') and k_ not in ('lbt', 'lte')}
    if g2:
        dd = (Pt['c_G2p'] - g2['P']['cr_Gp']).abs(); rel = dd / (g2['P']['cr_Gp'].abs() + 1e-3); ia = int(dd.argmax())
        print(f'   sanity CRPS worst traj: abs {float(dd[ia]):.1e} at ref {float(g2["P"]["cr_Gp"][ia]):.3e}; max rel {float(rel.max()):.1e}', flush=True)
    dG, dGp = (float((Pt['s_G2'] - g2['A']['G']['smape']).abs().max()), float(rel.max())) if g2 else (0.0, 0.0)
    print(f'== {TAG} [{BBN or "panda"}] Gbt3 H{H}: sanity 2-way vs gbt.py sMAPE {dG:.1e} CRPS rel {dGp:.1e}; median weights point f/M2/smap '
          + '/'.join(f'{float(v):.3f}' for v in R['w3'].median(0).values) + '  prob f/map/ens ' + '/'.join(f'{float(v):.3f}' for v in R['w3p'].median(0).values), flush=True)
    assert dG < 1e-2 and dGp < 1e-2 and (not g2 or abs(float(Pt['c_G2p'].median() - g2['P']['cr_Gp'].median())) < 1e-4)
    A = {n: {'smape': Pt[f's_{n}'], 'mse': Pt[f'm_{n}']} for n in names}
    print('   test ' + '  '.join(f'{n} {100 * float(v["smape"].median()):.2f}/{float(v["mse"].median()):.4f}' for n, v in A.items())
          + '  | CRPS ' + '  '.join(f'{k_} {float(Pt[k_].median()):.4f}' for k_ in ('c_f', 'c_ens', 'c_map', 'c_G2p', 'c_G3p')), flush=True)
    rng = np.random.default_rng(0)
    for x, rf in (('M2', 'f'), ('G3', 'G2'), ('G3', 'f'), ('G3', 'M2'), ('G3', 'sm'), ('sm', 'M2'), ('sm', 'f')):
        o = [f'{met} %+.3f [%+.3f, %+.3f]' % G.C.boot(sc * A[x][met].numpy(), sc * A[rf][met].numpy(), rng)[:3] for met, sc in (('smape', 100), ('mse', 1))]
        print(f'   {x:3s} - {rf:3s} ' + ' | '.join(o), flush=True)
    for x, rf in (('G2', 'E2'), ('G3', 'E3')):
        o = [f'{met} %+.3f [%+.3f, %+.3f]' % G.C.boot(sc * A[x][met].numpy(), sc * A[rf][met].numpy(), rng)[:3] for met, sc in (('smape', 100), ('mse', 1))]
        print(f'   {x:3s} - {rf:3s} ' + ' | '.join(o), flush=True)
    from scipy.stats import spearmanr
    lb, lt = R['lbt'].numpy(), R['lte'].numpy(); mk_ = np.isfinite(lb)
    print(f'   backtest vs test log(e_f/e_M2) Spearman {spearmanr(lb[mk_], lt[mk_])[0]:+.3f} (n {int(mk_.sum())}); sign agree {float(((lb[mk_] > 0) == (lt[mk_] > 0)).mean()):.3f}', flush=True)
    for x, rf in (('c_ens', 'c_f'), ('c_G2p', 'c_E2p'), ('c_G3p', 'c_E3p'), ('c_G3p', 'c_G2p'), ('c_map', 'c_ens'), ('c_G3p', 'c_ens'), ('c_G3p', 'c_f')):
        print(f'   {x} - {rf}  %+.4f [%+.4f, %+.4f]' % G.C.boot(Pt[x].numpy(), Pt[rf].numpy(), rng)[:3], flush=True)
    torch.save({'A': A, 'P': Pt, 'w3': R['w3'], 'w3p': R['w3p']}, O / f'{OUTN}_H{H}.pt')


if __name__ == '__main__':
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else (128, 256, 512)): run(H)
