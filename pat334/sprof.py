"""PAT-334 S: does the 5-scale similarity profile carry information about the future? (analysis; criteria fixed before results)
Per test query, from the scale heads (l = 16, 32, 48, 64, 96, o = 0; mode c/j as the setting's M2 key):
  log delta_l = log(top-1 distance_l / median candidate distance_l)          (5-dim profile)
  best       = min_l log delta_l                                             (what M2 gating already uses)
  lam        = OLS slope over l of log e(l), e(l) = MSE between the query's last l steps and the last l steps before the
               16-step head's top-1 analog (same analog for all l): backward separation rate per step
  mis        = log delta_96 - log delta_16   (short window matches well, long window does not -> large)
Targets: err_f / err_M2 = log MSE over H of Panda / M2 point forecast; vpt_n (Pathak 0.4, as vpt2.py) of f;
  chg = power-spectrum Hellinger distance between the future y[t:t+H] and the preceding H context steps.
Criteria (ctx2048, all H), Spearman with 95% trajectory-cluster bootstrap CI (1000 reps):
  S1 : OLS fitted on val; test Spearman(pred, err_f) of profile [5 log delta] minus that of [best] : CI > 0
  S2 : Spearman(lam, vpt_n of f) : CI < 0
  S3 : Spearman(mis, chg) : CI > 0
Reported: S1 for err_M2; S3 partial on best; tercile table of lam with median err_f / err_M2; ctx512 full.
usage: same env as mh.py + P334_H"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import mh as M
from mh import G, FC, dev, chunks, backbone, TAG, head, futs
from ens import spec
from vpt2 import vpt_n
from reanchor import parse
O = ROOT / 'pat334' / 'out'; LS = (16, 32, 48, 64, 96)


@torch.no_grad()
def feats(name, H, key):
    mode, fam, wn, k, tau = parse(key); assert fam == 'scale' and M.FAM[fam] == [(l, 0) for l in LS]
    d = torch.load(FC.OUT / f'{name}.pt'); S = d['S'].to(dev); fall = backbone(name, d); out = []
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        y = Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)]
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)
        ph, has, dns, a16 = [], [], [], None
        for (l, o) in M.FAM[fam]:
            js, vs, dn = head(Z, t, H, l, o, False, mode, loc, rep, ni)
            val = torch.isfinite(vs); fu = futs(Z, js, H); v1 = vs[:, :1]
            w = val[:, :k].float() if wn == 'u' else torch.exp(-(vs - v1) / v1.abs().clamp_min(1e-8)).masked_fill(~val, 0.0)[:, :k]
            ph.append((w[..., None] * fu[:, :k]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)); has.append(val[:, 0])
            dns.append(torch.where(val[:, 0], dn, torch.full_like(dn, float('nan'))))
            if l == 16: a16 = torch.where(val[:, 0], js[:, 0], torch.full_like(js[:, 0], -1))
        ph, has, dnn = torch.stack(ph, 1), torch.stack(has, 1), torch.stack(dns, 1); anyh = has.any(1)
        dinf = torch.where(has, dnn, torch.full_like(dnn, float('inf')))
        if tau == 'uni': a = has.float()
        elif tau == 'best': a = torch.nn.functional.one_hot(dinf.argmin(1), len(LS)).float() * has.float()
        else: a = torch.softmax((-dinf / tau).masked_fill(~has, float('-inf')), 1).nan_to_num(0.0)
        m2 = torch.where(anyh[:, None], (a[..., None] * ph).sum(1) / a.sum(1, keepdim=True).clamp_min(1e-12), f)
        ld = torch.log(dnn.clamp_min(1e-12))
        # backward separation along the 16-head top-1 analog
        okA = (a16 >= 96); aa = a16.clamp_min(96)
        L = torch.arange(96, device=dev); qs = Z[ar[:, None], t[:, None] - 96 + L]; an = Z[ar[:, None], aa[:, None] - 96 + L]
        sq = (qs - an).square().flip(1).cumsum(1)                                          # cumulative from the end
        e = torch.stack([sq[:, l - 1] / l for l in LS], 1); le = torch.log(e.clamp_min(1e-12))
        lv = torch.tensor(LS, device=dev, dtype=torch.float32); lc = lv - lv.mean()
        lam = ((le - le.mean(1, keepdim=True)) * lc).sum(1) / lc.square().sum()
        lam = torch.where(okA, lam, torch.full_like(lam, float('nan')))
        prev = Z[ar[:, None], t[:, None] - H + torch.arange(H, device=dev)]
        r = dict(ld=ld, best=torch.where(anyh, ld.nan_to_num(float('inf')).min(1).values, torch.full_like(lam, float('nan'))), lam=lam,
                 mis=ld[:, 4] - ld[:, 0], err_f=torch.log((f - y).square().mean(1).clamp_min(1e-12)),
                 err_M2=torch.log((m2 - y).square().mean(1).clamp_min(1e-12)), vpt_f=vpt_n(f, y, loc, ni),
                 chg=spec(prev[:, None], y)[:, 0], ids=ids)
        out.append({kk: v.cpu() for kk, v in r.items()})
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    return {kk: torch.cat([o[kk] for o in out])[inv] for kk in out[0]}


def rk(x): return x.argsort().argsort().double()


def wcorr(x, y, w):
    sw = w.sum(-1, keepdim=True); mx = (w * x).sum(-1, keepdim=True) / sw; my = (w * y).sum(-1, keepdim=True) / sw
    return (w * (x - mx) * (y - my)).sum(-1) / ((w * (x - mx).square()).sum(-1) * (w * (y - my).square()).sum(-1)).sqrt()


def boot(pairs, g, R=1000, seed=0):
    """pairs: list of (x, y) on the same rows; returns point and bootstrap reps of Spearman for each (trajectory clusters)."""
    g = torch.unique(g, return_inverse=True)[1].to(dev); ng = int(g.max()) + 1; gen = torch.Generator(device=dev).manual_seed(seed)
    P = [(rk(x).to(dev), rk(y).to(dev)) for x, y in pairs]; one = torch.ones_like(P[0][0])
    pt = [float(wcorr(a, b, one)) for a, b in P]; reps = [[] for _ in P]
    for _ in range(R // 50):
        cnt = torch.stack([torch.bincount(torch.randint(0, ng, (ng,), device=dev, generator=gen), minlength=ng) for _ in range(50)]).double()
        w = cnt[:, g]
        for i, (a, b) in enumerate(P): reps[i].append(wcorr(a[None], b[None], w).cpu())
    return pt, [torch.cat(r).numpy() for r in reps]


def ci(v): return np.percentile(v, 2.5), np.percentile(v, 97.5)


def run(H):
    pk = torch.load(O / f'mh_{TAG}_H{H}.pt', weights_only=False); key = pk['pick'][f'{pk["hm"]}_M2'][0]
    V, T = feats('val', H, key), feats('test', H, key)
    def fit(D, cols, tgt):
        X = torch.cat([torch.ones(len(D[tgt]), 1, dtype=torch.float64)] + [D[c].double().reshape(len(D[tgt]), -1) for c in cols], 1)
        ok = torch.isfinite(X).all(1) & torch.isfinite(D[tgt].double()); return X, ok
    print(f'== {TAG} S H{H}: M2 = {key}; test queries {len(T["ids"])}', flush=True)
    for tgt in ('err_f', 'err_M2'):
        Xa, oa = fit(V, ['best'], tgt); Xb, ob = fit(V, ['ld'], tgt); o = oa & ob
        ba = torch.linalg.lstsq(Xa[o], V[tgt].double()[o, None]).solution; bb = torch.linalg.lstsq(Xb[o], V[tgt].double()[o, None]).solution
        Ta, ta = fit(T, ['best'], tgt); Tb, tb = fit(T, ['ld'], tgt); m = ta & tb
        pa, pb = (Ta[m] @ ba)[:, 0], (Tb[m] @ bb)[:, 0]; y = T[tgt].double()[m]
        pt, R = boot([(pa, y), (pb, y)], T['ids'][m])
        dd = R[1] - R[0]; lo, hi = ci(dd)
        print(f'   S1 {tgt:6s}: Spearman best {pt[0]:.4f}  profile {pt[1]:.4f}  diff {pt[1] - pt[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]  (n {int(m.sum())}; coef profile {bb[1:, 0].numpy().round(3).tolist()})', flush=True)
    m = torch.isfinite(T['lam']); pt, R = boot([(T['lam'][m].double(), T['vpt_f'][m].double()), (T['lam'][m].double(), T['err_f'][m].double())], T['ids'][m])
    print(f'   S2 Spearman(lam, vpt_n f) {pt[0]:+.4f} [{ci(R[0])[0]:+.4f}, {ci(R[0])[1]:+.4f}] | reported (lam, err_f) {pt[1]:+.4f} [{ci(R[1])[0]:+.4f}, {ci(R[1])[1]:+.4f}]  (n {int(m.sum())})', flush=True)
    m = torch.isfinite(T['mis']) & torch.isfinite(T['best'])
    mis, chg, bst = T['mis'][m].double(), T['chg'][m].double(), T['best'][m].double()
    def resid(a, b):
        a, b = rk(a), rk(b); bc = b - b.mean(); return a - a.mean() - (a - a.mean()).mul(bc).sum() / bc.square().sum() * bc
    pt, R = boot([(mis, chg), (resid(mis, bst), resid(chg, bst))], T['ids'][m])
    print(f'   S3 Spearman(mis, chg) {pt[0]:+.4f} [{ci(R[0])[0]:+.4f}, {ci(R[0])[1]:+.4f}] | reported partial on best {pt[1]:+.4f} [{ci(R[1])[0]:+.4f}, {ci(R[1])[1]:+.4f}]', flush=True)
    m = torch.isfinite(T['lam']); q = T['lam'][m].quantile(torch.tensor([1 / 3, 2 / 3]).to(T['lam'].dtype))
    for nm, sel in (('low lam', T['lam'][m] <= q[0]), ('mid lam', (T['lam'][m] > q[0]) & (T['lam'][m] <= q[1])), ('high lam', T['lam'][m] > q[1])):
        ef, em, vf = T['err_f'][m][sel], T['err_M2'][m][sel], T['vpt_f'][m][sel]
        print(f'   tercile {nm:8s}: median MSE f {float(ef.exp().median()):.4f}  M2 {float(em.exp().median()):.4f}  M2<f {float((em < ef).float().mean()):.3f}  vpt_n f {float(vf.median()):.1f}', flush=True)
    torch.save({'V': V, 'T': T}, O / f'sprof_{TAG}_H{H}.pt')


if __name__ == '__main__':
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else (128, 256, 512)): run(H)
