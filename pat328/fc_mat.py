"""PAT-328: the user's retrieval transform on dysts. Retrieval and transform are two separate steps.

  1. retrieval  frozen PAT-321 FreshCast fc scorer: candidates j (window x_j, future y+_j = readout R_j, 400 entries =
                200 candidates x {value, shape} readouts) and weights w_j. Its descriptors are used only here.
  2. transform  learned; sees only x_t, x_j, y+_j (pair-m) or the observed path (path-m); never the retrieval
                descriptors or weights.
  3. merge      r = sum_j w_j yhat_j

  yhat_j = [I + Dec (M_j - I) E] y+_j        E: H -> D linear (no bias), Dec: D -> H linear, zero-initialised
                                              (training starts exactly at r-only)
  A = I + U V^T, U, V in R^{D x r} = 0.03 tanh(.)   (general low rank, not Householder)
  pair-m  M_j = A_j,  (U_j, V_j) = Net(x_t, x_j, y+_j)
  path-m  steps i_k = t - H - 16 k, k = 0 .. n_j - 1, n_j = floor((t - H - j) / 16) + 1 (walk from t - H back to j; no
          step after t - H, so y+_{i_k} = y(i_k + 1 : i_k + H) is fully observed: no leakage)
          A_k = I + U_k V_k^T, (U_k, V_k) = Net(x_{i_k}, y+_{i_k}, x_t)     (shared by all candidates of a query)
          M_j = A_0 A_1 ... A_{n_j - 1}
All windows / futures are context-standardised like the readouts (x_j in the readout's units, see pat327 phis).
Training / selection as PAT-327. usage: python pat328/fc_mat.py --stage train --arm pairm|pathm --H 128 ; --stage test"""
import argparse, copy, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('P321_OUT', str(ROOT / 'pat321' / 'data'))
sys.path.insert(0, str(ROOT / 'pat327'))
import numpy as np
import torch
import torch.nn as nn
import fc_corr as C  # noqa: E402  (pat327: phis, step_inputs, smap_corr, boot, load_fc)
FC = C.FC

OUT = ROOT / 'pat328' / 'out'
LQ, STEP = FC.LQ, 16
D, RK, HID, SC = 16, 2, 128, 0.03
LAMS = C.LAMS


class Mat(nn.Module):
    def __init__(self, arm, H):
        super().__init__()
        self.arm, self.H = arm, H
        self.fy = nn.Linear(H, 64)                      # future encoder for the matrix generator
        self.net = nn.Sequential(nn.Linear(LQ + LQ + 64, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU(), nn.Linear(HID, 2 * D * RK))
        self.E = nn.Linear(H, D, bias=False)
        self.dec = nn.Linear(D, H)
        nn.init.zeros_(self.dec.weight); nn.init.zeros_(self.dec.bias)

    def uv(self, xa, xb, y):
        o = SC * torch.tanh(self.net(torch.cat([xa, xb, self.fy(y)], -1)))
        U, V = o[..., :D * RK].unflatten(-1, (D, RK)), o[..., D * RK:].unflatten(-1, (D, RK))
        return U, V

    def forward(self, w, R, qp, cp, steps=None, n=None):
        """sum_j w_j (yhat_j - y+_j) = Dec(sum_j w_j (M_j - I) E y+_j)  [B, H]"""
        e = self.E(R)                                                                     # [B, 2N, D]
        if self.arm == 'pairm':
            U, V = self.uv(qp[:, None].expand_as(cp), cp, R)                              # [B, 2N, D, RK]
            v = (U @ (V.transpose(-1, -2) @ e[..., None]))[..., 0]                         # (A - I) e
        else:
            xs, ys, ok = steps                                                            # [B, K, LQ], [B, K, H], [B, K]
            U, V = self.uv(qp[:, None].expand_as(xs), xs, ys)                             # [B, K, D, RK]
            I = torch.eye(D, device=R.device)
            A = I + U @ V.transpose(-1, -2)
            A = torch.where(ok[..., None, None], A, I)
            B, K = ok.shape
            Ms = [I.expand(B, D, D)]; M = Ms[0]
            for k in range(K):                                                            # prefix products A_0 ... A_{k}
                M = M @ A[:, k]; Ms.append(M)
            Ms = torch.stack(Ms, 1)                                                       # Ms[:, m] = A_0 ... A_{m-1}
            Mj = Ms.gather(1, n[..., None, None].expand(-1, -1, D, D))                    # [B, 2N, D, D]
            v = (Mj @ e[..., None])[..., 0] - e
        return self.dec((w[..., None] * v).sum(1))


def path_steps(se, rows, t, b, H):
    """observed (window, future) pairs at i_k = t - H - 16k (k = 0..K-1), context-standardised; valid if i_k >= LQ.
    n_j = floor((t - H - j) / 16) + 1 for every entry j (= b['s'])."""
    S, dev = se.S, se.S.device
    i0 = t - H
    K = int(((i0 - LQ) // STEP).max()) + 1
    k = torch.arange(K, device=dev)
    ik = i0[:, None] - STEP * k[None]                                                     # [B, K]
    ok = ik >= LQ
    ikc = torch.where(ok, ik, torch.full_like(ik, LQ))                                   # invalid steps: read a safe
    ikc = torch.minimum(ikc, (t[:, None] - H).clamp_min(LQ))                              # in-range slot, zeroed below
    ok = ok & (ikc + H <= t[:, None])
    cm, cs = b['cm'][:, None, None], b['cs'][:, None, None]
    xs = (S[rows[:, None, None], ikc[..., None] + torch.arange(-LQ, 0, device=dev)] - cm) / cs
    ys = (S[rows[:, None, None], ikc[..., None] + torch.arange(H, device=dev)] - cm) / cs
    assert bool(((ikc + H) <= t[:, None])[ok].all()), 'leakage: a used step future reaches past t'
    n = ((i0[:, None] - b['s']) // STEP + 1).clamp(1, K)
    return (xs.clamp(-FC.CLIP, FC.CLIP) * ok[..., None], ys.clamp(-FC.CLIP, FC.CLIP) * ok[..., None], ok), n


def corr_of(model, se, rows, t, b, R, w, qp, cp, H):
    if model.arm == 'pathm':
        steps, n = path_steps(se, rows, t, b, H)
        return model(w, R, qp, cp, steps=steps, n=n)
    return model(w, R, qp, cp)


@torch.no_grad()
def val_loss(model, sv, dv, vsel, veps, H, dev, fcm, bs=256):
    model.eval(); tot = 0.0
    for i in range(0, len(vsel), bs):
        idx = vsel[i:i + bs]
        rows, t, b, R, w, zn, qp, cp = C.step_inputs(sv, dv, idx, H, dev, fcm)
        y = FC.target(sv, rows, t, b, H)
        r = (w[..., None] * R).sum(1) + corr_of(model, sv, rows, t, b, R, w, qp, cp, H)
        tot += float(((r - y).square().mean(1) / veps[i:i + bs].to(dev)).clamp_max(FC.C_MAX).sum())
    model.train()
    return tot / len(vsel)


def run_train(a, dev):
    H, arm = a.H, a.arm
    fcm = C.load_fc(H, dev)
    d, se = FC.load_set('train', dev)
    g = torch.Generator().manual_seed(a.seed)
    sel = torch.randperm(len(d['rows']), generator=g)[:a.n_q]
    sel, eps, _, _ = FC.untrained_eps(se, d, sel, H, dev)
    dv, sv = FC.load_set('val', dev)
    vsel, veps, _, _ = FC.untrained_eps(sv, dv, torch.arange(len(dv['rows'])), H, dev)
    torch.manual_seed(a.seed)
    model = Mat(arm, H).to(dev); ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=FC.LR, weight_decay=FC.WD)
    bs = a.bs; n_step = math.ceil(len(sel) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * n_step)
    v0 = val_loss(ema, sv, dv, vsel, veps, H, dev, fcm)
    best, hist = (v0, -1, copy.deepcopy(ema.state_dict())), []
    print(f'[train] {arm} H{H}: {len(sel)} queries, {n_step} steps/epoch, val loss at r-only {v0:.4f}', flush=True)
    for ep in range(a.epochs):
        perm = torch.randperm(len(sel), generator=g); tot = 0.0; t0 = time.time()
        for i in range(n_step):
            pi = perm[i * bs:(i + 1) * bs]; idx = sel[pi]
            rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
            y = FC.target(se, rows, t, b, H)
            r = (w[..., None] * R).sum(1) + corr_of(model, se, rows, t, b, R, w, qp, cp, H)
            loss = ((r - y).square().mean(1) / eps[pi].to(dev)).clamp_max(FC.C_MAX).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(FC.EMA).add_(pm, alpha=1 - FC.EMA)
            tot += float(loss.detach())
            if a.max_steps and i + 1 >= a.max_steps: break
        vl = val_loss(ema, sv, dv, vsel, veps, H, dev, fcm); hist.append((tot / n_step, vl))
        if vl < best[0]: best = (vl, ep, copy.deepcopy(ema.state_dict()))
        print(f'[train] {arm} H{H} epoch {ep} loss {tot / n_step:.4f} val {vl:.4f} (r-only {v0:.4f}) {time.time() - t0:.0f}s', flush=True)
        if a.max_steps: break
    print(f'[train] {arm} H{H}: best val {best[0]:.4f} at epoch {best[1]} (r-only {v0:.4f})', flush=True)
    if not a.max_steps:
        torch.save(dict(state=best[2], hist=hist, best_val=best[0], best_ep=best[1], v0=v0), OUT / f'mat_{arm}_H{H}_s{a.seed}.pt')


@torch.no_grad()
def predict(name, H, dev, arms, bs=128):
    fcm = C.load_fc(H, dev)
    d, se = FC.load_set(name, dev)
    models = {}
    for arm in arms:
        m = Mat(arm, H).to(dev); m.load_state_dict(torch.load(OUT / f'mat_{arm}_H{H}_s99.pt')['state']); m.eval(); models[arm] = m
    out = {k: [] for k in ['fc', 'y'] + [f'smap{l}' for l in LAMS] + list(models)}
    for i in range(0, len(d['rows']), bs):
        idx = torch.arange(i, min(i + bs, len(d['rows'])))
        rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
        f = (d['f'][idx, :H].to(dev) - b['cm'][:, None]) / b['cs'][:, None]
        any_ = b['any'][:, None]
        r0 = (w[..., None] * R).sum(1)
        out['fc'].append(torch.where(any_, r0, f).cpu()); out['y'].append(FC.target(se, rows, t, b, H).cpu())
        for l, c in zip(LAMS, C.smap_corr(w, qp, cp, R, LAMS)):
            out[f'smap{l}'].append(torch.where(any_, r0 + c, f).cpu())
        for arm, m in models.items():
            out[arm].append(torch.where(any_, r0 + corr_of(m, se, rows, t, b, R, w, qp, cp, H), f).cpu())
    return d, {k: torch.cat(v) for k, v in out.items()}


def run_test(a, dev):
    res = {}
    learned = a.arms.split(',')
    for H in [int(x) for x in a.Hs.split(',')]:
        dv, pv = predict('val', H, dev, learned)
        vs = {l: float(FC.smape_traj(pv[f'smap{l}'], pv['y'], dv['ids'], dv['inst']).median()) for l in LAMS}
        lam = min(vs, key=vs.get)
        dt, pt = predict('test', H, dev, learned)
        sm = {k: FC.smape_traj(pt[k], pt['y'], dt['ids'], dt['inst']).numpy() for k in ['fc', f'smap{lam}'] + learned}
        sm['smap'] = sm.pop(f'smap{lam}')
        torch.save({k: torch.from_numpy(v) for k, v in sm.items()}, OUT / f'inst_H{H}_s99.pt')
        rng = np.random.default_rng(0)
        r = dict(lam=lam, val_smap=vs, median={k: float(np.median(v)) for k, v in sm.items()}, cmp={})
        pairs = [(p, q) for p in learned for q in ('fc', 'smap')] + ([('pathm', 'pairm')] if len(learned) == 2 else [])
        for x, y in pairs:
            r['cmp'][f'{x}-{y}'] = C.boot(sm[x], sm[y], rng)
        res[f'H{H}'] = r
        print(f'[test] H{H} (smap lambda {lam}) test r-only sMAPE ' + ' '.join(f'{k} {v:.3f}' for k, v in r['median'].items()), flush=True)
        for k, (d0, lo, hi, fr) in r['cmp'].items():
            print(f'   {k:13s} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] frac {fr:.3f}', flush=True)
    for arm in learned:
        go = all(res[f'H{H}']['cmp'][f'{arm}-{q}'][2] < 0 for H in (128, 512) for q in ('fc', 'smap') if f'H{H}' in res)
        print(f'[verdict] {arm}: GO={go} (CI upper < 0 vs fc and smap at L128 and L512)', flush=True)
    if len(learned) == 2:
        pp = all(res[f'H{H}']['cmp']['pathm-pairm'][2] < 0 for H in (128, 512) if f'H{H}' in res)
        print(f'[verdict] path-m better than pair-m: {pp}', flush=True)
    json.dump(res, open(OUT / 'results_s99.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['train', 'test'])
    ap.add_argument('--arm', default='pairm'); ap.add_argument('--arms', default='pairm,pathm')
    ap.add_argument('--H', type=int, default=128); ap.add_argument('--Hs', default='128,256,512')
    ap.add_argument('--n_q', type=int, default=32000); ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--epochs', type=int, default=FC.EPOCHS); ap.add_argument('--bs', type=int, default=FC.BATCH)
    ap.add_argument('--max_steps', type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'train': run_train, 'test': run_test}[a.stage](a, torch.device('cuda'))
