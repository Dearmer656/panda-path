"""PAT-327: transform the retrieved continuations of FreshCast r-only (PAT-321, dysts), instead of only averaging them.

  r = sum_c w_c (R_c + g_c)          w_c: frozen PAT-321 fc scorer (scorer_fc_H{H}_s99.pt), R_c: its 400 readouts
Arms (g_c):
  fc    0                                                       r-only baseline (PAT-321)
  smap  S-map / local linear, training-free: weighted ridge (phi_c - phibar) -> (R_c - rbar) with weights w_c over the
        400 entries, correction (phi_q - phibar) B; phi = 12 delay coordinates (every 8th step of the 96-step window) in
        the readout's units; lambda (relative to tr/m) chosen on val r-only sMAPE from {0.01, 0.1, 1}
  pair  Head_H(MLP([phi_q - phi_c, z_c]))                       per (query, candidate) transform, learned
  path  Head_H(M_{t<-s_c} u_c), u_c = MLP([phi_c, z_c]); M = A_0 A_1 ... A_{n_c - 1} over the n_c = (t - s_c) // 16
        observed 16-step patches between the candidate end s_c and t (A_0 nearest to t),
        A_k = gamma_k (I - beta_k w_k w_k^T), gamma_k = exp(0.05 tanh(.)) (may expand), beta_k = 2 sigmoid(. - 4)
        (starts near identity), w_k / beta_k / gamma_k from the patch's context-standardised values. No query input.
Head is zero-initialised: training starts exactly at r-only. Units: context-standardised (as the official metric).
Training as PAT-321 fc (32000 queries, 30 epochs, AdamW 2e-3, wd 1e-2, EMA 0.995, clipped normalised MSE);
the EMA state at the epoch with the lowest val loss (held-out TRAIN systems) is kept.
usage: python pat327/fc_corr.py --stage train --arm pair|path --H 128
       python pat327/fc_corr.py --stage test"""
import argparse, copy, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('P321_OUT', str(ROOT / 'pat321' / 'data'))
sys.path.insert(0, str(ROOT / 'pat321'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fc_dysts as FC  # noqa: E402  (data, candidates, readouts, scorer, metric; unchanged)

OUT = ROOT / 'pat327' / 'out'
LQ, PATCH, NZ = FC.LQ, FC.PATCH, FC.NZ
SUB = torch.arange(LQ - 1, -1, -8).flip(0)          # 12 delay coordinates, last one = most recent step
D, HID, NPMAX = 32, 128, (FC.T_LEN - LQ) // PATCH + 1
LAMS = (0.01, 0.1, 1.0)


def load_fc(H, dev):
    m = FC.Scorer('fc').to(dev)
    m.load_state_dict(torch.load(Path(os.environ['P321_OUT']) / f'scorer_fc_H{H}_s99.pt')['state']); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def phis(se, rows, t, b):
    """96-step windows in the readout's units: query [B, LQ]; entries [B, 2N, LQ] (value entries raw, shape entries mapped
    to the query's window mean / sd, as the readouts are); both context-standardised."""
    S, dev = se.S, se.S.device
    kq = torch.arange(-LQ, 0, device=dev)
    cm, cs = b['cm'][:, None], b['cs'][:, None]
    q = (S[rows[:, None], t[:, None] + kq] - cm) / cs
    s = b['s']
    x = S[rows[:, None, None], s[..., None] + kq]
    mc, sc = se.mu[rows[:, None], s][..., None], se.sd[rows[:, None], s].clamp_min(1e-10)[..., None]
    mq, sq = se.mu[rows, t][:, None, None], se.sd[rows, t][:, None, None]
    x = torch.where(b['val'][..., None], x, mq + sq / sc * (x - mc))
    c = (x - cm[..., None]) / cs[..., None]
    return q.clamp(-FC.CLIP, FC.CLIP), c.clamp(-FC.CLIP, FC.CLIP)


def patches(se, rows, t, b):
    """observed 16-step patches back from t, context-standardised: [B, NPMAX, 16] (patch k = [t-16(k+1), t-16k)), valid mask."""
    S, dev = se.S, se.S.device
    k = torch.arange(NPMAX, device=dev)
    st = t[:, None] - PATCH * (k[None] + 1)
    ok = st >= 0
    idx = st.clamp_min(0)[..., None] + torch.arange(PATCH, device=dev)
    P = (S[rows[:, None, None], idx] - b['cm'][:, None, None]) / b['cs'][:, None, None]
    return P.clamp(-FC.CLIP, FC.CLIP) * ok[..., None], ok


class Corr(nn.Module):
    def __init__(self, arm, H):
        super().__init__()
        self.arm = arm
        din = LQ + NZ
        dout = HID if arm == 'pair' else D
        self.enc = nn.Sequential(nn.Linear(din, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU(), nn.Linear(HID, dout))
        self.head = nn.Linear(dout, H)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        if arm == 'path':
            self.pe = nn.Sequential(nn.Linear(PATCH, 64), nn.GELU(), nn.Linear(64, D + 2))

    def transport(self, P, ok, n):
        """M_n u for every entry: M_n = A_0 ... A_{n-1}; returns stacked M [B, nmax+1, D, D]."""
        o = self.pe(P)                                                                  # [B, NP, D + 2]
        w = F.normalize(o[..., :D], dim=-1)
        beta = 2 * torch.sigmoid(o[..., D] - 4) * ok
        gam = torch.exp(0.05 * torch.tanh(o[..., D + 1])) * ok + (~ok).float()
        B = P.shape[0]; nmax = int(n.max())
        I = torch.eye(D, device=P.device).expand(B, D, D)
        Ms = [I]; M = I
        for k in range(nmax):
            A = gam[:, k, None, None] * (I - beta[:, k, None, None] * w[:, k, :, None] * w[:, k, None, :])
            M = M @ A; Ms.append(M)
        return torch.stack(Ms, 1)

    def forward(self, w, zn, qp, cp, t=None, s=None, P=None, ok=None):
        """sum_c w_c g_c  [B, H]"""
        if self.arm == 'pair':
            h = self.enc(torch.cat([qp[:, None] - cp, zn], -1))                          # [B, 2N, HID]
        else:
            u = self.enc(torch.cat([cp, zn], -1))                                         # [B, 2N, D]
            n = ((t[:, None] - s) // PATCH).clamp(0, NPMAX - 1)                           # [B, 2N]
            Ms = self.transport(P, ok, n)
            M = Ms.gather(1, n[..., None, None].expand(-1, -1, D, D))                   # [B, 2N, D, D]
            h = (M @ u[..., None])[..., 0]
        return self.head((w[..., None] * h).sum(1))                                       # head linear: sum_c w_c Head(h_c)


def smap_corr(w, qp, cp, R, lam):
    """weighted local-linear (S-map) correction [B, H] for each lam in lam list -> list."""
    Xq, X = qp[:, SUB], cp[..., SUB]                                                     # [B, m], [B, 2N, m]
    xb = (w[..., None] * X).sum(1); rb = (w[..., None] * R).sum(1)
    Xc, Yc = X - xb[:, None], R - rb[:, None]
    A = (Xc * w[..., None]).transpose(1, 2) @ Xc                                         # [B, m, m]
    XY = (Xc * w[..., None]).transpose(1, 2) @ Yc                                        # [B, m, H]
    m = A.shape[-1]; tr = A.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-8) / m
    I = torch.eye(m, device=A.device)
    out = []
    for l in lam:
        Bm = torch.linalg.solve(A.double() + (l * tr)[:, None, None].double() * I, XY.double()).float()
        out.append(((Xq - xb)[:, None] @ Bm)[:, 0])
    return out


def step_inputs(se, d, idx, H, dev, fcm):
    rows, t = d['rows'][idx].to(dev), d['t'][idx].to(dev)
    b = FC.build(se, rows, t, H)
    R = FC.readout(se, rows, t, b, H)
    with torch.no_grad():
        w = fcm(b)
    w = torch.nan_to_num(w, nan=0.0)
    zn = (b['z'] - fcm.zm) / fcm.zs
    qp, cp = phis(se, rows, t, b)
    return rows, t, b, R, w, zn, qp, cp


def corr_of(model, se, rows, t, b, w, zn, qp, cp):
    if model.arm == 'path':
        P, ok = patches(se, rows, t, b)
        return model(w, zn, qp, cp, t=t, s=b['s'], P=P, ok=ok)
    return model(w, zn, qp, cp)


@torch.no_grad()
def val_loss(model, sv, dv, vsel, veps, H, dev, fcm, bs=256):
    model.eval(); tot = 0.0
    for i in range(0, len(vsel), bs):
        idx = vsel[i:i + bs]
        rows, t, b, R, w, zn, qp, cp = step_inputs(sv, dv, idx, H, dev, fcm)
        y = FC.target(sv, rows, t, b, H)
        r = (w[..., None] * R).sum(1) + corr_of(model, sv, rows, t, b, w, zn, qp, cp)
        tot += float(((r - y).square().mean(1) / veps[i:i + bs].to(dev)).clamp_max(FC.C_MAX).sum())
    model.train()
    return tot / len(vsel)


def run_train(a, dev):
    H, arm = a.H, a.arm
    fcm = load_fc(H, dev)
    d, se = FC.load_set('train', dev)
    g = torch.Generator().manual_seed(a.seed)
    sel = torch.randperm(len(d['rows']), generator=g)[:a.n_q]                             # same draw as PAT-321
    sel, eps, _, _ = FC.untrained_eps(se, d, sel, H, dev)
    dv, sv = FC.load_set('val', dev)
    vsel, veps, _, _ = FC.untrained_eps(sv, dv, torch.arange(len(dv['rows'])), H, dev)
    torch.manual_seed(a.seed)
    model = Corr(arm, H).to(dev); ema = copy.deepcopy(model)
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
            rows, t, b, R, w, zn, qp, cp = step_inputs(se, d, idx, H, dev, fcm)
            y = FC.target(se, rows, t, b, H)
            r = (w[..., None] * R).sum(1) + corr_of(model, se, rows, t, b, w, zn, qp, cp)
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
    torch.save(dict(state=best[2], hist=hist, best_val=best[0], best_ep=best[1], v0=v0), OUT / f'corr_{arm}_H{H}_s{a.seed}.pt')


@torch.no_grad()
def predict(name, H, dev, arms, bs=128):
    """r-only forecasts of every arm (smap: one per lambda) + target, context-standardised."""
    fcm = load_fc(H, dev)
    d, se = FC.load_set(name, dev)
    models = {}
    for arm in arms:
        if arm in ('pair', 'path'):
            m = Corr(arm, H).to(dev); m.load_state_dict(torch.load(OUT / f'corr_{arm}_H{H}_s99.pt')['state']); m.eval(); models[arm] = m
    out = {k: [] for k in ['fc', 'y'] + [f'smap{l}' for l in LAMS] + list(models)}
    for i in range(0, len(d['rows']), bs):
        idx = torch.arange(i, min(i + bs, len(d['rows'])))
        rows, t, b, R, w, zn, qp, cp = step_inputs(se, d, idx, H, dev, fcm)
        f = (d['f'][idx, :H].to(dev) - b['cm'][:, None]) / b['cs'][:, None]
        any_ = b['any'][:, None]
        r0 = (w[..., None] * R).sum(1)
        out['fc'].append(torch.where(any_, r0, f).cpu()); out['y'].append(FC.target(se, rows, t, b, H).cpu())
        for l, c in zip(LAMS, smap_corr(w, qp, cp, R, LAMS)):
            out[f'smap{l}'].append(torch.where(any_, r0 + c, f).cpu())
        for arm, m in models.items():
            out[arm].append(torch.where(any_, r0 + corr_of(m, se, rows, t, b, w, zn, qp, cp), f).cpu())
    return d, {k: torch.cat(v) for k, v in out.items()}


def boot(x, y, rng, B=2000):
    n = len(x); d0 = float(np.median(x) - np.median(y))
    idx = rng.integers(0, n, (B, n))
    ds = np.median(x[idx], 1) - np.median(y[idx], 1)
    return d0, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float((x < y).mean())


def run_test(a, dev):
    res = {}
    learned = [x for x in a.arms.split(',') if x in ('pair', 'path')]
    for H in [int(x) for x in a.Hs.split(',')]:
        dv, pv = predict('val', H, dev, learned)
        vs = {l: float(FC.smape_traj(pv[f'smap{l}'], pv['y'], dv['ids'], dv['inst']).median()) for l in LAMS}
        lam = min(vs, key=vs.get)
        print(f'[test] H{H} val r-only sMAPE fc {FC.smape_traj(pv["fc"], pv["y"], dv["ids"], dv["inst"]).median():.3f} '
              + ' '.join(f'smap{l} {v:.3f}' for l, v in vs.items()) + f' -> lambda {lam}', flush=True)
        dt, pt = predict('test', H, dev, learned)
        sm = {k: FC.smape_traj(pt[k], pt['y'], dt['ids'], dt['inst']).numpy() for k in ['fc', f'smap{lam}'] + learned}
        sm['smap'] = sm.pop(f'smap{lam}')
        torch.save({k: torch.from_numpy(v) for k, v in sm.items()}, OUT / f'inst_H{H}_s99.pt')
        rng = np.random.default_rng(0)
        r = dict(lam=lam, val_smap=vs, median={k: float(np.median(v)) for k, v in sm.items()}, cmp={})
        for x, y in [(p, q) for p in learned for q in ('fc', 'smap')] + [('smap', 'fc')] + ([('path', 'pair')] if len(learned) == 2 else []):
            r['cmp'][f'{x}-{y}'] = boot(sm[x], sm[y], rng)
        res[f'H{H}'] = r
        print(f'[test] H{H} test r-only sMAPE ' + ' '.join(f'{k} {v:.3f}' for k, v in r['median'].items()), flush=True)
        for k, (d0, lo, hi, fr) in r['cmp'].items():
            print(f'   {k:12s} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] frac {fr:.3f}', flush=True)
    for arm in learned:
        go = all(res[f'H{H}']['cmp'][f'{arm}-{q}'][2] < 0 for H in (128, 512) for q in ('fc', 'smap') if f'H{H}' in res)
        print(f'[verdict] {arm}: GO={go} (CI upper < 0 vs fc and smap at L128 and L512)', flush=True)
    if len(learned) == 2:
        pp = all(res[f'H{H}']['cmp']['path-pair'][2] < 0 for H in (128, 512) if f'H{H}' in res)
        print(f'[verdict] path better than pair: {pp}', flush=True)
    json.dump(res, open(OUT / f'results_s99{a.out_tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['train', 'test'])
    ap.add_argument('--arm', default='pair'); ap.add_argument('--arms', default='pair,path')
    ap.add_argument('--H', type=int, default=128); ap.add_argument('--Hs', default='128,256,512')
    ap.add_argument('--n_q', type=int, default=32000); ap.add_argument('--seed', type=int, default=99)
    ap.add_argument('--epochs', type=int, default=FC.EPOCHS); ap.add_argument('--bs', type=int, default=FC.BATCH)
    ap.add_argument('--max_steps', type=int, default=0); ap.add_argument('--out_tag', default='')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {'train': run_train, 'test': run_test}[a.stage](a, torch.device('cuda'))
