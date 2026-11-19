"""PAT-329 S-map coordinate density (ctx 512): delay-coordinate stride in {1, 2, 4, 8} of the LQ window (8 = PAT-327 default
for LQ 96), lambda in {0.01, 0.1, 1, 10} chosen on val per stride (sMAPE); test sMAPE and MSE per trajectory.
usage: P321_LQ=96 P321_OUT=pat321/data python pat329/stride.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327')); sys.path.insert(0, str(ROOT / 'pat329'))
import torch
import fc_corr as C  # noqa: E402
import mse_eval as ME  # noqa: E402  (per_traj)
FC = C.FC; LQ = FC.LQ; dev = torch.device('cuda')
STRIDES, LAMS = (1, 2, 4, 8), (0.01, 0.1, 1.0, 10.0)


@torch.no_grad()
def predict(name, H):
    fcm = C.load_fc(H, dev); d, se = FC.load_set(name, dev)
    sm, ms = {}, {}
    for i in range(0, len(d['rows']), 64):
        idx = torch.arange(i, min(i + 64, len(d['rows'])))
        rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
        y = FC.target(se, rows, t, b, H); f = (d['f'][idx, :H].to(dev) - b['cm'][:, None]) / b['cs'][:, None]
        has = b['any'][:, None]; r0 = (w[..., None] * R).sum(1)
        for st in STRIDES:
            C.SUB = torch.arange(LQ - 1, -1, -st).flip(0)
            for l, c in zip(LAMS, C.smap_corr(w, qp, cp, R, LAMS)):
                p = torch.where(has, r0 + c, f); k = f's{st}_l{l}'
                sm.setdefault(k, []).append(((p - y).abs() / (p.abs() + y.abs() + 1e-10)).mean(1).cpu())
                ms.setdefault(k, []).append((p - y).square().mean(1).cpu())
    return d, {k: ME.per_traj(torch.cat(v), d['ids'], d['inst']) for k, v in sm.items()}, {k: ME.per_traj(torch.cat(v), d['ids'], d['inst']) for k, v in ms.items()}


for H in (128, 256, 512):
    _, vs, _ = predict('val', H)
    _, ts, tm = predict('test', H)
    out = {}
    for st in STRIDES:
        lam = min(LAMS, key=lambda l: float(vs[f's{st}_l{l}'].median()))
        k = f's{st}_l{lam}'; out[st] = dict(lam=lam, smape=ts[k], mse=tm[k])
        print(f'[stride] LQ {LQ} H{H} stride {st} ({len(range(LQ - 1, -1, -st))} coords) lambda {lam}: test sMAPE {100 * float(ts[k].median()):.3f} MSE {float(tm[k].median()):.4f}', flush=True)
    torch.save(out, ROOT / 'pat329' / 'out' / f'stride_lq{LQ}_H{H}.pt')
