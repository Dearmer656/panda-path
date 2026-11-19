"""PAT-329 top-k: keep only the k largest FreshCast weights (renormalised) of the 400 entries; r-top-k (weighted mean)
and smap-top-k (S-map local-linear correction on those k entries, lambda grid). k in {1, 5, 20, 50, all}.
Run per LQ (env P321_LQ / P321_OUT as lq_scan.py); writes per-config val medians and per-trajectory test sMAPE.
usage: P321_LQ=32 P321_OUT=pat329/lq32 python pat329/topk.py --H 128"""
import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--H', type=int, default=128); ap.add_argument('--bs', type=int, default=128)
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327'))
import torch
import fc_corr as C  # noqa: E402
FC = C.FC
LQ = FC.LQ; dev = torch.device('cuda'); H = a.H
KS = (1, 5, 20, 50, 400)


@torch.no_grad()
def predict(name):
    fcm = C.load_fc(H, dev)
    d, se = FC.load_set(name, dev)
    out = {}; ys = []
    for i in range(0, len(d['rows']), a.bs):
        idx = torch.arange(i, min(i + a.bs, len(d['rows'])))
        rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
        f = (d['f'][idx, :H].to(dev) - b['cm'][:, None]) / b['cs'][:, None]
        has = b['any'][:, None]
        ys.append(FC.target(se, rows, t, b, H).cpu())
        for k in KS:
            kk = min(k, w.shape[1])
            top = w.topk(kk, 1).indices
            wk = torch.zeros_like(w).scatter_(1, top, w.gather(1, top))
            wk = wk / wk.sum(1, keepdim=True).clamp_min(1e-30)
            r = (wk[..., None] * R).sum(1)
            out.setdefault(f'r_k{k}', []).append(torch.where(has, r, f).cpu())
            if k >= 5:
                for l, c in zip(C.LAMS, C.smap_corr(wk, qp, cp, R, C.LAMS)):
                    out.setdefault(f'smap_k{k}_l{l}', []).append(torch.where(has, r + c, f).cpu())
    return d, {k: torch.cat(v) for k, v in out.items()}, torch.cat(ys)


dv, pv, yv = predict('val')
val = {k: float(FC.smape_traj(v, yv, dv['ids'], dv['inst']).median()) for k, v in pv.items()}
dt, pt, yt = predict('test')
test = {k: FC.smape_traj(v, yt, dt['ids'], dt['inst']) for k, v in pt.items()}
best = min(val, key=val.get)
print(f'[topk] LQ {LQ} H{H} val best {best} {val[best]:.3f}', flush=True)
for k in sorted(val, key=val.get)[:8]:
    print(f'   {k:16s} val {val[k]:.3f} test {float(test[k].median()):.3f}', flush=True)
out = ROOT / 'pat329' / 'out'
torch.save(test, out / f'topk_lq{LQ}_H{H}.pt')
json.dump(dict(LQ=LQ, H=H, val=val, test={k: float(v.median()) for k, v in test.items()}), open(out / f'topk_lq{LQ}_H{H}.json', 'w'), indent=1)
