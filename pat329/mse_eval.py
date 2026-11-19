"""PAT-329 MSE check (ctx 512, official windows): per-trajectory MSE (context-standardised; window = mean over steps and
channels, trajectory = mean over its windows, report median over trajectories) and amplitude ratio std(pred)/std(y),
for methods with hyperparameters already chosen on val. One process per LQ (module constant); writes per-trajectory arrays.
usage: P321_LQ=96 P321_OUT=pat321/data python pat329/mse_eval.py"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327'))
import torch
import fc_corr as C  # noqa: E402
FC = C.FC
LQ = FC.LQ; dev = torch.device('cuda')
CTX = FC.CTX
# hyperparameters chosen on val earlier (PAT-327 / ctx_scan / parrot.py), per backbone context
LAM96 = {512: {128: 0.01, 256: 0.01, 512: 0.1}, 1024: {128: 0.01, 256: 0.1, 512: 0.1}, 2048: {128: 0.01, 256: 0.1, 512: 0.1}}[CTX]
D_FULL = {512: {128: 32, 256: 64, 512: 64}, 1024: {128: 64, 256: 64, 512: 64}, 2048: {128: 32, 256: 64, 512: 64}}[CTX]
D_CTX = {512: 8, 1024: 8, 2048: 16}[CTX]
ONLY = os.environ.get('P329_ONLY')                          # 'panda': backbone only (e.g. the RoPE data dir)
TAG = '' if CTX == 512 else f'_ctx{CTX}' + (f'_{ONLY}' if ONLY else '')
PLAN = {96: {H: (['panda'] if ONLY else ['r_all', 'r_top1', f'smap_{LAM96[H]}', 'panda', 'parrot_full', 'parrot_ctx']) for H in (128, 256, 512)},
        32: {128: ['smap_0.01'], 256: [], 512: ['r_top1']},
        64: {128: [], 256: ['smap_0.01'], 512: ['smap_0.1']}}[LQ]


def per_traj(err, ids, inst):
    """err [nq] per query -> mean over the queries of a window, mean over windows of a trajectory."""
    n = len(inst)
    s = torch.zeros(n, dtype=torch.float64).index_add_(0, ids, err.double())
    c = torch.zeros(n, dtype=torch.float64).index_add_(0, ids, torch.ones_like(err, dtype=torch.float64))
    w = s / c
    traj = torch.tensor([j for j, _ in inst]); nt = int(traj.max()) + 1
    st = torch.zeros(nt, dtype=torch.float64).index_add_(0, traj, w)
    ct = torch.zeros(nt, dtype=torch.float64).index_add_(0, traj, torch.ones_like(w))
    return (st / ct)[ct > 0]


@torch.no_grad()
def parrot(S, rows, t, Z, D, H, full):
    B = len(rows); ar = torch.arange(B, device=dev)
    q = Z[ar[:, None], t[:, None] + torch.arange(-D, 0, device=dev)]
    s = torch.arange(D, int(t.max()), device=dev)
    sc = s[None].expand(B, -1).clamp(max=(t - 1)[:, None])
    win = Z[ar[:, None, None], sc[..., None] + torch.arange(-D, 0, device=dev)]
    dist = (win - q[:, None]).square().sum(-1)
    ok = (s[None] + H <= t[:, None]) if full else ((s[None] - D >= (t - FC.CTX)[:, None]) & (s[None] <= (t - 1)[:, None]))
    dm = dist.masked_fill(~ok, float('inf')); best = dm.argmin(1); has = torch.isfinite(dm.min(1).values)
    sb = s[best]; h = torch.arange(H, device=dev)[None]
    idx = sb[:, None] + h if full else sb[:, None] + (h % (t - sb).clamp_min(1)[:, None])
    return Z[ar[:, None], idx.clamp(max=Z.shape[1] - 1)], has


@torch.no_grad()
def run(H, methods):
    fcm = C.load_fc(H, dev)
    d, se = FC.load_set('test', dev)
    acc = {m: dict(mse=[], amp=[], smape=[]) for m in methods}
    for i in range(0, len(d['rows']), 64):
        idx = torch.arange(i, min(i + 64, len(d['rows'])))
        rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
        y = FC.target(se, rows, t, b, H)
        f = (d['f'][idx, :H].to(dev) - b['cm'][:, None]) / b['cs'][:, None]
        has = b['any'][:, None]
        Z = (se.S[rows] - b['cm'][:, None]) / b['cs'][:, None]
        preds = {}
        if 'r_all' in methods: preds['r_all'] = torch.where(has, (w[..., None] * R).sum(1), f)
        if 'r_top1' in methods: preds['r_top1'] = torch.where(has, R.gather(1, w.argmax(1)[:, None, None].expand(-1, 1, H))[:, 0], f)
        for m in methods:
            if m.startswith('smap_'):
                lam = float(m.split('_')[1]); r0 = (w[..., None] * R).sum(1)
                preds[m] = torch.where(has, r0 + C.smap_corr(w, qp, cp, R, [lam])[0], f)
        if 'panda' in methods: preds['panda'] = f
        for name, full, D in (('parrot_full', True, D_FULL[H]), ('parrot_ctx', False, D_CTX)):
            if name in methods:
                p, ok = parrot(se.S, rows, t, Z, D, H, full); preds[name] = torch.where(ok[:, None], p, f)
        for m, p in preds.items():
            acc[m]['mse'].append((p - y).square().mean(1).cpu())
            acc[m]['amp'].append((p.std(1) / y.std(1).clamp_min(1e-8)).cpu())
            acc[m]['smape'].append(((p - y).abs() / (p.abs() + y.abs() + 1e-10)).mean(1).cpu())
    out = {}
    for m, v in acc.items():
        out[f'{m}@LQ{LQ}'] = {k: per_traj(torch.cat(x), d['ids'], d['inst']) for k, x in v.items()}
        print(f'[mse] H{H} {m}@LQ{LQ}: MSE {float(out[f"{m}@LQ{LQ}"]["mse"].median()):.4f} amp {float(torch.cat(v["amp"]).median()):.3f} '
              f'sMAPE {100 * float(out[f"{m}@LQ{LQ}"]["smape"].median()):.3f}', flush=True)
    torch.save(out, ROOT / 'pat329' / 'out' / f'mse{TAG}_lq{LQ}_H{H}.pt')


if __name__ == '__main__':   # stride.py imports per_traj only
    for H, methods in PLAN.items():
        if methods: run(H, methods)
