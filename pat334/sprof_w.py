"""PAT-334 S follow-up (criteria fixed before results). Reads out/sprof_{TAG}_H{H}.pt.
Correction: sprof.py clustered the bootstrap on `ids` = instances (one origin t, all channels), not trajectories; each
trajectory has ~6 origins. Here clusters = trajectories (inst[ids][0]) for everything.
  S1-S3 recomputed with trajectory clusters (same statistics as sprof.py).
  Within-trajectory (x and y demeaned by their trajectory mean before ranking; removes system / parameter / trajectory level):
  W1 : OLS fitted on val (demeaned), test Spearman(pred, err_f) profile minus best : CI > 0
  W2 : Spearman(lam, vpt_n f) : CI < 0
  W3 : Spearman(mis, chg) : CI > 0        (ctx2048, all H; ctx512 reported)
usage: same env as mh.py + P334_H"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
from mh import FC, TAG
from sprof import boot, ci, O


def traj(name, ids):
    d = torch.load(FC.OUT / f'{name}.pt'); tr = torch.tensor([i[0] for i in d['inst']]); return tr[ids]


def dm(x, g):
    g = torch.unique(g, return_inverse=True)[1]; n = int(g.max()) + 1; x = x.double()
    s = torch.zeros(n, dtype=torch.float64).index_add_(0, g, x); c = torch.zeros(n, dtype=torch.float64).index_add_(0, g, torch.ones_like(x))
    return x - (s / c)[g]


def run(H):
    P = torch.load(O / f'sprof_{TAG}_H{H}.pt', weights_only=False); V, T = P['V'], P['T']
    gV, gT = traj('val', V['ids']), traj('test', T['ids'])
    print(f'== {TAG} H{H}: test trajectories {len(gT.unique())}, origins/trajectory {len(T["ids"].unique()) / len(gT.unique()):.1f}', flush=True)
    for lab, tf in (('pooled', lambda x, g: x.double()), ('within', dm)):
        for tgt in ('err_f', 'err_M2'):
            okV = torch.isfinite(V['best']) & torch.isfinite(V['ld']).all(1) & torch.isfinite(V[tgt]); okT = torch.isfinite(T['best']) & torch.isfinite(T['ld']).all(1) & torch.isfinite(T[tgt])
            def X(D, ok, g, cols): return torch.cat([torch.ones(int(ok.sum()), 1, dtype=torch.float64)] + [torch.stack([tf(c, g[ok]) for c in (D[k][ok].reshape(int(ok.sum()), -1).T)], 1) for k in cols], 1)
            yv, yt = tf(V[tgt][okV], gV[okV]), tf(T[tgt][okT], gT[okT])
            ba = torch.linalg.lstsq(X(V, okV, gV, ['best']), yv[:, None]).solution; bb = torch.linalg.lstsq(X(V, okV, gV, ['ld']), yv[:, None]).solution
            pa, pb = (X(T, okT, gT, ['best']) @ ba)[:, 0], (X(T, okT, gT, ['ld']) @ bb)[:, 0]
            pt, R = boot([(pa, yt), (pb, yt)], gT[okT]); lo, hi = ci(R[1] - R[0])
            print(f'   {lab} S1 {tgt:6s}: best {pt[0]:.4f} profile {pt[1]:.4f} diff {pt[1] - pt[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]', flush=True)
        m = torch.isfinite(T['lam']); g = gT[m]
        pt, R = boot([(tf(T['lam'][m], g), tf(T['vpt_f'][m], g)), (tf(T['lam'][m], g), tf(T['err_f'][m], g))], g)
        print(f'   {lab} S2 (lam, vpt_n f) {pt[0]:+.4f} [{ci(R[0])[0]:+.4f}, {ci(R[0])[1]:+.4f}] | (lam, err_f) {pt[1]:+.4f} [{ci(R[1])[0]:+.4f}, {ci(R[1])[1]:+.4f}]', flush=True)
        m = torch.isfinite(T['mis']) & torch.isfinite(T['best']); g = gT[m]
        pt, R = boot([(tf(T['mis'][m], g), tf(T['chg'][m], g))], g)
        print(f'   {lab} S3 (mis, chg) {pt[0]:+.4f} [{ci(R[0])[0]:+.4f}, {ci(R[0])[1]:+.4f}]', flush=True)


if __name__ == '__main__':
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else (128, 256, 512)): run(H)
