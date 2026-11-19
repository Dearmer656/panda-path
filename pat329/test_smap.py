"""PAT-329 check: for a few val queries, smap (theta, lambda -> 0) equals weighted least squares with intercept on the
12 delay coordinates computed independently with numpy; library windows end <= t - H (no leakage)."""
import sys, numpy as np, torch
sys.path.insert(0, 'pat329')
import smap_classic as P
C, FC = P.C, P.FC
dev = torch.device('cuda')
d = torch.load(P.Path(P.os.environ['P321_OUT']) / 'val.pt'); S = d['S'].to(dev)
H = 128; rows, t = d['rows'][:4].to(dev), d['t'][:4].to(dev)
win, fut, valid, q = P.library(S, rows, t, H, dev)
for th in (0.0, 2.0):
    Xd, Xq = win[..., C.SUB], q[:, C.SUB]
    dist = (Xd - Xq[:, None]).norm(dim=-1).masked_fill(~valid, float('inf'))
    dbar = (dist.masked_fill(~valid, 0).sum(1) / valid.sum(1))[:, None]
    w = torch.exp(-th * dist / dbar).masked_fill(~valid, 0); w = w / w.sum(1, keepdim=True)
    got = ((w[..., None] * fut).sum(1) + C.smap_corr(w, q, win, fut, [1e-9])[0]).double().cpu().numpy()
    for b in range(4):
        v = valid[b].cpu().numpy(); Xb = Xd[b].double().cpu().numpy()[v]; Yb = fut[b].double().cpu().numpy()[v]; wb = w[b].double().cpu().numpy()[v]
        A = np.hstack([np.ones((len(Xb), 1)), Xb]) * np.sqrt(wb)[:, None]
        coef = np.linalg.lstsq(A, Yb * np.sqrt(wb)[:, None], rcond=None)[0]
        ref = np.concatenate([[1.0], Xq[b].double().cpu().numpy()]) @ coef
        err = np.abs(got[b] - ref).max() / np.abs(ref).max()
        print(f'theta {th} q{b} n_lib {v.sum()} rel err {err:.2e}'); assert err < 1e-2
print('ALL OK')
