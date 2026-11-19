"""PAT-327 checks: (1) zero-init head -> r == fc r-only exactly; (2) path transport M_n u equals the naive product
A_0 ... A_{n-1} u for random entries; (3) S-map correction -> 0 as lambda -> inf, and exact for an affine toy;
(4) phis: value entries equal the raw window, shape entries have the query window's mean / sd."""
import sys, torch
sys.path.insert(0, 'pat327')
import fc_corr as C
FC = C.FC
dev = torch.device('cuda'); torch.manual_seed(0)
d, se = FC.load_set('val', dev)
H = 128; fcm = C.load_fc(H, dev)
idx = torch.arange(0, 64)
rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
r0 = (w[..., None] * R).sum(1)
for arm in ('pair', 'path'):
    m = C.Corr(arm, H).to(dev)
    g = C.corr_of(m, se, rows, t, b, w, zn, qp, cp)
    print(arm, 'zero-init |g| max', float(g.abs().max())); assert float(g.abs().max()) == 0.0
# (2) transport
m = C.Corr('path', H).to(dev)
for p in m.pe.parameters(): torch.nn.init.normal_(p, std=0.5)
P, ok = C.patches(se, rows, t, b)
n = ((t[:, None] - b['s']) // C.PATCH).clamp(0, C.NPMAX - 1)
Ms = m.transport(P, ok, n)
o = m.pe(P); wv = torch.nn.functional.normalize(o[..., :C.D], dim=-1)
beta = 2 * torch.sigmoid(o[..., C.D] - 4) * ok; gam = torch.exp(0.05 * torch.tanh(o[..., C.D + 1])) * ok + (~ok).float()
u = torch.randn(C.D, device=dev)
for bi, ci in [(0, 0), (3, 17), (10, 250), (63, 399)]:
    k = int(n[bi, ci]); v = u.clone()
    for j in reversed(range(k)):   # A_{k-1} applied first
        v = gam[bi, j] * (v - beta[bi, j] * (wv[bi, j] @ v) * wv[bi, j])
    err = float((Ms[bi, k] @ u - v).abs().max() / v.abs().max().clamp_min(1e-12))
    print(f'transport b{bi} c{ci} n={k} rel err {err:.2e}'); assert err < 1e-4
# patch content: patch 0 = last 16 observed steps
cm, cs = b['cm'], b['cs']
p0 = (se.S[rows[0], t[0] - 16:t[0]] - cm[0]) / cs[0]
assert torch.allclose(P[0, 0], p0.clamp(-FC.CLIP, FC.CLIP)), 'patch 0 mismatch'
# (3) smap
big = C.smap_corr(w, qp, cp, R, [1e8])[0]
print('smap lambda 1e8 |corr| max', float(big.abs().max())); assert float(big.abs().max()) < 1e-3
Bt = torch.randn(12, H, device=dev)
Rt = (cp[..., C.SUB] - 0.3) @ Bt
c = C.smap_corr(w, qp, cp, Rt, [1e-9])[0]
exact = (qp[:, C.SUB] - 0.3) @ Bt
r_s = (w[..., None] * Rt).sum(1) + c
okq = b['any'] & ((w > 1e-4).sum(1) >= 14)
err = float(((r_s - exact)[okq]).abs().max() / exact[okq].abs().max())
print('smap affine toy rel err', err, 'on', int(okq.sum()), 'queries'); assert err < 1e-3
# (4) phis
qi, ci = 5, 3
v_raw = (se.S[rows[qi], b['s'][qi, ci] - 96:b['s'][qi, ci]] - cm[qi]) / cs[qi]
assert bool(b['val'][qi, ci]) and torch.allclose(cp[qi, ci], v_raw, atol=1e-5)
N = b['s'].shape[1] // 2
sh = cp[qi, N + ci]; qv = qp[qi]
print('shape entry mean/sd', float(sh.mean()), float(sh.std(unbiased=False)), 'query', float(qv.mean()), float(qv.std(unbiased=False)))
assert abs(float(sh.mean() - qv.mean())) < 1e-3 and abs(float(sh.std(unbiased=False) - qv.std(unbiased=False))) < 1e-3
print('ALL OK')
