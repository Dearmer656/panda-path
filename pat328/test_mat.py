"""PAT-328 checks: (1) zero-init Dec -> correction exactly 0 (r == r-only); (2) path-m: step i_k = t-H-16k, its window
ends at i_k and its future y(i_k+1 : i_k+H) ends at or before t (no leakage); (3) M_j (gathered prefix product) equals the
naive A_0 ... A_{n_j-1} with n_j = floor((t-H-j)/16)+1, applied to E y+_j; (4) pair-m (A_j - I) e equals the explicit
matrix product; (5) the transform never reads the retrieval descriptors (forward signature has no z)."""
import inspect, sys, torch
sys.path.insert(0, 'pat328')
import fc_mat as M
C, FC = M.C, M.FC
dev = torch.device('cuda'); torch.manual_seed(0)
d, se = FC.load_set('val', dev)
for H in (128, 512):
    fcm = C.load_fc(H, dev)
    idx = torch.arange(0, 48)
    rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
    for arm in ('pairm', 'pathm'):
        m = M.Mat(arm, H).to(dev)
        g = M.corr_of(m, se, rows, t, b, R, w, qp, cp, H)
        assert float(g.abs().max()) == 0.0, arm
    print(f'H{H} zero-init ok')
    (xs, ys, ok), n = M.path_steps(se, rows, t, b, H)
    cm, cs = b['cm'], b['cs']
    vb = [i for i in range(len(idx)) if bool(ok[i, 0])]
    print(f'H{H}: {len(vb)}/{len(idx)} queries have at least one valid step')
    for bi, k in [(vb[0], 0), (vb[len(vb) // 2], 1), (vb[-1], int(ok[vb[-1]].sum()) - 1)]:
        assert bool(ok[bi, k])
        i = int(t[bi]) - H - 16 * k
        assert i + H <= int(t[bi])
        xw = (se.S[rows[bi], i - 96:i] - cm[bi]) / cs[bi]; yw = (se.S[rows[bi], i:i + H] - cm[bi]) / cs[bi]
        assert torch.allclose(xs[bi, k], xw.clamp(-FC.CLIP, FC.CLIP)) and torch.allclose(ys[bi, k], yw.clamp(-FC.CLIP, FC.CLIP)), (bi, k)
    print(f'H{H} step windows / futures ok, no leakage')
    m = M.Mat('pathm', H).to(dev)
    for p in m.net.parameters(): torch.nn.init.normal_(p, std=0.3)
    torch.nn.init.normal_(m.dec.weight, std=0.1)
    U, V = m.uv(qp[:, None].expand_as(xs), xs, ys)
    I = torch.eye(M.D, device=dev)
    A = torch.where(ok[..., None, None], I + U @ V.transpose(-1, -2), I)
    e = m.E(R)
    full = m(w, R, qp, cp, steps=(xs, ys, ok), n=n)
    v_naive = torch.zeros_like(e)
    for bi in range(len(idx)):
        for ci in range(e.shape[1]):
            j = int(b['s'][bi, ci]); nj = min(max((int(t[bi]) - H - j) // 16 + 1, 1), xs.shape[1])
            assert nj == int(n[bi, ci]), (bi, ci, nj, int(n[bi, ci]))
            if ci % 37 == 0:
                vv = e[bi, ci]
                for kk in reversed(range(nj)): vv = A[bi, kk] @ vv      # A_{n-1} first, A_0 last
                v_naive[bi, ci] = vv - e[bi, ci]
    sel = torch.zeros_like(w, dtype=torch.bool); sel[:, ::37] = True
    wsel = torch.where(sel, w, torch.zeros_like(w))
    ref = m.dec((wsel[..., None] * v_naive).sum(1))
    got = m(wsel, R, qp, cp, steps=(xs, ys, ok), n=n)
    err = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
    print(f'H{H} path-m prefix product vs naive rel err {err:.2e}, max n {int(n.max())}'); assert err < 1e-4
    mp = M.Mat('pairm', H).to(dev)
    for p in mp.net.parameters(): torch.nn.init.normal_(p, std=0.3)
    torch.nn.init.normal_(mp.dec.weight, std=0.1)
    U, V = mp.uv(qp[:, None].expand_as(cp), cp, R); e = mp.E(R)
    Aj = torch.eye(M.D, device=dev) + U @ V.transpose(-1, -2)
    ref = mp.dec((w[..., None] * ((Aj @ e[..., None])[..., 0] - e)).sum(1))
    got = mp(w, R, qp, cp)
    err = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
    print(f'H{H} pair-m vs explicit rel err {err:.2e}'); assert err < 1e-4
assert 'zn' not in inspect.signature(M.Mat.forward).parameters
print('ALL OK')
