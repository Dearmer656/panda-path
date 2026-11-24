"""PAT-335/336: fp32 closed-form transport scores vs explicit fp64 step-by-step products.

Reference: k_j is pushed through the explicit E x E transitions one step at a time in fp64
  causal       v <- A_s v for s = j+1 .. i,  S[i, j] = q_i . v
  anticausal   v <- A_s^T v for s = j .. i+1 (descending), S[i, j] = q_i . v      (path2: the anticausal reflections)
with A_s built in fp64 from the fp64 Gram-Schmidt basis (lie) / normalised w (path) -- i.e. the INTENDED operator.
Error = max |S32 - S64| / max |S64| over the score matrix. Tolerance (documented, fixed before running): 1e-3.
Parts
  (a) real inputs: transport layers of each checkpoint at ctx 2048 (N = 128 patches), first base40 test trajectory of
      4 systems, 8 sequence-heads per layer;
  (b) synthetic stress, N = 128, E = 64, r = 4: lie with near-dependent raw bases (rows = b0 + eps * noise,
      eps 1e-2 / 1e-4 / 1e-6), lie expansion stress (lambda at LMAX, dt -> 2, large rotations), path with beta -> 2.
usage: python -m pat336.numcheck [CKPT ...]"""
import sys, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from panda.patchtst.path_pos import bidir_path_scores
from panda.patchtst.lie_pos import lie_factors, lowrank_scores
dev = torch.device('cuda'); TOL = 1e-3; D = torch.float64


def explicit(q, k, A, Ab):
    """q, k [n, N, E] fp64; A, Ab [n, N, E, E] fp64 (Ab = anticausal operators, transposed inside). returns [n, N, N]."""
    n, N, E = q.shape; S = torch.zeros(n, N, N, dtype=D, device=q.device)
    for j in range(N):
        v = k[:, j]; S[:, j, j] = (q[:, j] * v).sum(-1)
        for i in range(j + 1, N): v = (A[:, i] @ v[..., None])[..., 0]; S[:, i, j] = (q[:, i] * v).sum(-1)
        v = k[:, j]
        for i in range(j - 1, -1, -1): v = (Ab[:, i + 1].transpose(-1, -2) @ v[..., None])[..., 0]; S[:, i, j] = (q[:, i] * v).sum(-1)
    return S


def lie_ref(q, k, basis, gen, mode):
    r = basis.shape[2]; Q, G = lie_factors(basis.double(), gen.double(), r, mode)
    A = torch.eye(q.shape[-1], dtype=D, device=q.device) + Q.transpose(-1, -2) @ G @ Q
    return explicit(q.double(), k.double(), A, A)


def path_ref(q, k, w, be, wb=None, bb=None):
    I = torch.eye(q.shape[-1], dtype=D, device=q.device); w = torch.nn.functional.normalize(w.double(), dim=-1)
    A = I - be.double()[..., None, None] * w[..., :, None] * w[..., None, :]
    if wb is None: Ab = A
    else: wb = torch.nn.functional.normalize(wb.double(), dim=-1); Ab = I - bb.double()[..., None, None] * wb[..., :, None] * wb[..., None, :]
    return explicit(q.double(), k.double(), A, Ab)


def err(S32, S64): return float((S32.double() - S64).abs().max() / S64.abs().max())


def orth(basis, r):
    Q, _ = lie_factors(basis, torch.zeros(*basis.shape[:2], r * (r - 1) // 2 + r + 1, dtype=basis.dtype, device=basis.device), r)
    return float((Q @ Q.transpose(-1, -2) - torch.eye(r, dtype=basis.dtype, device=basis.device)).norm(dim=(-1, -2)).max())


@torch.no_grad()
def real(ck):
    import numpy as np
    from pat336.props import load, systems, qk, T0
    import pat336.props as PR
    PR.NSYS = 4
    p = load(ck); m = p.model; m.config.context_length = 2048; layers = [l.temporal_self_attn for l in m.model.encoder.layers]
    data = systems()[:4]; cap = {}
    hs = [a.register_forward_hook(lambda mod, args, kw, out, i=i: cap.setdefault(i, (args[0] if args else kw['hidden_states'], kw.get('path_ctx'))) and None, with_kwargs=True) for i, a in enumerate(layers)]
    x0 = torch.tensor(data[0][1][:, T0 - 2048:T0], device=dev).T[None]; m(past_values=x0)
    for h in hs: h.remove()
    print(f'== {ck} (ctx 2048, {data[0][0]})', flush=True); bad = []
    for i, a in enumerate(layers):
        x, pc = cap[i]; bsz, N, _ = x.shape; H, E = a.num_heads, a.head_dim; ps = (bsz * H, N, E)
        q, k, _, _ = qk(a, x); sel = torch.arange(0, bsz * H, max(1, bsz * H // 8), device=dev)[:8]
        if 'path' in a.pos:
            hp = x if pc is None else pc
            w = torch.nn.functional.normalize(a._shape(a.w_proj(hp), -1, bsz).view(*ps), dim=-1); be = 2 * torch.sigmoid(a.beta_proj(hp)).transpose(1, 2).reshape(bsz * H, N)
            wb = bb = None
            if a.pos == 'path2':
                wb = torch.nn.functional.normalize(a._shape(a.path_wb_proj(hp), -1, bsz).view(*ps), dim=-1); bb = 2 * torch.sigmoid(a.path_sb_proj(hp)).transpose(1, 2).reshape(bsz * H, N)
            S32 = bidir_path_scores(q, k, w, be, wb, bb)[sel]
            S64 = path_ref(q[sel], k[sel], w[sel], be[sel], None if wb is None else wb[sel], None if bb is None else bb[sel])
        elif a.pos.startswith('lie'):
            r = a.lie_rank
            basis = a.lie_basis_proj(x).view(bsz, N, H, r, E).transpose(1, 2).reshape(bsz * H, N, r, E); gen = a.lie_gen_proj(x).view(bsz, N, H, -1).transpose(1, 2).reshape(bsz * H, N, -1)
            Q, G = lie_factors(basis, gen, r, a.lie_mode); S32 = lowrank_scores(q, k, Q, G)[sel]; S64 = lie_ref(q[sel], k[sel], basis[sel], gen[sel], a.lie_mode)
        else:
            continue
        e = err(S32, S64); print(f'  L{i} {a.pos}: rel max err fp32 vs explicit fp64 {e:.2e}' + ('  FAIL' if e > TOL else ''), flush=True)
        if e > TOL: bad.append(i)
    print(f'  real-input check: {"PASS" if not bad else "FAIL layers " + str(bad)} (tol {TOL})', flush=True)


@torch.no_grad()
def stress():
    torch.manual_seed(0); n, N, E, r = 4, 128, 64, 4; nk = r * (r - 1) // 2; f = torch.float32
    q, k = torch.randn(n, N, E, device=dev) / 8, torch.randn(n, N, E, device=dev)
    print('== synthetic stress (N 128, E 64, r 4)', flush=True)
    for eps in (1e-2, 1e-4, 1e-6):
        b0 = torch.randn(n, N, 1, E, device=dev); basis = b0 + eps * torch.randn(n, N, r, E, device=dev); gen = torch.randn(n, N, nk + r + 1, device=dev)
        Q, G = lie_factors(basis, gen, r); e = err(lowrank_scores(q, k, Q, G), lie_ref(q, k, basis, gen, 'ks'))
        print(f'  lie near-dependent basis eps {eps:.0e}: orth err fp32 {orth(basis, r):.1e} fp64 {orth(basis.double(), r):.1e}; score rel err {e:.2e}' + ('  FAIL' if e > TOL else ''), flush=True)
    basis = torch.randn(n, N, r, E, device=dev); gen = torch.cat([torch.randn(n, N, nk, device=dev) * 5, torch.full((n, N, r), 10., device=dev), torch.full((n, N, 1), 10., device=dev)], -1)
    Q, G = lie_factors(basis, gen, r); e = err(lowrank_scores(q, k, Q, G), lie_ref(q, k, basis, gen, 'ks'))
    print(f'  lie expansion stress (lambda = LMAX, dt = 2, |omega| ~ 5): score rel err {e:.2e}, max|S| {float(lie_ref(q, k, basis, gen, "ks").abs().max()):.3g}' + ('  FAIL' if e > TOL else ''), flush=True)
    w = torch.randn(n, N, E, device=dev); be = torch.full((n, N), 1.999, device=dev)
    e = err(bidir_path_scores(q, k, torch.nn.functional.normalize(w, dim=-1), be), path_ref(q, k, w, be))
    print(f'  path beta = 1.999: score rel err {e:.2e}' + ('  FAIL' if e > TOL else ''), flush=True)


if __name__ == '__main__':
    stress()
    for ck in sys.argv[1:]: real(ck)
