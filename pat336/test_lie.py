"""PAT-336 checks for temporal_pos = lie / lie_s / lie_k (python -m pat336.test_lie from the panda root).
1. closed-form scores == brute force: causal q_i^T A_i ... A_{j+1} k_j, anticausal q_i^T A_{i+1}^T ... A_j^T k_j
   (TRANSPOSE convention, not the inverse), with A_t = matrix_exp of the dense E x E generator Q^T m Q (fp64).
2. Gram-Schmidt rows are orthonormal, I + Q^T G Q == exp(Q^T m Q); pure-rotation steps (lambda = 0) are orthogonal,
   and with stretch A^T != A^{-1} (the two conventions really differ).
3. score-KERNEL equivalence: lowrank_scores with r = 1, Q = w, G = -beta reproduces bidir_path_scores for any
   beta in (0, 2). This is a statement about the kernel only, not about the parameterised Lie family.
4. attention module: forward / backward finite, gradients reach lie_basis_proj and lie_gen_proj; fp32 vs fp64 agree.
5. save -> from_pretrained round trip keeps the lie_* weights (HF key renames).
6. model-FAMILY reachability through lie_factors (r = 1, m = dt * lambda in (-2, 0.2)): a reachable beta (0.6) built
   from gen reproduces PaTH; beta = 1 - exp(m) stays inside (1 - e^{0.2}, 1 - e^{-2}) ~= (-0.2214, 0.8647), so PaTH's
   beta >= 0.8647 is unreachable.
7. modes: lie_s (K = 0) gives symmetric A_t; lie_k (lambda = 0) gives orthogonal A_t (A^T = A^{-1}); same parameter
   count for lie / lie_s / lie_k; module output differs between modes."""
import torch, tempfile
from panda.patchtst.lie_pos import bidir_lie_scores, lie_factors, lowrank_scores, LMAX
from panda.patchtst.path_pos import bidir_path_scores
from panda.patchtst.patchtst import PatchTSTRopeAttention
torch.manual_seed(0); dt = torch.float64
B, N, E, r = 3, 7, 6, 4; nk = r * (r - 1) // 2
q, k = torch.randn(B, N, E, dtype=dt), torch.randn(B, N, E, dtype=dt)
basis, gen = torch.randn(B, N, r, E, dtype=dt), torch.randn(B, N, nk + r + 1, dtype=dt) * 1.5
S = bidir_lie_scores(q, k, basis, gen)
Q, G = lie_factors(basis, gen, r)
Qc = Q.transpose(-1, -2)                                                     # [B, N, E, r] orthonormal columns
iu = torch.triu_indices(r, r, 1); K = torch.zeros(B, N, r, r, dtype=dt); K[..., iu[0], iu[1]] = gen[..., :nk]; K = K - K.transpose(-1, -2)
t = torch.tanh(gen[..., nk:nk + r]); lam = torch.where(t > 0, LMAX * t, t); d = 2 * torch.sigmoid(gen[..., -1:])
A = torch.linalg.matrix_exp(Qc @ (d[..., None] * (K + torch.diag_embed(lam))) @ Qc.transpose(-1, -2)); I = torch.eye(E, dtype=dt)
Sb = torch.zeros(B, N, N, dtype=dt)
for i in range(N):
    for j in range(N):
        P = I.expand(B, E, E).clone()
        if j <= i:
            for s in range(j + 1, i + 1): P = A[:, s] @ P                     # A_i ... A_{j+1}
        else:
            for s in range(i + 1, j + 1): P = P @ A[:, s].transpose(-1, -2)    # A_{i+1}^T ... A_j^T
        Sb[:, i, j] = torch.einsum('be,bef,bf->b', q[:, i], P, k[:, j])
print('1 max |closed - brute|', float((S - Sb).abs().max())); assert torch.allclose(S, Sb, atol=1e-9)
assert torch.allclose(Q @ Q.transpose(-1, -2), torch.eye(r, dtype=dt).expand(B, N, r, r), atol=1e-10)
assert torch.allclose(I + Qc @ G @ Q, A, atol=1e-10)
g0 = gen.clone(); g0[..., nk:nk + r] = 0; Q0, G0 = lie_factors(basis, g0, r); O = I + Q0.transpose(-1, -2) @ G0 @ Q0
assert torch.allclose(O @ O.transpose(-1, -2), I.expand_as(O), atol=1e-10)
assert not torch.allclose(A.transpose(-1, -2), torch.linalg.inv(A), atol=1e-3); print('2 ok (A^T != A^-1 with stretch)')
w = torch.nn.functional.normalize(torch.randn(B, N, E, dtype=dt), dim=-1); be = 2 * torch.rand(B, N, dtype=dt)
assert torch.allclose(lowrank_scores(q, k, w[:, :, None], -be[..., None, None]), bidir_path_scores(q, k, w, be), atol=1e-10)
print('3 ok (score-kernel equivalence with PaTH, both directions; NOT family equivalence, see 6)')
a = PatchTSTRopeAttention(embed_dim=32, num_heads=4, pos='lie')
x = torch.randn(5, 32, 32, requires_grad=True)
y = a(x)[0]; y.square().mean().backward()
assert torch.isfinite(y).all() and a.lie_basis_proj.weight.grad.abs().sum() > 0 and a.lie_gen_proj.weight.grad.abs().sum() > 0
a64 = PatchTSTRopeAttention(embed_dim=32, num_heads=4, pos='lie').to(dt); a64.load_state_dict({kk: v.to(dt) for kk, v in a.state_dict().items()})
print('4 fp32 vs fp64', float((a(x)[0].double() - a64(x.double())[0]).abs().max())); assert torch.allclose(a(x)[0].double(), a64(x.double())[0], atol=1e-4)
from transformers import PatchTSTConfig
from panda.patchtst.patchtst import PatchTSTForPrediction
c = PatchTSTConfig.from_pretrained('pat318/ckpt/p318_path_s99_st100000-0/checkpoint-final')
c.num_hidden_layers = 2; c.temporal_pos = 'lie'
m = PatchTSTForPrediction(c); sd = {kk: v.clone() for kk, v in m.state_dict().items() if 'lie_' in kk}
with tempfile.TemporaryDirectory() as dd:
    m.save_pretrained(dd); m2 = PatchTSTForPrediction.from_pretrained(dd)
sd2 = m2.state_dict(); assert len(sd) > 0 and all(torch.equal(sd[kk], sd2[kk]) for kk in sd); print('5 ok', len(sd), 'lie tensors')

# 6. family reachability through lie_factors, r = 1 (gen = [lambda_raw, dt_raw])
import math
b0 = 0.6; m0 = math.log(1 - b0)                                            # beta = 1 - exp(m), m = dt * lambda
dt0 = 1.0; lam0 = m0 / dt0; graw = torch.tensor([math.atanh(lam0), 0.0], dtype=dt)  # dt = 2 sigmoid(0) = 1, lambda < 0 branch
Q1, G1 = lie_factors(w[:, :, None], graw.expand(B, N, 2), 1)
assert torch.allclose(G1, torch.full_like(G1, -b0), atol=1e-12)
assert torch.allclose(lowrank_scores(q, k, Q1, G1), bidir_path_scores(q, k, w, torch.full((B, N), b0, dtype=dt)), atol=1e-10)
g = torch.randn(200000, 2, dtype=dt) * 20; _, Gs = lie_factors(torch.randn(200000, 1, 1, 3, dtype=dt), g[:, None], 1)
bs = -Gs.flatten(); lo, hi = 1 - math.exp(0.2), 1 - math.exp(-2)
print(f'6 reachable beta via lie_factors in [{float(bs.min()):.4f}, {float(bs.max()):.4f}] (bound ({lo:.4f}, {hi:.4f}))')
assert float(bs.min()) >= lo - 1e-12 and float(bs.max()) <= hi + 1e-12; print('6 ok (family covers beta < 0.8647 only; PaTH beta in [0.8647, 2) unreachable)')

# 7. modes
for mode in ('s', 'k'):
    Qm, Gm = lie_factors(basis, gen, r, mode); Am = I + Qm.transpose(-1, -2) @ Gm @ Qm
    if mode == 's': assert torch.allclose(Am, Am.transpose(-1, -2), atol=1e-12)
    else: assert torch.allclose(Am @ Am.transpose(-1, -2), I.expand_as(Am), atol=1e-10)
    Sm = bidir_lie_scores(q, k, basis, gen, mode); assert not torch.allclose(Sm, S, atol=1e-3)
torch.manual_seed(1); mods = {p: PatchTSTRopeAttention(embed_dim=32, num_heads=4, pos=p) for p in ('lie', 'lie_s', 'lie_k')}
n = {p: sum(v.numel() for v in mm.parameters()) for p, mm in mods.items()}; assert len(set(n.values())) == 1
sdl = mods['lie'].state_dict(); [mods[p].load_state_dict(sdl) for p in ('lie_s', 'lie_k')]
ys = {p: mm(x)[0] for p, mm in mods.items()}; assert not torch.allclose(ys['lie'], ys['lie_s']) and not torch.allclose(ys['lie'], ys['lie_k'])
print('7 ok (modes: lie_s symmetric, lie_k orthogonal, params', n, ')')
