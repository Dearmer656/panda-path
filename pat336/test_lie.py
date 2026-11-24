"""PAT-336 checks for temporal_pos = lie (python -m pat336.test_lie from the panda root).
1. closed-form scores == brute force q_i^T C_i C_j^{-1} k_j with dense matrix_exp of the full E x E generator (fp64).
2. A_t^{-1} really inverts A_t; pure-rotation steps (lambda -> 0) are orthogonal.
3. PaTH special case: the low-rank causal kernel with P = -beta w, R = w reproduces causal_path_scores.
4. attention module: forward / backward finite, gradients reach lie_basis_proj and lie_gen_proj; fp32 vs fp64 agree.
5. save -> from_pretrained round trip keeps the lie_* weights (HF key renames)."""
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
assert torch.allclose(O @ O.transpose(-1, -2), I.expand_as(O), atol=1e-10); print('2 ok')
w = torch.nn.functional.normalize(torch.randn(B, N, E, dtype=dt), dim=-1); be = 2 * torch.rand(B, N, dtype=dt)
assert torch.allclose(lowrank_scores(q, k, w[:, :, None], -be[..., None, None]), bidir_path_scores(q, k, w, be), atol=1e-10)
print('3 ok (PaTH special case, both directions)')
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
