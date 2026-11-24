"""PAT-336: Lie-exponential transport for Panda's (encoder, non-causal) temporal attention.

Per head and patch t, from the attention input x_t:
    Q_t   in R^{r x E}, orthonormal rows (Gram-Schmidt of a linear projection)        -- the subspace the step acts on
    m_t   = dt_t * (K(omega_t) + diag(lambda_t))  in R^{r x r}                      -- generator in that subspace
            K skew (rotation part), lambda_t = tanh(.) scaled to (-1, LMAX) (contraction / bounded expansion),
            dt_t = 2 sigmoid(.) in (0, 2)  (data-dependent time step)
    A_t   = exp(Q_t^T m_t Q_t) = I + Q_t^T (exp(m_t) - I) Q_t       (exact; rank-r element of GL(E))
Scores (same bidirectional convention as PaTH, whose H_t is symmetric):
    S[i, j] = q_i^T A_i ... A_{j+1} k_j                      (j <= i)
            = q_i^T A_{i+1}^T ... A_j^T k_j                 (j >  i)   (= q_i^T (A_j ... A_{i+1})^T k_j)
For pure rotations (lambda = 0) both equal q_i^T C_i C_j^{-1} k_j with C_i = A_i ... A_1 (RoPE-like relative transport);
the transpose keeps contraction a decay in both directions (the inverse would turn it into growth up to e^{62}).
PaTH = r = 1, Q = w, exp(m) - 1 = -beta.
Closed form: with x_t = Q_t (A_{t-1} ... A_{j+1} k_j), (I - L) x = Q k (L block strictly lower, L[t, s] = Q_t Q_s^T G_s),
S = tril(q k^T) + (q Q^T G) x; the anticausal half is the same with a block strictly upper U and G^T. Both halves share
the (N r) x (N r) Gram matrix Q Q^T and Q k^T, so only three E-dim matmuls are needed.
"""
import torch

LMAX = 0.1


def lie_factors(basis, gen, r):
    """basis: [B, N, r, E] raw vectors; gen: [B, N, r(r-1)/2 + r + 1]. Returns Q [B, N, r, E] orthonormal rows, G = exp(m) - I."""
    rows = []                                                                # Gram-Schmidt (batched QR on GPU is ~300x slower)
    for a in range(r):
        v = basis[..., a, :]
        for u in rows: v = v - (u * v).sum(-1, keepdim=True) * u
        rows.append(torch.nn.functional.normalize(v, dim=-1, eps=1e-6))
    Q = torch.stack(rows, -2)
    nk = r * (r - 1) // 2
    om, lam, dt = gen[..., :nk], gen[..., nk:nk + r], 2 * torch.sigmoid(gen[..., -1:])
    iu = torch.triu_indices(r, r, 1, device=gen.device)
    K = torch.zeros(*gen.shape[:-1], r, r, dtype=gen.dtype, device=gen.device)
    K[..., iu[0], iu[1]] = om; K = K - K.transpose(-1, -2)
    t = torch.tanh(lam); lam = torch.where(t > 0, LMAX * t, t)
    m = dt[..., None] * (K + torch.diag_embed(lam))
    return Q, torch.linalg.matrix_exp(m) - torch.eye(r, dtype=gen.dtype, device=gen.device)


def lowrank_scores(q, k, Q, G):
    """q, k: [B, N, E]; Q: [B, N, r, E]; G: [B, N, r, r], A_t = I + Q_t^T G_t Q_t. Full bidirectional score matrix [B, N, N]."""
    B, N, E = q.shape; r = Q.shape[2]; M = N * r
    Qf = Q.reshape(B, M, E)
    Gr = Qf @ Qf.transpose(1, 2)                                             # [B, M, M] Gram
    Qk = Qf @ k.transpose(1, 2)                                              # [B, M, N]
    qQ = (q @ Qf.transpose(1, 2)).view(B, N, N, r)                           # [B, N(i), N(t), r]
    tt = torch.arange(N, device=q.device).repeat_interleave(r); n = torch.arange(N, device=q.device)
    eye = torch.eye(M, dtype=q.dtype, device=q.device)

    def rmul(X, Gb):                                                         # X [B, M, M] times blockdiag(Gb) on the right
        return torch.einsum('bmsa,bsac->bmsc', X.view(B, M, N, r), Gb).reshape(B, M, M)
    # causal: L[(t,a),(s,b)] = (Gr blockdiag(G))[..], t > s ; rhs t > j ; readout t <= i
    L = rmul(Gr, G) * (tt[:, None] > tt[None, :])
    X = torch.linalg.solve_triangular(eye - L, Qk * (tt[:, None] > n[None, :]), upper=False, unitriangular=True)
    qG = torch.einsum('bita,btac->bitc', qQ, G).reshape(B, N, M) * (n[:, None] >= tt[None, :])
    S_c = (q @ k.transpose(1, 2)).tril() + qG @ X
    # anticausal (A^T): U[(t,a),(s,b)], s > t ; rhs t <= j ; readout t > i
    Gt = G.transpose(-1, -2)
    U = rmul(Gr, Gt) * (tt[None, :] > tt[:, None])
    Y = torch.linalg.solve_triangular(eye - U, Qk * (tt[:, None] <= n[None, :]), upper=True, unitriangular=True)
    qGt = torch.einsum('bita,btac->bitc', qQ, Gt).reshape(B, N, M) * (tt[None, :] > n[:, None])
    S_a = (q @ k.transpose(1, 2)).triu(1) + qGt @ Y
    return torch.where(torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1), S_a, S_c)


def bidir_lie_scores(q, k, basis, gen):
    """q, k: [B, N, E]; basis: [B, N, r, E]; gen: [B, N, r(r-1)/2 + r + 1]. Full score matrix [B, N, N] (>= fp32 inside)."""
    dt0 = q.dtype; ct = torch.promote_types(dt0, torch.float32); q, k, basis, gen = q.to(ct), k.to(ct), basis.to(ct), gen.to(ct)
    Q, G = lie_factors(basis, gen, basis.shape[2])
    return lowrank_scores(q, k, Q, G).to(dt0)
