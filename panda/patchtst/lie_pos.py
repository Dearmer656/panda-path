"""PAT-336: Lie-exponential transport for Panda's (encoder, non-causal) temporal attention.

Per head and patch t, from the attention input x_t:
    Q_t   in R^{E x r}, orthonormal (QR of a linear projection)                    -- the subspace the step acts on
    m_t   = dt_t * (K(omega_t) + diag(lambda_t))  in R^{r x r}                      -- generator in that subspace
            K skew (rotation part), lambda_t = tanh(.) scaled to (-1, LMAX) (contraction / bounded expansion),
            dt_t = 2 sigmoid(.) in (0, 2)  (data-dependent time step)
    M_t   = Q_t m_t Q_t^T  (rank-r element of gl(E): skew + symmetric parts)
    A_t   = exp(M_t) = I + Q_t (exp(m_t) - I) Q_t^T,   A_t^{-1} = I + Q_t (exp(-m_t) - I) Q_t^T     (exact)
Scores, with C_i = A_i ... A_1:
    S[i, j] = q_i^T C_i C_j^{-1} k_j
            = q_i^T A_i ... A_{j+1} k_j                    (j <= i)
            = q_i^T A_{i+1}^{-1} ... A_j^{-1} k_j           (j >  i)
PaTH is the special case A_t = I - beta w w^T (self-inverse, so its bidirectional walk equals C_i C_j^{-1} too).
Computed without forming C (no cancellation): each step is I + P_t R_t^T (rank r) and the causal product has the
closed form of path_pos.causal_path_scores with an (N r) x (N r) unit-lower-triangular solve.
"""
import torch

LMAX = 0.1


def causal_lowrank_scores(q, k, P, R):
    """q, k: [B, N, E]; P, R: [B, N, E, r], A_t = I + P_t R_t^T. Returns S[i, j] = q_i^T A_i ... A_{j+1} k_j (j <= i), 0 above."""
    B, N, E = q.shape; r = P.shape[-1]
    Pf = P.permute(0, 1, 3, 2).reshape(B, N * r, E)                         # rows (t, a): column a of P_t
    Rf = R.permute(0, 1, 3, 2).reshape(B, N * r, E)
    tt = torch.arange(N, device=q.device).repeat_interleave(r)               # time of each (t, a) row
    strict = (tt[:, None] > tt[None, :])                                     # [Nr, Nr], t > s
    L = (Rf @ Pf.transpose(1, 2)) * strict                                   # L[(t,a),(s,b)] = R_t[:,a] . P_s[:,b]
    Rk = (Rf @ k.transpose(1, 2)) * (tt[:, None] > torch.arange(N, device=q.device)[None, :])   # [Nr, N], t > j
    eye = torch.eye(N * r, dtype=q.dtype, device=q.device)
    X = torch.linalg.solve_triangular(eye - L, Rk, upper=False, unitriangular=True)
    qP = (q @ Pf.transpose(1, 2)) * (torch.arange(N, device=q.device)[:, None] >= tt[None, :])  # t <= i
    return (q @ k.transpose(1, 2)).tril() + qP @ X


def lie_factors(basis, gen, r):
    """basis: [B, N, E, r] raw vectors; gen: [B, N, r(r-1)/2 + r + 1]. Returns Q, G = exp(m) - I, Gi = exp(-m) - I."""
    Q = torch.linalg.qr(basis).Q                                             # [B, N, E, r]
    nk = r * (r - 1) // 2
    om, lam, dt = gen[..., :nk], gen[..., nk:nk + r], 2 * torch.sigmoid(gen[..., -1:])
    iu = torch.triu_indices(r, r, 1, device=gen.device)
    K = torch.zeros(*gen.shape[:-1], r, r, dtype=gen.dtype, device=gen.device)
    K[..., iu[0], iu[1]] = om; K = K - K.transpose(-1, -2)
    t = torch.tanh(lam); lam = torch.where(t > 0, LMAX * t, t)
    m = dt[..., None] * (K + torch.diag_embed(lam))
    I = torch.eye(r, dtype=gen.dtype, device=gen.device)
    return Q, torch.linalg.matrix_exp(m) - I, torch.linalg.matrix_exp(-m) - I


def bidir_lie_scores(q, k, basis, gen):
    """q, k: [B, N, E]; basis: [B, N, E, r]; gen: [B, N, r(r-1)/2 + r + 1]. Full score matrix [B, N, N] (fp32 inside)."""
    dt0 = q.dtype; ct = torch.promote_types(dt0, torch.float32); q, k, basis, gen = q.to(ct), k.to(ct), basis.to(ct), gen.to(ct)
    r = basis.shape[-1]
    Q, G, Gi = lie_factors(basis, gen, r)
    S_c = causal_lowrank_scores(q, k, Q @ G, Q)                              # A_t = I + Q G Q^T
    # anticausal: reversed time t' = N-1-t; the walk q_i^T A_{i+1}^{-1} ... A_j^{-1} k_j becomes the causal walk with
    # B_{t'} = A^{-1}_{N - t'} (t' >= 1); B_0 unused -> zero update
    Pi = (Q @ Gi).flip(1); Qr = Q.flip(1)
    Pi = torch.cat([torch.zeros_like(Pi[:, :1]), Pi[:, :-1]], 1); Qr = torch.cat([Qr[:, :1], Qr[:, :-1]], 1)
    S_a = causal_lowrank_scores(q.flip(1), k.flip(1), Pi, Qr).flip(1).flip(2)
    N = q.shape[1]
    upper = torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1)
    return torch.where(upper, S_a, S_c).to(dt0)
