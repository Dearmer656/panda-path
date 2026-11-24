"""PAT-318: bidirectional PaTH scores for Panda's (encoder, non-causal) temporal attention.

H_t = I - beta_t w_t w_t^T (|w_t| = 1, beta_t in (0, 2)).
    j <= i : S[i, j] = q_i^T H_i H_{i-1} ... H_{j+1} k_j      (causal PaTH, identical to fla's path attention)
    j >  i : S[i, j] = q_i^T H_{i+1} H_{i+2} ... H_j k_j      (the same reflections, walked in the other direction)

Causal part in closed form for a single chunk (N = number of patches, 32 in Panda), following fla's naive_path_attn:
    S_c = tril(Q K^T) - tril(Q W^T) (I + L)^{-1} strict_tril((W*beta) K^T),   L = strict_tril((W*beta) W^T).
The anticausal part is the causal formula on the time-reversed sequence with the reflections shifted by one
(H'_t' = H_{N-t'}, beta'_0 = 0).
"""
import torch


def causal_path_scores(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """q, k, w: [B, N, E]; beta: [B, N]. Returns [B, N, N]; entries with j > i are 0."""
    N = q.shape[1]
    wb = w * beta[..., None]
    strict = torch.ones(N, N, dtype=torch.bool, device=q.device).tril(-1)
    eye = torch.eye(N, dtype=q.dtype, device=q.device)
    L = (wb @ w.transpose(1, 2)) * strict                                    # [B, N, N], L[t, s] = beta_t w_t.w_s, t > s
    R = (wb @ k.transpose(1, 2)) * strict                                    # R[t, j] = beta_t w_t.k_j, t > j
    X = torch.linalg.solve_triangular(eye + L, R, upper=False, unitriangular=True)
    qw = (q @ w.transpose(1, 2)).tril()                                      # qw[i, t] = q_i.w_t, t <= i
    return (q @ k.transpose(1, 2)).tril() - qw @ X


def bidir_path_scores(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, beta: torch.Tensor,
                      wb: torch.Tensor | None = None, bb: torch.Tensor | None = None) -> torch.Tensor:
    """q, k, w: [B, N, E]; beta: [B, N]. Full non-causal PaTH score matrix [B, N, N].
    PAT-337 path2: wb / bb (same shapes) = separate reflections for the anticausal walk (j > i):
    S[i, j] = q_i^T Hb_{i+1} ... Hb_j k_j; None -> the forward ones (original PaTH)."""
    S_c = causal_path_scores(q, k, w, beta)
    if wb is not None: w, beta = wb, bb
    qr, kr = q.flip(1), k.flip(1)
    wr = torch.cat([w[:, :1], w.flip(1)[:, :-1]], 1)                         # w'_t' = w_{N-t'} for t' >= 1
    br = torch.cat([torch.zeros_like(beta[:, :1]), beta.flip(1)[:, :-1]], 1)  # beta'_0 = 0 (unused)
    S_a = causal_path_scores(qr, kr, wr, br).flip(1).flip(2)                 # lower part of reversed = upper part here
    N = q.shape[1]
    upper = torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1)
    return torch.where(upper, S_a, S_c)
