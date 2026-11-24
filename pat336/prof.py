import torch, time
from panda.patchtst.lie_pos import bidir_lie_scores, lie_factors
from panda.patchtst.path_pos import bidir_path_scores
dev = 'cuda'; S, N, E, r = 128 * 3 * 8, 32, 64, 4
q, k = torch.randn(S, N, E, device=dev, requires_grad=True), torch.randn(S, N, E, device=dev, requires_grad=True)
basis = torch.randn(S, N, E, r, device=dev, requires_grad=True); gen = torch.randn(S, N, 11, device=dev, requires_grad=True)
w = torch.randn(S, N, E, device=dev, requires_grad=True); be = torch.rand(S, N, device=dev, requires_grad=True)
def T(name, f, n=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); print(f'{name:28s} {(time.time() - t) / n * 1000:9.1f} ms', flush=True)
T('path fwd+bwd', lambda: bidir_path_scores(q, k, torch.nn.functional.normalize(w, dim=-1), be).sum().backward())
T('qr fwd', lambda: torch.linalg.qr(basis).Q, 1)
m = torch.randn(S, N, r, r, device=dev, requires_grad=True)
T('matrix_exp fwd+bwd', lambda: torch.linalg.matrix_exp(m).sum().backward())
T('lie_factors fwd+bwd', lambda: sum(x.sum() for x in lie_factors(basis, gen, r)).backward())
T('lie scores fwd+bwd', lambda: bidir_lie_scores(q, k, basis, gen).sum().backward())
