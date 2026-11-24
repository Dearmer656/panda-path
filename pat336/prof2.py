import torch, time
import panda.patchtst.lie_pos as LP
dev = 'cuda'; S, N, E = 128 * 3 * 8, 32, 64
def T(name, f, n=5):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); print(f'{name:34s} {(time.time() - t) / n * 1000:9.1f} ms', flush=True)
from panda.patchtst.path_pos import bidir_path_scores
w = torch.randn(S, N, E, device=dev, requires_grad=True); be = torch.rand(S, N, device=dev, requires_grad=True); qq, kk = torch.randn(S, N, E, device=dev, requires_grad=True), torch.randn(S, N, E, device=dev, requires_grad=True)
T('path fwd+bwd', lambda: bidir_path_scores(qq, kk, torch.nn.functional.normalize(w, dim=-1), be).sum().backward())
for r in (4, 2):
    q, k = torch.randn(S, N, E, device=dev, requires_grad=True), torch.randn(S, N, E, device=dev, requires_grad=True)
    basis = torch.randn(S, N, r, E, device=dev, requires_grad=True); gen = torch.randn(S, N, r * (r - 1) // 2 + r + 1, device=dev, requires_grad=True)
    T(f'r{r} lie_factors fwd+bwd', lambda: sum(x.sum() for x in LP.lie_factors(basis, gen, r)).backward())
    T(f'r{r} lie scores fwd+bwd', lambda: LP.bidir_lie_scores(q, k, basis, gen).sum().backward())
    cf = torch.compile(LP.bidir_lie_scores)
    T(f'r{r} lie scores compiled fwd+bwd', lambda: cf(q, k, basis, gen).sum().backward())
