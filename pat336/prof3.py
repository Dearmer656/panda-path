import torch
import panda.patchtst.lie_pos as LP
dev = 'cuda'; S, N, E, r = 128 * 3 * 8, 32, 64, 4
q, k = torch.randn(S, N, E, device=dev, requires_grad=True), torch.randn(S, N, E, device=dev, requires_grad=True)
basis = torch.randn(S, N, r, E, device=dev, requires_grad=True); gen = torch.randn(S, N, 11, device=dev, requires_grad=True)
f = lambda: LP.bidir_lie_scores(q, k, basis, gen).sum().backward()
f(); torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    f(); torch.cuda.synchronize()
print(p.key_averages().table(sort_by='cuda_time_total', row_limit=14))
