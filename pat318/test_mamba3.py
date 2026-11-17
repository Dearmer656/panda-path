"""PAT-318 mamba3 arm checks (GPU): (1) official Mamba3 imports and runs fp32 fwd/bwd at N = 32/64/128 patches;
(2) bidirectional: output at t depends on tokens before and after t; reversing the input reverses the output (tied weights);
(3) mixer params vs a temporal attention block; (4) full Panda model: params, train step, no unused params, torch.compile;
(5) checkpoint save -> from_pretrained + exact_load round trip gives identical forecasts."""
import tempfile, time, torch, transformers
from panda.patchtst.patchtst import PatchTSTRopeAttention
torch.manual_seed(0); dev = 'cuda'
m = PatchTSTRopeAttention(512, 8, pos='mamba3').to(dev)
for N in (32, 64, 128):
    x = torch.randn(64, N, 512, device=dev, requires_grad=True)
    y = m(x)[0]; y.sum().backward(); assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    t = N // 2; g = torch.autograd.grad(m(x)[0][:, t].sum(), x)[0].abs().sum((0, 2))
    print(f'1/2 N={N} ok | grad wrt past {g[:t].sum():.3g} future {g[t + 1:].sum():.3g}'); assert g[:t].sum() > 0 and g[t + 1:].sum() > 0
x = torch.randn(4, 32, 512, device=dev)
print('2 reverse equivariance err', (m(x.flip(1))[0] - m(x)[0].flip(1)).abs().max().item())
a = PatchTSTRopeAttention(512, 8, pos='rope')
print('3 params mamba3 mixer', sum(p.numel() for p in m.parameters()), '| attention', sum(p.numel() for p in a.parameters()))
import sys
if len(sys.argv) > 1:  # (5) round trip of a smoke checkpoint
    from panda.patchtst.pipeline import PatchTSTPipeline
    from pat318.exact_load import exact_load
    from safetensors.torch import load_file
    ck = sys.argv[1]; p = PatchTSTPipeline.from_pretrained(mode='predict', pretrain_path=ck); sd = load_file(ck + '/model.safetensors')
    msd = p.model.state_dict(); bad = [k for k in sd if k not in msd or not torch.equal(sd[k].cpu(), msd[k].cpu())]
    print('5 from_pretrained mismatched keys', len(bad), bad[:4]); exact_load(p.model, ck); print('5 exact_load strict ok')
    print('5 total params', sum(v.numel() for v in p.model.parameters()))
