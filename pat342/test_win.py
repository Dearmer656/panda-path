"""PAT-342 eval-only interventions P342_WIN (transport layers attend to |i-j| <= W) and P342_POOL_LAST (head mean over the last n
patches): unset == huge value (no-op), set changes the output; attention beyond W is exactly 0. usage: python pat342/test_win.py CKPT"""
import os, sys, torch; sys.path.insert(0, '.')
from pat336.props import load
m = load(sys.argv[1]).model; m.config.context_length = 2048; torch.manual_seed(0); x = torch.randn(1, 2048, 3, device='cuda')
def run(**env):
    for k in ('P342_WIN', 'P342_POOL_LAST'): os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    with torch.no_grad(): return m(past_values=x).prediction_outputs
y0 = run(); print('noop win', torch.equal(y0, run(P342_WIN=10 ** 6)), 'noop pool', torch.equal(y0, run(P342_POOL_LAST=10 ** 6)))
print('win63 diff', float((run(P342_WIN=63) - y0).abs().max()), 'pool64 diff', float((run(P342_POOL_LAST=64) - y0).abs().max()))
a = m.model.encoder.layers[3].temporal_self_attn; got = {}
h = a.register_forward_hook(lambda mod, i, o: got.__setitem__('w', o[1]))
os.environ['P342_WIN'] = '63'
with torch.no_grad(): m(past_values=x, output_attentions=True)
h.remove(); w = got['w']; N = w.shape[-1]; ii = torch.arange(N, device=w.device); far = (ii[:, None] - ii[None]).abs() > 63
print('layer3 pos', a.pos, 'N', N, 'max attn beyond 63', float(w[..., far].abs().max()), 'row sums ~1', float((w.sum(-1) - 1).abs().max()))
