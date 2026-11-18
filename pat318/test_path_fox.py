"""PAT-318 path_fox arm checks: (1) path_fox logits == path logits + fox bias exactly (same q/k/v/w/beta/fgate weights);
(2) f = 1 gives exactly the path arm; (3) grads reach fgate_proj, w_proj, beta_proj; (4) no rope; key names ok (no beta/gamma in fgate keys)."""
import torch
from panda.patchtst.patchtst import PatchTSTRopeAttention
torch.manual_seed(0)
E, NH, N, B = 32, 4, 9, 3
pf = PatchTSTRopeAttention(E, NH, pos='path_fox').double(); pa = PatchTSTRopeAttention(E, NH, pos='path').double(); fx = PatchTSTRopeAttention(E, NH, pos='fox').double()
sd = pf.state_dict()
pa.load_state_dict({k: v for k, v in sd.items() if not k.startswith('fgate')})
fx.load_state_dict({k: v for k, v in sd.items() if not (k.startswith('w_proj') or k.startswith('beta_proj'))})
nope = PatchTSTRopeAttention(E, NH, pos='nope').double(); nope.load_state_dict({k: v for k, v in sd.items() if k.split('.')[0] in ('q_proj', 'k_proj', 'v_proj', 'out_proj')})
x = torch.randn(B, N, E, dtype=torch.float64)
L = lambda m: torch.log(m(x, output_attentions=True)[1].reshape(B * NH, N, N))
d = (L(pf) - L(pa)) - (L(fx) - L(nope)); d = d - d.mean(-1, keepdim=True)
print('1 path_fox = path + fox bias, max err', d.abs().max().item()); assert d.abs().max() < 1e-5
with torch.no_grad(): pf.fgate_proj.bias.fill_(60.); pf.fgate_proj.weight.zero_()
print('2 f=1 vs path max err', (pf(x)[0] - pa(x)[0]).abs().max().item()); assert (pf(x)[0] - pa(x)[0]).abs().max() < 1e-10
p2 = PatchTSTRopeAttention(E, NH, pos='path_fox').double(); p2(x)[0].sum().backward()
for n_ in ('fgate_proj', 'w_proj', 'beta_proj'):
    g = getattr(p2, n_).weight.grad.abs().sum().item(); print('3 grad', n_, g); assert g > 0
assert not p2.use_rope and not [k for k in sd if 'fgate' in k and ('beta' in k or 'gamma' in k)]; print('4 ok')
