"""PAT-335 checks for temporal_pos = path_shared (run from the panda root: python -m pat335.test_shared).
1. attention: path_ctx = hidden_states reproduces 'path' exactly; path_ctx = None is the old code path.
2. encoder layer, path_shared: w / beta inputs identical across channels (channel-mean), works for C = 1, 3, 7.
3. identical channels -> path_shared == path layer with the same weights.
4. channel-permutation equivariance of the path_shared layer.
5. gradients reach w_proj / beta_proj through the pooled context."""
import torch
from transformers import PatchTSTConfig
from panda.patchtst.patchtst import PatchTSTRopeAttention, PatchTSTEncoderLayerWithRope

torch.manual_seed(0)
dt = torch.float64


def cfg(pos):
    c = PatchTSTConfig(d_model=32, num_attention_heads=4, ffn_dim=32, channel_attention=True, norm_type='rmsnorm',
                       pre_norm=True, attention_dropout=0.0, path_dropout=0.0, ff_dropout=0.0, dropout=0.0)
    c.temporal_pos = pos; c.rope_percent = 0.75; c.max_wavelength = 500; c.channel_rope = False
    return c


# 1
a = PatchTSTRopeAttention(embed_dim=32, num_heads=4, pos='path_shared').to(dt).eval()
b = PatchTSTRopeAttention(embed_dim=32, num_heads=4, pos='path').to(dt).eval(); b.load_state_dict(a.state_dict())
x = torch.randn(5, 9, 32, dtype=dt)
assert torch.allclose(a(x, path_ctx=x)[0], b(x)[0], atol=1e-12) and torch.allclose(a(x)[0], b(x)[0], atol=1e-12)
print('1 ok')

# 2-4
L = PatchTSTEncoderLayerWithRope(cfg('path_shared')).to(dt).eval()
P = PatchTSTEncoderLayerWithRope(cfg('path')).to(dt).eval(); P.load_state_dict(L.state_dict())
seen = []
h = L.temporal_self_attn.w_proj.register_forward_hook(lambda m, i, o: seen.append(i[0].detach()))
for C in (1, 3, 7):
    seen.clear(); z = torch.randn(2, C, 9, 32, dtype=dt); L(z)
    ctx = seen[0].view(2, C, 9, 32)
    assert torch.allclose(ctx, ctx[:, :1].expand_as(ctx), atol=1e-12)
    zn = L.norm_sublayer1(z.reshape(2 * C, 9, 32)).view(2, C, 9, 32).mean(1)
    assert torch.allclose(ctx[:, 0], zn, atol=1e-12)
h.remove(); print('2 ok')
z1 = torch.randn(2, 1, 9, 32, dtype=dt).expand(2, 3, 9, 32).contiguous()
assert torch.allclose(L(z1)[0], P(z1)[0], atol=1e-10); print('3 ok')
z = torch.randn(2, 3, 9, 32, dtype=dt); pm = torch.tensor([2, 0, 1])
assert torch.allclose(L(z[:, pm])[0], L(z)[0][:, pm], atol=1e-10); print('4 ok')
L.train(); z = torch.randn(2, 3, 9, 32, dtype=dt); L(z)[0].sum().backward()
assert L.temporal_self_attn.w_proj.weight.grad.abs().sum() > 0 and L.temporal_self_attn.beta_proj.weight.grad.abs().sum() > 0
z3 = torch.randn(2, 3, 9, 32, dtype=dt); assert not torch.allclose(L(z3)[0], P(z3)[0], atol=1e-6)
print('5 ok; all checks passed')
