"""PAT-338 rope1_lie: layer 0 RoPE, layers >= 1 lie (rank P336_LIE_RANK, default 4). Checks: per-layer types / params,
layer-wise equivalence with 'rope' and 'lie' models (fp64), save -> from_pretrained + exact_load round trip (same forecast)."""
import torch, tempfile
from transformers import PatchTSTConfig
from panda.patchtst.patchtst import PatchTSTForPrediction
from pat318.exact_load import exact_load
torch.manual_seed(0); dt = torch.float64
base = PatchTSTConfig.from_pretrained('pat318/ckpt/p318_path_s99_st100000-0/checkpoint-final')
def build(pos, L=3):
    c = PatchTSTConfig.from_dict(base.to_dict()); c.temporal_pos = pos; c.num_hidden_layers = L
    return PatchTSTForPrediction(c).to(dt).eval()
M = build('rope1_lie'); lay = M.model.encoder.layers
assert lay[0].temporal_self_attn.pos == 'rope' and not hasattr(lay[0].temporal_self_attn, 'lie_basis_proj')
assert all(l.temporal_self_attn.pos == 'lie' and l.temporal_self_attn.lie_rank == 4 for l in lay[1:]); print('1 ok')
R, Lm = build('rope'), build('lie'); sdR, sdL = R.state_dict(), Lm.state_dict()
M.load_state_dict({k: (sdR[k] if k.startswith('model.encoder.layers.0.') else sdL[k]) for k in M.state_dict()})
x = torch.randn(1, 3, 32, base.d_model, dtype=dt)
with torch.no_grad():
    for i, l in enumerate(lay): assert torch.allclose(l(x)[0], (R if i == 0 else Lm).model.encoder.layers[i](x)[0], atol=1e-12), i
print('2 ok')
ctx = torch.randn(1, 512, 3, dtype=dt)
with tempfile.TemporaryDirectory() as dd:
    M.save_pretrained(dd); M2 = PatchTSTForPrediction.from_pretrained(dd).to(dt).eval(); exact_load(M2, dd)
    for a, b in zip(M.modules(), M2.modules()):
        if hasattr(a, 'patch_indices'): b.patch_indices = a.patch_indices
    with torch.no_grad(): assert torch.allclose(M(past_values=ctx).prediction_outputs, M2(past_values=ctx).prediction_outputs, atol=1e-12)
print('3 ok')
