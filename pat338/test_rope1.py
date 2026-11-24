"""PAT-338 checks for temporal_pos = rope1_path (python -m pat338.test_rope1 from the panda root).
1. layer 0 temporal attention is RoPE (no PaTH params), layers >= 1 are path (3-layer model); channel attention unchanged.
2. layer-wise equivalence: with weights copied, layer 0 == a 'rope' model's layer 0 and layer l >= 1 == a 'path' model's
   layer l on the same input (fp64).
3. state-dict keys = path model's keys minus layer 0's w_proj / beta_proj.
4. save -> from_pretrained + exact_load round trip; the reloaded model gives the same forecast.
5. nope1_path: layer 0 is nope; old configs (path, rope) build exactly as before."""
import torch, tempfile
from transformers import PatchTSTConfig
from panda.patchtst.patchtst import PatchTSTForPrediction
from pat318.exact_load import exact_load
torch.manual_seed(0); dt = torch.float64
base = PatchTSTConfig.from_pretrained('pat318/ckpt/p318_path_s99_st100000-0/checkpoint-final')
def build(pos, L=8):
    c = PatchTSTConfig.from_dict(base.to_dict()); c.temporal_pos = pos; c.num_hidden_layers = L
    return PatchTSTForPrediction(c).to(dt).eval()
M = build('rope1_path', 3); lay = M.model.encoder.layers
assert lay[0].temporal_self_attn.pos == 'rope' and not hasattr(lay[0].temporal_self_attn, 'w_proj')
assert all(l.temporal_self_attn.pos == 'path' and hasattr(l.temporal_self_attn, 'w_proj') for l in lay[1:]); print('1 ok')
R, P = build('rope', 3), build('path', 3)
sdR, sdP = R.state_dict(), P.state_dict()
sd = {k: (sdR[k] if k.startswith('model.encoder.layers.0.') else sdP[k]) for k in M.state_dict()}
M.load_state_dict(sd)
x = torch.randn(1, 3, 32, base.d_model, dtype=dt)
with torch.no_grad():
    for i, l in enumerate(lay):
        ref = (R if i == 0 else P).model.encoder.layers[i]
        a, b = l(x)[0], ref(x)[0]; assert torch.allclose(a, b, atol=1e-12), i
print('2 ok')
kM, kP = set(M.state_dict()), set(sdP)
assert kP - kM == {k for k in kP if k.startswith('model.encoder.layers.0.temporal_self_attn.') and ('w_proj' in k or 'beta_proj' in k)} and not kM - kP; print('3 ok', len(kP - kM), 'keys dropped')
ctx = torch.randn(1, 512, 3, dtype=dt)
with tempfile.TemporaryDirectory() as dd:
    M.save_pretrained(dd); M2 = PatchTSTForPrediction.from_pretrained(dd).to(dt).eval(); exact_load(M2, dd)
    assert M2.config.temporal_pos == 'rope1_path'
    for a, b in zip(M.modules(), M2.modules()):          # Panda's poly patch_indices are drawn at build, not saved
        if hasattr(a, 'patch_indices'): b.patch_indices = a.patch_indices
    with torch.no_grad(): assert torch.allclose(M(past_values=ctx).prediction_outputs, M2(past_values=ctx).prediction_outputs, atol=1e-12)
print('4 ok')
N = build('nope1_path', 2); assert N.model.encoder.layers[0].temporal_self_attn.pos == 'nope' and N.model.encoder.layers[1].temporal_self_attn.pos == 'path'
assert all(l.temporal_self_attn.pos == 'path' for l in P.model.encoder.layers) and all(l.temporal_self_attn.pos == 'rope' for l in R.model.encoder.layers); print('5 ok')
