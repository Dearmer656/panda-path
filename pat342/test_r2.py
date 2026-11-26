"""PAT-342 r = 2 Lie (one rotation plane per step + 2 stretches): build nope1_lie with P336_LIE_RANK=2 from the nope1_lie config, check
per-layer pos / rank / gen width, forward, save -> reload round trip (all params equal). Run with P336_LIE_RANK=2."""
import os, sys, json, tempfile, torch, transformers
sys.path.insert(0, '.')
from transformers import PatchTSTConfig
from panda.patchtst.patchtst import PatchTSTForPrediction
from pat318.exact_load import exact_load
assert os.environ.get('P336_LIE_RANK') == '2'
src = 'pat318/ckpt/p339_nope1_lie_s99_st100000-0/checkpoint-10000'
cfg = json.load(open(src + '/config.json')); [cfg.pop(k, None) for k in ('architectures', 'model_type', 'transformers_version', 'torch_dtype')]
transformers.set_seed(99); m = PatchTSTForPrediction(PatchTSTConfig(**cfg)).cuda()
L = [l.temporal_self_attn for l in m.model.encoder.layers]
print([a.pos for a in L], [getattr(a, 'lie_rank', None) for a in L], [getattr(a, 'lie_mode', None) for a in L])
print(' gen_proj out', [a.lie_gen_proj.out_features for a in L if hasattr(a, 'lie_gen_proj')], 'basis_proj out', [a.lie_basis_proj.out_features for a in L if hasattr(a, 'lie_basis_proj')])
x = torch.randn(2, 512, 3, device='cuda'); y = m(past_values=x).prediction_outputs; print(' out', tuple(y.shape), bool(torch.isfinite(y).all()))
y.float().pow(2).mean().backward(); print(' grad gen', all(a.lie_gen_proj.weight.grad is not None for a in L if hasattr(a, 'lie_gen_proj')))
d = tempfile.mkdtemp(); m.save_pretrained(d, safe_serialization=True); m2 = PatchTSTForPrediction.from_pretrained(d).cuda(); exact_load(m2, d)
sd, sd2 = m.state_dict(), m2.state_dict(); print(' exact_load ok, all equal', all(torch.equal(sd[k], sd2[k]) for k in sd))
