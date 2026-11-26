"""nope1_lie_fox / nope1_path_fox: build from the nope1_lie config, check per-layer pos and fgate_proj, forward, save -> reload round trip
(all params equal, no missing / random keys; cf. HF beta rename)."""
import sys, json, tempfile, torch, transformers
sys.path.insert(0, '.')
from transformers import PatchTSTConfig
from panda.patchtst.patchtst import PatchTSTForPrediction
src = 'pat318/ckpt/p339_nope1_lie_s99_st100000-0/checkpoint-10000'
for arm in ('nope1_lie_fox', 'nope1_path_fox'):
    cfg = json.load(open(src + '/config.json')); [cfg.pop(k, None) for k in ('architectures', 'model_type', 'transformers_version', 'torch_dtype')]
    cfg['temporal_pos'] = arm; transformers.set_seed(99); m = PatchTSTForPrediction(PatchTSTConfig(**cfg)).cuda()
    L = [l.temporal_self_attn for l in m.model.encoder.layers]; print(arm, [a.pos for a in L], [hasattr(a, 'fgate_proj') for a in L])
    x = torch.randn(2, 512, 3, device='cuda'); y = m(past_values=x).prediction_outputs; print(' out', tuple(y.shape), bool(torch.isfinite(y).all()))
    d = tempfile.mkdtemp(); m.save_pretrained(d, safe_serialization=True); m2 = PatchTSTForPrediction.from_pretrained(d).cuda(); from pat318.exact_load import exact_load; exact_load(m2, d)
    sd, sd2 = m.state_dict(), m2.state_dict()
    print(" exact_load ok,", 'all equal', all(torch.equal(sd[k], sd2[k]) for k in sd))
