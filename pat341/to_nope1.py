"""PAT-341: turn a trained all-layer path checkpoint into a nope1_path checkpoint by removing layer 0's transport
(drops layers.0 temporal w_proj / beta_proj -> layer 0 temporal attention = plain softmax(q k^T), i.e. NoPE); every other
tensor is copied unchanged. usage: python pat341/to_nope1.py SRC_CKPT DST_CKPT"""
import sys, json, shutil, re
from pathlib import Path
from safetensors.torch import load_file, save_file
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
sd = load_file(str(src / 'model.safetensors'))
pat = re.compile(r'(^|\.)layers\.0\.temporal_self_attn\.(w_proj|beta_proj)\.')
drop = [k for k in sd if pat.search(k)]
assert len(drop) in (2, 4), drop
for k in drop: del sd[k]
save_file(sd, str(dst / 'model.safetensors'), metadata={'format': 'pt'})
cfg = json.loads((src / 'config.json').read_text()); assert cfg['temporal_pos'] == 'path', cfg['temporal_pos']
cfg['temporal_pos'] = 'nope1_path'; (dst / 'config.json').write_text(json.dumps(cfg, indent=2))
for f in ('generation_config.json', 'training_info.json'):
    if (src / f).exists(): shutil.copy(src / f, dst / f)
print('dropped', drop, '->', dst, len(sd), 'tensors')
