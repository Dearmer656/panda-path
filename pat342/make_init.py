"""PAT-342 control A: untrained nope1_lie (KS) model, built exactly as scripts/panda/train.py does (set_seed(99), then
PatchTSTForPrediction(PatchTSTConfig(**cfg))), config copied from the trained run. usage: python pat342/make_init.py SRC_CKPT DST"""
import sys, json, transformers
from transformers import PatchTSTConfig
from panda.patchtst.patchtst import PatchTSTForPrediction
src, dst = sys.argv[1], sys.argv[2]
cfg = json.load(open(src + "/config.json")); [cfg.pop(k, None) for k in ("architectures", "model_type", "transformers_version", "torch_dtype")]
transformers.set_seed(99); m = PatchTSTForPrediction(PatchTSTConfig(**cfg)); m.save_pretrained(dst, safe_serialization=True)
print("saved", dst, sum(p.numel() for p in m.parameters()))
