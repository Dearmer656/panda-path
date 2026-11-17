"""PAT-318: official scripts/panda/evaluate.py with two optional env hooks (no other change):
  P318_LOAD_SEED  seed torch's global RNG right before the model is built (the dynamics embedding's polynomial
                  patch_indices are a plain list, not saved in the checkpoint, and are re-drawn at construction).
  always     reload the checkpoint's tensors by their exact names (pat318/exact_load.py): transformers 4.40's from_pretrained
             renames any key containing 'beta' to 'bias', so PaTH's beta_proj was silently re-initialised at load.
  P318_LOGN       if set, multiply temporal-attention logits by log(N)/log(32) at inference (N = patches; 32 = training
                  length), the log-length softmax scaling against attention dilution. Identity at ctx 512. Only square
                  [bs*h, N, N] logits with N >= 32 are touched, so channel attention (N = number of channels) is unchanged.
  P318_CTX        evaluate with a longer context than training (K3): sets model.config.context_length after loading;
                  evaluate.py then builds test windows of that length and the pipeline keeps that many points.
                  The model mean-pools over patches, so nothing else depends on the context length.
"""
import os
import runpy
import sys

import transformers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pat318.exact_load import exact_load  # noqa: E402

from panda.patchtst import pipeline as _pl

_orig = _pl.PatchTSTPipeline.from_pretrained.__func__


def _wrapped(cls, *a, **k):
    if os.environ.get('P318_LOAD_SEED'):
        transformers.set_seed(int(os.environ['P318_LOAD_SEED']))
    p = _orig(cls, *a, **k)
    exact_load(p.model, k.get('pretrain_path', a[1] if len(a) > 1 else None))
    if os.environ.get('P318_CTX'):
        p.model.config.context_length = int(os.environ['P318_CTX'])
    return p


_pl.PatchTSTPipeline.from_pretrained = classmethod(_wrapped)

if os.environ.get('P318_LOGN'):
    import math
    import torch.nn.functional as _F
    _sm = _F.softmax

    def _softmax_logn(x, dim=None, *a, **k):
        if x.dim() == 3 and x.shape[-1] >= 32 and x.shape[-1] == x.shape[-2]:
            x = x * (math.log(x.shape[-1]) / math.log(32))
        return _sm(x, dim=dim, *a, **k)
    _F.softmax = _softmax_logn
if os.environ.get('P318_BETA_C') or os.environ.get('P318_BETA_SHUF'):
    import pat318.beta_interv  # noqa: F401  (eval-only beta scaling / shuffling, see that file)
sys.argv[0] = 'scripts/panda/evaluate.py'
runpy.run_path('scripts/panda/evaluate.py', run_name='__main__')
