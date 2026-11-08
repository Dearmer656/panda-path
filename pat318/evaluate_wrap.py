"""PAT-318: official scripts/panda/evaluate.py with two optional env hooks (no other change):
  P318_LOAD_SEED  seed torch's global RNG right before the model is built (the dynamics embedding's polynomial
                  patch_indices are a plain list, not saved in the checkpoint, and are re-drawn at construction).
  P318_CTX        evaluate with a longer context than training (K3): sets model.config.context_length after loading;
                  evaluate.py then builds test windows of that length and the pipeline keeps that many points.
                  The model mean-pools over patches, so nothing else depends on the context length.
"""
import os
import runpy
import sys

import transformers

from panda.patchtst import pipeline as _pl

_orig = _pl.PatchTSTPipeline.from_pretrained.__func__


def _wrapped(cls, *a, **k):
    if os.environ.get('P318_LOAD_SEED'):
        transformers.set_seed(int(os.environ['P318_LOAD_SEED']))
    p = _orig(cls, *a, **k)
    if os.environ.get('P318_CTX'):
        p.model.config.context_length = int(os.environ['P318_CTX'])
    return p


_pl.PatchTSTPipeline.from_pretrained = classmethod(_wrapped)
sys.argv[0] = 'scripts/panda/evaluate.py'
runpy.run_path('scripts/panda/evaluate.py', run_name='__main__')
