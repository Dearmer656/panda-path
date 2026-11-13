"""PAT-321: run the official Panda eval (pat318/evaluate_wrap.py, unchanged behaviour) and record every sampled test window.
NumInstanceSampler.__call__ is wrapped to append (full target [dim, T], sampled indices) to P321_DUMP (one pickle per call,
written from inside the DataLoader worker, in iteration order). Sampling itself is untouched, so the windows and the
metrics are exactly those of the official run. usage: P321_DUMP=path python pat321/dump_test.py <evaluate.py hydra args>"""
import os
import pickle
import runpy
import sys

from panda import dataset as _s

_orig = _s.NumInstanceSampler.__call__


def _call(self, ts):
    idx = _orig(self, ts)
    with open(os.environ['P321_DUMP'], 'ab') as f:
        pickle.dump((ts.astype('float32'), idx.copy()), f)
    return idx


_s.NumInstanceSampler.__call__ = _call
sys.argv[0] = 'pat318/evaluate_wrap.py'
runpy.run_path('pat318/evaluate_wrap.py', run_name='__main__')
