import sys, time, torch
sys.path.insert(0, "pat334")
sys.argv = ['x']; import mh_hid as M
d = torch.load(M.FC.OUT / 'val.pt'); dt = torch.load(M.FC.OUT / 'test.pt')
print('n_inst val', d['n_inst'], 'test', dt['n_inst'], 't range', int(d['t'].min()), int(d['t'].max()), flush=True)
p = M.FC.load_backbone(M.ROOT / M.CK['PaTH'], M.dev)
orig = torch.load
def small(f, *a, **k):
    x = orig(f, *a, **k)
    if isinstance(x, dict) and 'n_inst' in x: x = dict(x); x['n_inst'] = 4
    return x
M.torch.load = small
t0 = time.time(); M.run_set('val', p) if False else None
import torch as T
try:
    M.run_set('val', p)
except Exception as e: print('err (expected at per_traj)', type(e).__name__, flush=True)
print('sec per inst', (time.time() - t0) / 4, flush=True)
