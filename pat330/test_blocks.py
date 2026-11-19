"""PAT-330 check: block descriptors in build() equal per-segment MSE of fc_corr.phis windows (independent path), and the
first 8 descriptors are unchanged vs BLOCKS = 0. usage: P330_BLOCKS=6 P321_OUT=pat321/data python pat330/test_blocks.py"""
import os, sys, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327'))
import torch
import fc_corr as C
FC = C.FC; dev = torch.device('cuda'); NB = FC.BLOCKS; assert NB == 6
d, se = FC.load_set('val', dev)
idx = torch.arange(0, len(d['rows']), max(1, len(d['rows']) // 64))[:64]
rows, t = d['rows'][idx].to(dev), d['t'][idx].to(dev)
for H in (128, 512):
    b = FC.build(se, rows, t, H)
    q, c = C.phis(se, rows, t, b)
    e = (c - q[:, None]).square().reshape(len(idx), -1, NB, FC.LQ // NB).mean(-1)
    got = torch.exp(b['z'][..., 8:]) - 1e-4
    m = b['valid'] & (c.abs() < FC.CLIP - 1).all(-1)
    err = ((got - e).abs() / (e.abs() + 1e-3))[m].max().item()
    print(f'H{H} block descriptor max rel err {err:.2e} over {int(m.sum())} entries', flush=True); assert err < 1e-3
    os.environ['P330_BLOCKS'] = '0'; FC0 = importlib.reload(FC); b0 = FC0.build(se, rows, t, H); os.environ['P330_BLOCKS'] = '6'; importlib.reload(FC)
    assert torch.equal(b0['z'], b['z'][..., :8]) and torch.equal(b0['valid'], b['valid']); print(f'H{H} base descriptors unchanged', flush=True)
print('ok')
