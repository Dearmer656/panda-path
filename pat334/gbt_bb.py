"""PAT-334 Gbt step 1: Panda (frozen PaTH, last 512 steps, as the setting's backbone) forecasts at backtest origins
t'_b = t - 512 - 128 b (b = 0..3) of every test query; the backtest truth [t', t' + 512) lies before t. Valid only if the
Panda context [t' - 512, t') is inside the allowed history (ctx: >= t - CTX; full: >= 0).
usage: same env as mh.py; python pat334/gbt_bb.py b   (b in 0..3)   |   python pat334/gbt_bb.py chk   (sanity: b = origin t)"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import torch
import sig as G
from sig import FC, dev, backbone, TAG, MODE
from trunc import forecast, CK
O = ROOT / 'pat334' / 'out'; NB, DT = 4, 128


def origins(t, b): return t - 512 - DT * b


def valid(t, tp): return (tp - 512) >= ((t - G.CTX) if MODE == 'ctx' else 0)


if __name__ == '__main__':
    arg = sys.argv[1]; d = torch.load(FC.OUT / 'test.pt')
    p = FC.load_backbone(ROOT / CK['PaTH' if MODE == 'ctx' else 'RoPE'], dev); p.model.config.context_length = 512   # data/test.pt f was made with RoPE (PAT-321 default)
    if arg == 'chk':
        n = 2000; keep = d['ids'] < n; dd = {'S': d['S'], 'rows': d['rows'][keep], 't': d['t'][keep], 'ids': d['ids'][keep], 'n_inst': n}
        f0 = backbone('test', d)[keep]; f1 = forecast(p, dd)
        print(f'{TAG} sanity: recomputed vs stored backbone max|diff| {float((f0 - f1).abs().max()):.3e}, rel {float(((f0 - f1).abs() / f0.abs().clamp_min(1e-3)).median()):.2e}', flush=True)
    else:
        b = int(arg); tp = origins(d['t'], b); ok = valid(d['t'], tp)
        t0 = time.time(); dd = dict(d); dd['t'] = torch.where(ok, tp, d['t'])
        f = forecast(p, dd).half()
        torch.save({'f': f, 'ok': ok, 'tp': tp}, O / f'gbt_bb_{TAG}_{b}.pt')
        print(f'{TAG} b{b}: valid {float(ok.float().mean()):.3f}, {time.time() - t0:.0f}s', flush=True)
