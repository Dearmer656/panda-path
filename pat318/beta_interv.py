"""PAT-318 beta intervention (evaluation only; no weight, w, mask or head change). Imported by evaluate_wrap.py when
P318_BETA_C or P318_BETA_SHUF is set. It wraps the bidir_path_scores that PaTH attention actually calls, so the modified
beta is the one entering the path matrix product H_t = I - beta_t w_t w_t^T.
  P318_BETA_C=c       beta -> clip(c * beta, 0, 2)            (c = 1 is the identity: beta = 2 sigmoid(.) is already in (0, 2))
  P318_BETA_SHUF=s    beta permuted across time positions, independently for every (sample, head) row of each layer,
                      seeded torch generator s; per-row value multiset unchanged, w unchanged.
Only positions 1..N-1 are touched and counted: beta_0 never enters the score (causal walk uses H_i..H_{j+1},
the reversed walk uses H_{j}..H_{i+1} with j > i; see path_pos.py), so permuting it in would change the used distribution.
P318_BETA_STATS=path  per-layer json: mean / std of beta before and after, fraction clipped (c * beta > 2).
"""
import atexit, json, os
import torch
import panda.patchtst.patchtst as PT

C = float(os.environ.get('P318_BETA_C', '1'))
SHUF = os.environ.get('P318_BETA_SHUF')
NL = int(os.environ.get('P318_NLAYERS', '8'))
from panda.patchtst.path_pos import bidir_path_scores as _orig  # the true function, never a previous hook
_gen = {}
_st = [dict(n=0, s0=0., q0=0., s1=0., q1=0., clip=0) for _ in range(NL)]
_calls = [0]


def _hook(q, k, w, beta):
    layer = _calls[0] % NL; _calls[0] += 1
    b = beta[:, 1:]
    raw = C * b                                      # in beta.dtype: c = 1 is bit-exact
    nb = raw.clamp(0, 2)
    if SHUF is not None:
        if beta.device not in _gen:
            _gen[beta.device] = torch.Generator(device=beta.device).manual_seed(int(SHUF))
        perm = torch.argsort(torch.rand(nb.shape, generator=_gen[beta.device], device=nb.device), dim=1)
        nb = torch.gather(nb, 1, perm)
    st = _st[layer]
    st['n'] += b.numel(); st['s0'] += b.float().sum().item(); st['q0'] += (b.float() ** 2).sum().item()
    st['s1'] += nb.float().sum().item(); st['q1'] += (nb.float() ** 2).sum().item(); st['clip'] += (raw > 2).sum().item()
    return _orig(q, k, w, torch.cat([beta[:, :1], nb], 1))


PT.bidir_path_scores = _hook


@atexit.register
def _dump():
    out = dict(c=C, shuffle_seed=SHUF, calls=_calls[0], layers=[])
    for st in _st:
        n = max(st['n'], 1)
        m0, m1 = st['s0'] / n, st['s1'] / n
        out['layers'].append(dict(n=st['n'], mean_orig=m0, std_orig=max(st['q0'] / n - m0 ** 2, 0) ** .5,
                                  mean=m1, std=max(st['q1'] / n - m1 ** 2, 0) ** .5, clip_frac=st['clip'] / n))
    T = sum(s['n'] for s in _st) or 1
    out['all'] = dict(mean=sum(s['s1'] for s in _st) / T, std=max(sum(s['q1'] for s in _st) / T - (sum(s['s1'] for s in _st) / T) ** 2, 0) ** .5,
                      mean_orig=sum(s['s0'] for s in _st) / T, clip_frac=sum(s['clip'] for s in _st) / T)
    p = os.environ.get('P318_BETA_STATS')
    if p:
        os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, 'w'), indent=1)
    print('[beta_interv]', json.dumps(out['all']), 'calls', _calls[0], flush=True)
