"""PAT-318 checks (run from the panda root: python -m pat318.test_path).
1. causal_path_scores == explicit Householder products H_i ... H_{j+1}, j <= i.
2. causal PaTH attention == fla naive_path_attn (official reference, g = 0).
3. bidir_path_scores == explicit products in both directions (j > i: H_{i+1} ... H_j).
4. beta = 0 -> 'path' attention == 'nope' attention exactly.
5. 'rope' attention (pos=None/use_rope) unchanged: equals a by-hand official formula; grads reach w_proj/beta_proj.
6. Released Panda checkpoint loads with temporal_pos defaulting to rope and gives the same forecast as git HEAD code.
"""
import subprocess
import sys

import torch

from panda.patchtst.modules import apply_p_rope_to_qk
from panda.patchtst.patchtst import PatchTSTRopeAttention
from panda.patchtst.path_pos import bidir_path_scores, causal_path_scores

import importlib.util  # noqa: E402  (load fla's naive reference file directly; the fla package import needs triton)
_spec = importlib.util.spec_from_file_location('fla_naive', '../fla_clean_ref/fla/ops/path_attn/naive.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
naive_path_attn = _m.naive_path_attn

torch.manual_seed(0)
dt = torch.float64
B, N, E = 3, 9, 6


def explicit(q, k, w, beta):
    S = torch.zeros(q.shape[0], N, N, dtype=dt)
    I = torch.eye(E, dtype=dt)
    for b in range(q.shape[0]):
        for i in range(N):
            for j in range(N):
                P = I.clone()
                ts = range(j + 1, i + 1) if j <= i else range(i + 1, j + 1)   # j<=i: H_i...H_{j+1}; j>i: H_{i+1}...H_j
                for t in ts:
                    Ht = I - beta[b, t] * torch.outer(w[b, t], w[b, t])
                    P = Ht @ P if j <= i else P @ Ht
                S[b, i, j] = q[b, i] @ P @ k[b, j]
    return S


q, k = torch.randn(B, N, E, dtype=dt), torch.randn(B, N, E, dtype=dt)
w = torch.nn.functional.normalize(torch.randn(B, N, E, dtype=dt), dim=-1)
beta = 2 * torch.rand(B, N, dtype=dt)
ref = explicit(q, k, w, beta)
tril = torch.ones(N, N, dtype=torch.bool).tril()

e1 = float((causal_path_scores(q, k, w, beta) - ref).masked_fill(~tril, 0).abs().max())
print('1 causal vs explicit', e1); assert e1 < 1e-12

v = torch.randn(B, N, E, dtype=dt)
o_ref = naive_path_attn(q[:, :, None], k[:, :, None], v[:, :, None], w[:, :, None], beta[:, :, None],
                        torch.zeros(B, N, 1, dtype=dt), scale=1.0, chunk_size=4)[:, :, 0].to(dt)
S = causal_path_scores(q, k, w, beta).masked_fill(~tril, float('-inf'))
e2 = float((S.softmax(-1) @ v - o_ref).abs().max())
print('2 causal attn vs fla naive', e2); assert e2 < 1e-5

e3 = float((bidir_path_scores(q, k, w, beta) - ref).abs().max())
print('3 bidir vs explicit', e3); assert e3 < 1e-12

D, Hh, L = 16, 2, 7
x = torch.randn(2, L, D, dtype=dt)
a = PatchTSTRopeAttention(D, Hh, pos='path').to(dt)
torch.nn.init.zeros_(a.beta_proj.weight); torch.nn.init.constant_(a.beta_proj.bias, -1e4)
b = PatchTSTRopeAttention(D, Hh, pos='nope').to(dt)
b.load_state_dict({n: p for n, p in a.state_dict().items() if not n.startswith(('w_proj', 'beta_proj'))})
e4 = float((a(x)[0] - b(x)[0]).abs().max())
print('4 beta=0 path vs nope', e4); assert e4 < 1e-12

r = PatchTSTRopeAttention(D, Hh, use_rope=True, max_wavelength=500, rope_percent=0.75).to(dt)
assert r.pos == 'rope'
qh = (r.q_proj(x) * (D // Hh) ** -0.5).view(2, L, Hh, -1).transpose(1, 2).reshape(2 * Hh, L, -1)
kh = r.k_proj(x).view(2, L, Hh, -1).transpose(1, 2).reshape(2 * Hh, L, -1)
vh = r.v_proj(x).view(2, L, Hh, -1).transpose(1, 2).reshape(2 * Hh, L, -1)
kh, qh = apply_p_rope_to_qk(kh, qh, torch.arange(L, dtype=dt), D // Hh, 500, 0.75)
o = (qh @ kh.transpose(1, 2)).softmax(-1) @ vh
o = r.out_proj(o.view(2, Hh, L, -1).transpose(1, 2).reshape(2, L, D))
e5 = float((r(x)[0] - o).abs().max())
print('5 rope vs by-hand official', e5); assert e5 < 1e-12
rp = PatchTSTRopeAttention(D, Hh, pos='rope_path').to(dt)
rp(x)[0].square().sum().backward()
assert rp.w_proj.weight.grad.abs().sum() > 0 and rp.beta_proj.weight.grad.abs().sum() > 0
print('5 rope_path grads ok')

# 6: released checkpoint, modified code vs git HEAD code (in a subprocess on a clean worktree)
snippet = ("import torch,numpy as np;from panda.patchtst.pipeline import PatchTSTPipeline as P;torch.manual_seed(0);"
           "m=P.from_pretrained(mode='predict',pretrain_path='{ck}',device_map='cpu');torch.manual_seed(0);"
           "x=torch.randn(512,3);print(repr(m.predict(x,128,limit_prediction_length=True).double().sum().item()))")
ck = __import__('os').path.abspath('data/ckpt_panda')
cur = subprocess.run([sys.executable, '-c', snippet.format(ck=ck)], capture_output=True, text=True)
subprocess.run('rm -rf /tmp/_p318_head && git worktree add -f /tmp/_p318_head HEAD -q', shell=True, check=True)
head = subprocess.run([sys.executable, '-c', snippet.format(ck=ck)], capture_output=True, text=True, cwd='/tmp/_p318_head',
                      env={**__import__('os').environ, 'PYTHONPATH': '/tmp/_p318_head'})
subprocess.run('git worktree remove --force /tmp/_p318_head', shell=True)
c, h = float(cur.stdout.split()[-1]), float(head.stdout.split()[-1])
print('6 released ckpt forecast sum: current', c, 'HEAD', h); assert c == h, (cur.stderr[-500:], head.stderr[-500:])
print('all ok')
