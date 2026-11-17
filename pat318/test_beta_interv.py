"""PAT-318: unit checks of pat318/beta_interv.py on random inputs and on the real path checkpoint's forward.
(1) beta_0 never affects the scores; (2) c=1 hook == original exactly; (3) c=0 == plain q.k; (4) shuffle keeps each row's
multiset of beta[1:] and changes the scores; (5) in the real model the hook is reached 8x per forward and changes the output."""
import os, sys, importlib, torch
sys.path.insert(0, '.')
import panda.patchtst.patchtst as PT
from panda.patchtst.path_pos import bidir_path_scores as ref
torch.manual_seed(0)
B, N, E = 6, 32, 64
q, k = torch.randn(B, N, E, dtype=torch.float64), torch.randn(B, N, E, dtype=torch.float64)
w = torch.nn.functional.normalize(torch.randn(B, N, E, dtype=torch.float64), dim=-1); beta = 2 * torch.rand(B, N, dtype=torch.float64)
b2 = beta.clone(); b2[:, 0] = 1.7
print('beta_0 unused: max diff', (ref(q, k, w, beta) - ref(q, k, w, b2)).abs().max().item())
def load(c=None, s=None):
    for v in ('P318_BETA_C', 'P318_BETA_SHUF'): os.environ.pop(v, None)
    if c is not None: os.environ['P318_BETA_C'] = str(c)
    if s is not None: os.environ['P318_BETA_SHUF'] = str(s)
    PT.bidir_path_scores = ref
    import pat318.beta_interv as M; M = importlib.reload(M); return M
M = load(1); print('c=1 vs orig max diff', (PT.bidir_path_scores(q, k, w, beta) - ref(q, k, w, beta)).abs().max().item())
M = load(0); print('c=0 vs qk max diff', (PT.bidir_path_scores(q, k, w, beta) - q @ k.transpose(1, 2)).abs().max().item())
M = load(1.5); s15 = PT.bidir_path_scores(q, k, w, beta); bb = beta.clone(); bb[:, 1:] = (1.5 * beta[:, 1:]).clamp(0, 2)
print('c=1.5 vs manual clip max diff', (s15 - ref(q, k, w, bb)).abs().max().item(), 'clip frac', M._st[0]['clip'] / M._st[0]['n'], 'expected', (1.5 * beta[:, 1:] > 2).double().mean().item())
M = load(1, 3); cap = []
orig = M._orig
def spy(q_, k_, w_, b_): cap.append(b_.clone()); return orig(q_, k_, w_, b_)
M._orig = spy; ss = PT.bidir_path_scores(q, k, w, beta)
sb = cap[0]
print('shuffle: multiset equal', torch.equal(sb[:, 1:].sort(1).values, beta[:, 1:].sort(1).values), 'beta_0 kept', torch.equal(sb[:, 0], beta[:, 0]),
      'positions moved frac', (sb[:, 1:] != beta[:, 1:]).double().mean().item(), 'score change', (ss - ref(q, k, w, beta)).abs().max().item())
if len(sys.argv) > 1:  # real checkpoint forward on random context
    from panda.patchtst.pipeline import PatchTSTPipeline
    from pat318.exact_load import exact_load
    import transformers
    outs = {}
    for name, c, s in (('none', None, None), ('c1', 1, None), ('c0', 0, None), ('shuf', 1, 0)):
        M = load(c, s) if (c is not None) else None
        if M is None: PT.bidir_path_scores = ref
        transformers.set_seed(99); p = PatchTSTPipeline.from_pretrained(mode='predict', pretrain_path=sys.argv[1]); exact_load(p.model, sys.argv[1]); p.model.cuda().eval()
        x = torch.randn(4, 512, 1, generator=torch.Generator().manual_seed(1)).cumsum(1).cuda()
        with torch.no_grad(): outs[name] = p.model(past_values=x).prediction_outputs.float().cpu()
        if M is not None: print(name, 'calls', M._calls[0])
    for n in ('c1', 'c0', 'shuf'): print(f'model {n} vs none: max diff', (outs[n] - outs['none']).abs().max().item())
