"""PAT-318 fox arm checks: (1) bias[i,j] == sum_{s in (min,max]} log f_s by brute force, both directions;
(2) gate bias -> +inf (f = 1) gives exactly the nope attention; (3) grads reach fgate_proj; (4) no fgate key contains beta/gamma (HF from_pretrained renames those)."""
import tempfile, torch
from panda.patchtst.patchtst import PatchTSTRopeAttention
torch.manual_seed(0)
E, NH, N, B = 32, 4, 9, 3
a = PatchTSTRopeAttention(E, NH, pos='fox').double(); n = PatchTSTRopeAttention(E, NH, pos='nope').double()
n.load_state_dict({k: v for k, v in a.state_dict().items() if not k.startswith('fgate')})
x = torch.randn(B, N, E, dtype=torch.float64, requires_grad=True)
cap = {}
orig = torch.bmm
def hook(m, i, o): cap[m] = o
out_a = a(x, output_attentions=True); out_n = n(x, output_attentions=True)
lf = torch.nn.functional.logsigmoid(a.fgate_proj(x)).transpose(1, 2).reshape(B * NH, N)
bias = torch.zeros(B * NH, N, N, dtype=torch.float64)
for i in range(N):
    for j in range(N):
        lo, hi = min(i, j), max(i, j); bias[:, i, j] = lf[:, lo + 1:hi + 1].sum(-1)
lg_n = torch.log(out_n[1].reshape(B * NH, N, N)); lg_a = torch.log(out_a[1].reshape(B * NH, N, N))
d = (lg_a - lg_n) - bias; d = d - d.mean(-1, keepdim=True)   # softmax is shift invariant per row
print('1 brute-force bias max err', d.abs().max().item()); assert d.abs().max() < 1e-5  # gate computed in float32
with torch.no_grad(): a.fgate_proj.bias.fill_(60.); a.fgate_proj.weight.zero_()
print('2 f=1 vs nope max err', (a(x)[0] - n(x)[0]).abs().max().item()); assert (a(x)[0] - n(x)[0]).abs().max() < 1e-10
a2 = PatchTSTRopeAttention(E, NH, pos='fox').double(); a2(x)[0].sum().backward()
print('3 grad fgate', a2.fgate_proj.weight.grad.abs().sum().item()); assert a2.fgate_proj.weight.grad.abs().sum() > 0
assert not [k for k in a.state_dict() if 'fgate' in k and ('beta' in k or 'gamma' in k)]; print('4 key names ok')
