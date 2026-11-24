"""PAT-337 checks for temporal_pos = path2 (python -m pat337.test_path2 from the panda root).
1. bidir_path_scores with wb = w, bb = beta == original PaTH.
2. path2 scores == brute force: j <= i  q_i^T H_i ... H_{j+1} k_j ; j > i  q_i^T Hb_{i+1} ... Hb_j k_j (fp64).
3. module: path2 with path_wb_proj / path_sb_proj copied from w_proj / beta_proj == path module exactly.
4. gradients reach both reflection sets; causal half does not depend on the backward set.
5. save -> from_pretrained round trip keeps the path2 tensors (HF key renames)."""
import torch, tempfile
from panda.patchtst.path_pos import bidir_path_scores
from panda.patchtst.patchtst import PatchTSTRopeAttention
torch.manual_seed(0); dt = torch.float64
B, N, E = 3, 9, 5
q, k = torch.randn(B, N, E, dtype=dt), torch.randn(B, N, E, dtype=dt)
nz = lambda: torch.nn.functional.normalize(torch.randn(B, N, E, dtype=dt), dim=-1)
w, wb = nz(), nz(); be, bb = 2 * torch.rand(B, N, dtype=dt), 2 * torch.rand(B, N, dtype=dt)
assert torch.allclose(bidir_path_scores(q, k, w, be, w, be), bidir_path_scores(q, k, w, be), atol=1e-12); print('1 ok')
I = torch.eye(E, dtype=dt); H = I - be[..., None, None] * w[..., :, None] * w[..., None, :]; Hb = I - bb[..., None, None] * wb[..., :, None] * wb[..., None, :]
Sb = torch.zeros(B, N, N, dtype=dt)
for i in range(N):
    for j in range(N):
        P = I.expand(B, E, E).clone()
        if j <= i:
            for s in range(j + 1, i + 1): P = H[:, s] @ P
        else:
            for s in range(i + 1, j + 1): P = P @ Hb[:, s]
        Sb[:, i, j] = torch.einsum('be,bef,bf->b', q[:, i], P, k[:, j])
S = bidir_path_scores(q, k, w, be, wb, bb)
print('2 max |closed - brute|', float((S - Sb).abs().max())); assert torch.allclose(S, Sb, atol=1e-10)
a = PatchTSTRopeAttention(embed_dim=32, num_heads=4, pos='path2').to(dt); b = PatchTSTRopeAttention(embed_dim=32, num_heads=4, pos='path').to(dt)
sd = a.state_dict(); sd['path_wb_proj.weight'], sd['path_wb_proj.bias'] = sd['w_proj.weight'].clone(), sd['w_proj.bias'].clone()
sd['path_sb_proj.weight'], sd['path_sb_proj.bias'] = sd['beta_proj.weight'].clone(), sd['beta_proj.bias'].clone(); a.load_state_dict(sd)
b.load_state_dict({kk: v for kk, v in sd.items() if 'path_' not in kk})
x = torch.randn(4, 11, 32, dtype=dt); assert torch.allclose(a(x)[0], b(x)[0], atol=1e-12); print('3 ok')
a = PatchTSTRopeAttention(embed_dim=32, num_heads=4, pos='path2').to(dt); a(x.requires_grad_())[0].square().mean().backward()
assert all(float(getattr(a, n).weight.grad.abs().sum()) > 0 for n in ('w_proj', 'beta_proj', 'path_wb_proj', 'path_sb_proj'))
S1 = bidir_path_scores(q, k, w, be, wb, bb); S2 = bidir_path_scores(q, k, w, be, nz(), 2 * torch.rand(B, N, dtype=dt))
assert torch.allclose(S1.tril(), S2.tril(), atol=1e-12) and not torch.allclose(S1.triu(1), S2.triu(1)); print('4 ok')
from transformers import PatchTSTConfig
from panda.patchtst.patchtst import PatchTSTForPrediction
c = PatchTSTConfig.from_pretrained('pat318/ckpt/p318_path_s99_st100000-0/checkpoint-final'); c.num_hidden_layers = 2; c.temporal_pos = 'path2'
m = PatchTSTForPrediction(c); sd = {kk: v.clone() for kk, v in m.state_dict().items() if 'path_wb' in kk or 'path_sb' in kk or 'beta_proj' in kk}
with tempfile.TemporaryDirectory() as dd:
    m.save_pretrained(dd); m2 = PatchTSTForPrediction.from_pretrained(dd); from pat318.exact_load import exact_load; exact_load(m2, dd)  # beta_proj needs exact_load (HF rename)
sd2 = m2.state_dict(); assert len(sd) == 12 and all(kk in sd2 and torch.equal(sd[kk], sd2[kk]) for kk in sd); print('5 ok', len(sd), 'tensors')
