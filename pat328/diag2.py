"""PAT-328 v2 diagnostics: (1) gradients reach every part (net / fy / E / dec) after a few steps; (2) 5-epoch train
loss with an affine transform (E with bias: the matrix can then also shift the future) vs the linear one."""
import sys, copy, math, torch
sys.path.insert(0, 'pat328')
import fc_mat as M
C, FC = M.C, M.FC
dev = torch.device('cuda'); H = 128
fcm = C.load_fc(H, dev)
d, se = FC.load_set('train', dev)
g = torch.Generator().manual_seed(99)
sel = torch.randperm(len(d['rows']), generator=g)[:32000]
sel, eps, _, _ = FC.untrained_eps(se, d, sel, H, dev)
for affine in (False, True):
    torch.manual_seed(99)
    m = M.Mat('pairm', H).to(dev)
    if affine: m.E = torch.nn.Linear(H, M.D, bias=True).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=FC.LR, weight_decay=FC.WD)
    for ep in range(5):
        perm = torch.randperm(len(sel), generator=g); tot = 0; n = 0
        for i in range(0, len(sel), 256):
            pi = perm[i:i + 256]; idx = sel[pi]
            rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx, H, dev, fcm)
            y = FC.target(se, rows, t, b, H)
            r = (w[..., None] * R).sum(1) + M.corr_of(m, se, rows, t, b, R, w, qp, cp, H)
            loss = ((r - y).square().mean(1) / eps[pi].to(dev)).clamp_max(FC.C_MAX).mean()
            opt.zero_grad(); loss.backward()
            if ep == 0 and n == 10:
                gn = {k: sum(float(p.grad.norm()) for p in mod.parameters() if p.grad is not None) for k, mod in [('net', m.net), ('fy', m.fy), ('E', m.E), ('dec', m.dec)]}
                print(f'affine={affine} grad norms at step 10: ' + ' '.join(f'{k} {v:.2e}' for k, v in gn.items()), flush=True)
            opt.step(); tot += float(loss); n += 1
        print(f'affine={affine} epoch {ep} train loss {tot / n:.4f}', flush=True)
