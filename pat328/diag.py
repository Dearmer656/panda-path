"""PAT-328 diagnostic: how large is the learned correction vs the residual it should fix? Dec / (M - I) magnitudes."""
import sys, torch
sys.path.insert(0, 'pat328')
import fc_mat as M
C, FC = M.C, M.FC
dev = torch.device('cuda')
d, se = FC.load_set('val', dev)
for H in (128, 512):
    fcm = C.load_fc(H, dev)
    idx = torch.arange(0, 2048)
    for arm in ('pairm', 'pathm'):
        ck = torch.load(M.OUT / f'mat_{arm}_H{H}_s99.pt')
        m = M.Mat(arm, H).to(dev); m.load_state_dict(ck['state']); m.eval()
        cs, res, mi = [], [], []
        with torch.no_grad():
            for i in range(0, len(idx), 128):
                rows, t, b, R, w, zn, qp, cp = C.step_inputs(se, d, idx[i:i + 128], H, dev, fcm)
                y = FC.target(se, rows, t, b, H); r0 = (w[..., None] * R).sum(1)
                c = M.corr_of(m, se, rows, t, b, R, w, qp, cp, H)
                cs.append(c.abs().mean(1)); res.append((y - r0).abs().mean(1))
                U, V = m.uv(qp[:, None].expand_as(cp), cp, R) if arm == 'pairm' else (None, None)
                if U is not None: mi.append((U @ V.transpose(-1, -2)).flatten(-2).norm(dim=-1).mean())
        cs, res = torch.cat(cs), torch.cat(res)
        print(f'H{H} {arm}: best ep {ck["best_ep"]} | mean|corr| {cs.mean():.4f}  mean|y - r_only| {res.mean():.4f}  ratio {cs.mean() / res.mean():.4f} '
              f'| |Dec.W| {m.dec.weight.norm():.3f} |E.W| {m.E.weight.norm():.3f}' + (f' |A-I|_F {torch.stack(mi).mean():.4f}' if mi else ''), flush=True)
