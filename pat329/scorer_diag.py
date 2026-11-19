"""How do the three FreshCast scorers (H = 128 / 256 / 512) weight their candidates? Val queries, weights w over the 400 entries.
Reports: effective number of entries 1/sum w^2, weight on value (vs shape) readouts, weighted mean cosine, weighted mean age
(t - s, steps), inverse temperature beta, and the Spearman correlation between each descriptor and the score (per query, averaged)."""
import os, sys, torch
os.environ.setdefault('P321_OUT', 'pat321/data'); sys.path.insert(0, 'pat321')
import fc_dysts as FC
dev = torch.device('cuda')
d, se = FC.load_set('val', dev)
names = ['value', 'cos', 'log_age', 'gap', 'ratio', 'acf16', 'acf96', 'lmsd']
def rank(x): return x.argsort(-1).argsort(-1).float()
for H in (128, 256, 512):
    m = FC.Scorer('fc').to(dev); m.load_state_dict(torch.load(f'pat321/data/scorer_fc_H{H}_s99.pt')['state']); m.eval()
    acc = dict(neff=[], wval=[], cos=[], age=[], beta=[]); sp = torch.zeros(5, device=dev); n = 0
    with torch.no_grad():
        for i in range(0, len(d['rows']), 256):
            idx = torch.arange(i, min(i + 256, len(d['rows'])))
            rows, t, b, _ = FC.batch_inputs(se, d, idx, H, dev, 'fc')
            keep = b['any']
            if not keep.any(): continue
            w = torch.nan_to_num(m(b))[keep]; z = b['z'][keep]; s = b['s'][keep]; tt = t[keep]
            zn = (b['z'] - m.zm) / m.zs
            sc = (m.lin(zn)[..., 0] + m.mlp(zn)[..., 0])[keep]
            beta = torch.exp(m.b0 + m.bnet((b['u'] - m.um) / m.us)[..., 0].clamp(-2, 2))[keep]
            acc['neff'].append(1 / w.square().sum(1)); acc['wval'].append((w * z[..., 0]).sum(1))
            acc['cos'].append((w * z[..., 1]).sum(1)); acc['age'].append((w * (tt[:, None] - s).float()).sum(1)); acc['beta'].append(beta)
            v = b['valid'][keep]
            for f in range(5):   # per-candidate descriptors only (the 3 history stats are constant within a query)
                x = z[..., f]; rx, rs = rank(x.masked_fill(~v, -1e9)), rank(sc.masked_fill(~v, -1e9))
                rx = rx - rx.mean(1, keepdim=True); rs = rs - rs.mean(1, keepdim=True)
                sp[f] += ((rx * rs).sum(1) / (rx.norm(dim=1) * rs.norm(dim=1)).clamp_min(1e-9)).sum()
            n += int(keep.sum())
    med = {k: float(torch.cat(v).median()) for k, v in acc.items()}
    print(f'H{H}: N_eff {med["neff"]:.1f} | weight on value readouts {med["wval"]:.2f} | mean cos {med["cos"]:.3f} | mean age {med["age"]:.0f} steps | beta {med["beta"]:.2f} | '
          + 'spearman(score, desc): ' + ' '.join(f'{names[f]} {float(sp[f] / n):+.2f}' for f in range(5)), flush=True)
