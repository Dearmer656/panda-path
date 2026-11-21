"""PAT-334 G2: signal vs hidden keys inside the same multihead segment retrieval (ctx 512, whole own history, val + test).
Candidates: anchors e = t - 16 m, e >= 512, e + H <= t (stride 16 = Panda patch). Every anchor e gets the query's treatment: the
frozen PaTH Panda encodes the 512-step window ending at e. Head j (j = 0..5) compares the j-th patch from the end:
  sig     : Z[e - 16 j - 16, e - 16 j) vs Z[t - 16 j - 16, t - 16 j), mean square
  hid_cos : 1 - cos of the j-th-from-last last-layer patch states (512-d);  hid_eu : squared Euclid of the same states
Structures per key: whole (sig: 96 window; hid: last patch state, = head 0), shared (one anchor, mean over the 6 heads of the
median-normalised distances), seg16 (6 independent heads, mixing as in mh.py). Per-channel 'c' and joint 'j' (per-channel
median-normalised distances summed over channels). Readout copy Z[e, e + H); greedy exclusion radius 32; k, weights, mixing
chosen on val median sMAPE; hid = better of hid_cos / hid_eu on val per structure and mode.
Criteria (fixed before results), seg16 at all H, per mode:
  G2 signal sufficient : sig_seg16 - hid_seg16 sMAPE CI upper end < +0.5 (non-inferior); signal better if CI < 0.
  reported: seg16 - shared and seg16 - whole for each key (does multihead help hidden keys too).
usage: P321_LQ=96 P321_OUT=pat321/data python pat334/mh_hid.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in ('pat327', 'pat329', 'pat330', 'pat333'): sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(ROOT))
import numpy as np, torch
import torch.nn.functional as F
import fc_corr as C  # noqa: E402
from mse_eval import per_traj  # noqa: E402
from trunc import CK  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); assert FC.CTX == 512
HS, KS, KM, R, NH = (128, 256, 512), (1, 3, 5, 10, 20), 20, 32, 6
TAUS = ('uni', 'best', 0.03, 0.1, 0.3, 1.0); KEYS = ('sig', 'hid_cos', 'hid_eu'); STR = ('whole', 'shared', 'seg16')
O = ROOT / 'pat334' / 'out'


@torch.no_grad()
def encode(p, X, ends):
    """X [C, T] raw; ends [n] -> last NH patch states, index j = j-th from the end: [n, C, NH, 512]."""
    out = []
    for i in range(0, len(ends), 128):
        e = ends[i:i + 128]
        win = X[:, e[:, None] - 512 + torch.arange(512, device=dev)].permute(1, 2, 0)
        h = p.model.model(past_values=win).last_hidden_state                               # [b, C, NPW, d]
        out.append(h[:, :, -NH:].flip(2))
    return torch.cat(out)


def nms(D, a):
    js, vs = [], []
    for _ in range(KM):
        v, j = D.min(0); js.append(j); vs.append(v)
        D = D.masked_fill((a[:, None] - a[j][None]).abs() <= R, float('inf'))
    return torch.stack(js, -1), torch.stack(vs, -1)                                      # [C, KM] indices into a


def normed(dd, ok):
    """dd [n, C] -> divided per channel by its median over ok candidates."""
    m = dd[ok].median(0).values.abs().clamp_min(1e-12); return dd / m


@torch.no_grad()
def run_set(name, p):
    d = torch.load(FC.OUT / f'{name}.pt'); S = d['S'].to(dev)
    E = {H: {} for H in HS}; order = []
    for i in range(d['n_inst']):
        q = torch.nonzero(d['ids'] == i)[:, 0]; order.append(q)
        rows = d['rows'][q].to(dev); t = int(d['t'][q[0]]); X = S[rows]; nc = len(q)
        w = X[:, t - 512:t]; cm, cs = w.mean(1), w.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        ends = (t - 16 * torch.arange(8, max(9, (t - 512) // 16 + 1), device=dev)).clamp(min=512)
        Hc = encode(p, X, ends); Hq = encode(p, X, torch.tensor([t], device=dev))[0]             # [n, C, NH, d], [C, NH, d]
        hd = {}
        for j in range(NH):
            sq = Z[:, t - 16 * j - 16:t - 16 * j]; sc = Z[:, (ends - 16 * j)[:, None] - 16 + torch.arange(16, device=dev)]   # [C, n, 16]
            hd[('sig', j)] = (sc - sq[:, None]).square().mean(-1).T                                                    # [n, C]
            hd[('hid_cos', j)] = 1 - F.cosine_similarity(Hc[:, :, j], Hq[None, :, j], dim=-1)
            hd[('hid_eu', j)] = (Hc[:, :, j] - Hq[None, :, j]).square().sum(-1)
        W96 = Z[:, ends[:, None] - 96 + torch.arange(96, device=dev)]; s96 = (W96 - Z[:, t - 96:t][:, None]).square().mean(-1).T
        for H in HS:
            ok = (ends + H <= t) & (ends >= 512) & (t - ends >= 128); has = bool(ok.any())
            f = (d['f'][q, :H].to(dev) - cm[:, None]) / cs[:, None]; y = Z[:, t:t + H]

            def put(k, pr):
                e = E[H].setdefault(k, ([], []))
                e[0].append(((pr - y).abs() / (pr.abs() + y.abs() + 1e-10)).mean(1).cpu()); e[1].append((pr - y).square().mean(1).cpu())
            put('f', f)
            if not has:
                for key in KEYS:
                    for st in STR:
                        for mode in 'cj':
                            for wn in ('u', 'w'):
                                for k in KS:
                                    for tau in (TAUS if st == 'seg16' else ('uni',)): put(f'{key}_{st}_{mode}|{wn}{k}|{tau}', f)
                continue
            fut = Z[:, ends[:, None] + torch.arange(H, device=dev)].clamp(-1000, 1000)                # [C, n, H]
            for key in KEYS:
                heads = {'whole': [s96 if key == 'sig' else hd[(key, 0)]],
                         'shared': [sum(normed(hd[(key, j)], ok) for j in range(NH)) / NH],
                         'seg16': [hd[(key, j)] for j in range(NH)]}
                for st, hl in heads.items():
                    for mode in 'cj':
                        HD = []
                        for dd in hl:
                            dm = (normed(dd, ok) if mode == 'j' else dd).masked_fill(~ok[:, None], float('inf'))   # [n, C]
                            if mode == 'j': dm = dm.sum(1, keepdim=True).expand(-1, nc)
                            js, vs = nms(dm, ends)                                                           # [C, KM]
                            med = dd[ok].median(0).values.abs().clamp_min(1e-12) if mode == 'c' else dm[ok][:, 0].median().abs().clamp_min(1e-12)
                            dn = vs[:, 0] / med
                            fu = fut[torch.arange(nc, device=dev)[:, None], js]                              # [C, KM, H]
                            val = torch.isfinite(vs); v1 = vs[:, :1]
                            ww = torch.exp(-(vs - v1) / v1.abs().clamp_min(1e-8)).masked_fill(~val, 0.0)
                            HD.append((fu, val, ww, dn))
                        for wn in ('u', 'w'):
                            for k in KS:
                                ph = torch.stack([((val[:, :k].float() if wn == 'u' else ww[:, :k])[..., None] * fu[:, :k]).sum(1)
                                                  / (val[:, :k].float() if wn == 'u' else ww[:, :k]).sum(1, keepdim=True).clamp_min(1e-12)
                                                  for fu, val, ww, dn in HD], 1)
                                dns = torch.stack([h[3] for h in HD], 1)
                                for tau in (TAUS if len(HD) > 1 else ('uni',)):
                                    if tau == 'uni': a = torch.ones_like(dns)
                                    elif tau == 'best': a = F.one_hot(dns.argmin(1), len(HD)).float()
                                    else: a = torch.softmax(-dns / tau, 1)
                                    put(f'{key}_{st}_{mode}|{wn}{k}|{tau}', (a[..., None] * ph).sum(1) / a.sum(1, keepdim=True))
    o = torch.cat(order); inv = torch.empty_like(o); inv[o] = torch.arange(len(o))
    return {H: {k: {'smape': per_traj(torch.cat(s)[inv], d['ids'], d['inst']), 'mse': per_traj(torch.cat(m)[inv], d['ids'], d['inst'])}
                for k, (s, m) in E[H].items()} for H in HS}


def med(x): return 100 * float(x['smape'].median())


def main():
    O.mkdir(exist_ok=True)
    p = FC.load_backbone(ROOT / CK['PaTH'], dev)
    V = run_set('val', p); T = run_set('test', p)
    for H in HS:
        A, pick = {'f': T[H]['f']}, {}
        for key in KEYS:
            for st in STR:
                for mode in 'cj':
                    cs = [k for k in V[H] if k.startswith(f'{key}_{st}_{mode}|')]; b = min(cs, key=lambda k: med(V[H][k]))
                    A[f'{key}_{st}_{mode}'] = T[H][b]; pick[f'{key}_{st}_{mode}'] = (b, med(V[H][b]))
        for st in STR:
            for mode in 'cj':
                hb = min(('hid_cos', 'hid_eu'), key=lambda kk: pick[f'{kk}_{st}_{mode}'][1])
                A[f'hid_{st}_{mode}'] = A[f'{hb}_{st}_{mode}']; pick[f'hid_{st}_{mode}'] = pick[f'{hb}_{st}_{mode}']
        print(f'== H{H}', flush=True)
        for k, (b, v) in pick.items(): print(f'   pick {k:18s} {b:34s} val sMAPE {v:6.2f}', flush=True)
        for k, v in A.items(): print(f'   test {k:18s} sMAPE {med(v):6.2f}  MSE {float(v["mse"].median()):.4f}', flush=True)
        rng = np.random.default_rng(0)
        pairs = [(f'sig_{st}_{m}', f'hid_{st}_{m}') for m in 'cj' for st in STR]
        pairs += [(f'{kk}_seg16_{m}', f'{kk}_{r}_{m}') for kk in ('sig', 'hid') for m in 'cj' for r in ('shared', 'whole')]
        for x, ref in pairs:
            o2 = []
            for met, sc in (('smape', 100), ('mse', 1)):
                d0, lo, hi, _ = C.boot(sc * A[x][met].numpy(), sc * A[ref][met].numpy(), rng); o2.append(f'{met} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
            print(f'   {x:18s} - {ref:18s} ' + ' | '.join(o2), flush=True)
        torch.save({'A': A, 'pick': pick}, O / f'mh_hid_H{H}.pt')


if __name__ == '__main__':
    main()
