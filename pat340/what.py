"""PAT-340: what do the learned PaTH transitions H_t = I - beta_t w_t w_t^T of a trained Panda checkpoint do?

Same inputs as pat336/props.py (official held-out trajectories, first file per system, all channels jointly, window
x[:, T0 - ctx : T0], load seed 99). Temporal attention runs per (sample, channel) sequence; one row of statistics per
transport layer, pooled over heads / sequences (median unless stated). Bidirectional use: for j < i the key k_j is carried
forward through H_{j+1} .. H_i; for j > i it is carried backward through H_j .. H_{i+1} (original path: the SAME symmetric
H_t, so "forward" and "backward" differ only in the order of the product; path2: separate (w_b, beta_b) for the backward walk).

A  operator type      share of beta in (0, .1) ~ identity, (.8, 1.2) ~ projection that erases w_t, (1.8, 2) ~ reflection;
                      the eigenvalue of H_t along w_t is 1 - beta_t (|1 - beta| < 1: contraction of that one direction).
B  what w_t points at |cos| of w_t with q_t, k_t, k_{t-1}, the key change k_t - k_{t-1}; random unit vector
                      in E = 64 dims: E|cos| = 0.0995. Share of w_t energy in the top-4 principal subspace of the keys of
                      the same sequence-head (random: 4 / 64 = .0625).
C  persistence        |cos(w_t, w_{t-1})|; participation ratio of sum_t beta_t w_t w_t^T (1 = one fixed direction is
                      erased again and again -> direction-selective exponential forgetting; up to 64 = isotropic).
D  retention          rho(d) = |P k_j| / |k_j| after d transitions (d = 1, 4, 16, N-1), forward and backward walks;
                      score effect: (S - q k^T) regressed on distance; attention mass on the past vs the future and mean
                      |i - j| with transport vs the same q, k without transport (softmax(q k^T)).
E  path2 only         |cos(w_t, wb_t)|, Spearman(beta_t, beta_b_t), beta_b operator shares.
F  link to the signal Spearman over patches, within each sequence, of beta_t with the input patch's speed
                      mean |x_{s+1} - x_s|, its std, and the jump |mean_p - mean_{p-1}|; mean over sequences (and share > 0).
usage: python -m pat340.what OUT CKPT [CKPT ...]   (env P336_CTXS, P336_NSYS as props)"""
import sys, os, csv, math, torch
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat336.props import systems, load, transport, T0, CTXS
dev = torch.device('cuda'); PATCH = 16


def ucos(a, b): return (torch.nn.functional.normalize(a, dim=-1) * torch.nn.functional.normalize(b, dim=-1)).sum(-1).abs()


def spearman_rows(a, b):
    """a, b [n, T] -> Spearman per row [n] (nan if constant)."""
    ra = a.argsort(-1).argsort(-1).float(); rb = b.argsort(-1).argsort(-1).float()
    ra = ra - ra.mean(-1, keepdim=True); rb = rb - rb.mean(-1, keepdim=True)
    den = (ra.norm(dim=-1) * rb.norm(dim=-1)); return torch.where(den > 0, (ra * rb).sum(-1) / den.clamp_min(1e-12), torch.full_like(den, float('nan')))


def walk(k, w, be, backward):
    """explicit fp32 transport of every key; returns retention [n, N, N] r[d] indexed by (target i, source j)."""
    n, N, E = k.shape; R = torch.full((n, N, N), float('nan'), device=k.device)
    V = k.clone(); nk = k.norm(dim=-1)
    order = range(1, N) if not backward else range(N - 2, -1, -1)
    for t in order:
        # forward: keys j < t pass H_t; backward: keys j > t pass H_{t+1}
        src = slice(0, t) if not backward else slice(t + 1, N); ht = t if not backward else t + 1
        v = V[:, src]; ww = w[:, ht][:, None]; v = v - be[:, ht][:, None, None] * ww * (v * ww).sum(-1, keepdim=True)
        V[:, src] = v; R[:, t, src] = v.norm(dim=-1) / nk[:, src].clamp_min(1e-12)
    return R


@torch.no_grad()
def analyse(ck, data, wr):
    p = load(ck); m = p.model; layers = [l.temporal_self_attn for l in m.model.encoder.layers]
    print(f'== {ck}  temporal_pos = {[a.pos for a in layers]}', flush=True)
    for ctx in CTXS:
        m.config.context_length = ctx; acc = {}; cur = {}
        def hook(mod, args, kw, outp, i):
            a = layers[i]
            x = args[0] if args else kw['hidden_states']; pc = kw.get('path_ctx'); o = transport(a, x, pc)
            if 'path' not in a.pos:
                A = acc.setdefault(i, {}); P = torch.softmax(o['S'], -1); N = P.shape[-1]; ii = torch.arange(N, device=x.device); D = ii[:, None] - ii[None]
                for kk, v in attn_stats(P, D).items(): A.setdefault(kk, []).append(v.flatten().float().cpu())
                return
            q, k, w, be, S, S0 = o['q'], o['k'], o['w'], o['beta'], o['S'], o['S0']; n, N, E = q.shape; H = a.num_heads
            A = acc.setdefault(i, {}); ap = lambda kk, v: A.setdefault(kk, []).append(v.flatten().float().cpu())
            b = be[:, 1:]; ww = w[:, 1:]                                         # H_0 is never used
            ap('b_id', (b < .1).float()); ap('b_proj', ((b > .8) & (b < 1.2)).float()); ap('b_refl', (b > 1.8).float()); ap('beta', b)
            ap('eig', (1 - b).abs())
            ap('cos_q', ucos(ww, q[:, 1:])); ap('cos_k', ucos(ww, k[:, 1:])); ap('cos_kprev', ucos(ww, k[:, :-1])); ap('cos_dk', ucos(ww, k[:, 1:] - k[:, :-1]))
            kc = k - k.mean(1, keepdim=True); U = torch.linalg.svd(kc, full_matrices=False)[2][:, :4]  # [n, 4, E]
            ap('w_in_k4', ((ww @ U.transpose(1, 2)) ** 2).sum(-1))
            ap('cos_wprev', ucos(ww[:, 1:], ww[:, :-1]))
            M = (b[..., None, None] * ww[..., :, None] * ww[..., None, :]).sum(1); ev = torch.linalg.eigvalsh(M).clamp_min(0)
            ap('pr', ev.sum(-1) ** 2 / (ev ** 2).sum(-1).clamp_min(1e-12))
            sel = torch.arange(0, n, max(1, n // 64), device=x.device)[:64]
            Rf = walk(k[sel], w[sel], be[sel], False)
            if a.pos == 'path2': Rb = walk(k[sel], o_wb(a, x, pc)[sel], o['beta_b'][sel], True)
            else: Rb = walk(k[sel], w[sel], be[sel], True)
            ii = torch.arange(N, device=x.device); D = ii[:, None] - ii[None]
            for d in sorted({1, 4, 16, 31, 63, N - 1} & set(range(1, N))):
                ap(f'retF{d}', Rf[:, D == d]); ap(f'retB{d}', Rb[:, D == -d])
            P, P0 = torch.softmax(S, -1), torch.softmax(S0, -1)
            for nm, PP in (('', P), ('0', P0)):
                for kk, v in attn_stats(PP, D).items(): ap(kk + nm, v)
            dS = (S - S0); ap('dS_abs', dS.abs().mean((1, 2)) / S0.abs().mean((1, 2)).clamp_min(1e-12))
            for nm, msk in (('near', (D.abs() >= 1) & (D.abs() <= 2)), ('far', D.abs() >= N // 2)):
                ap('dSrel_' + nm, (dS.abs() * msk).sum((1, 2)) / ((S0.abs() * msk).sum((1, 2))).clamp_min(1e-12))
            if a.pos == 'path2':
                wbv = o_wb(a, x, pc); bb = o['beta_b'][:, 1:]
                ap('cos_wwb', ucos(ww, wbv[:, 1:])); ap('rho_bbb', spearman_rows(b, bb))
                ap('bb_id', (bb < .1).float()); ap('bb_proj', ((bb > .8) & (bb < 1.2)).float()); ap('bb_refl', (bb > 1.8).float()); ap('beta_b', bb)
            # F: link to the input signal; sequences are (sample, channel) batch-major, heads inner
            sig = cur['sig']                                                   # [bsz, N, 3] speed, std, jump
            bh = be.view(-1, H, N).mean(1)                                     # head-mean beta per sequence [bsz, N]
            for j, nm in enumerate(('speed', 'std', 'jump')):
                r = spearman_rows(bh[:, 1:], sig[:, 1:, j]); ap('rF_' + nm, r)
        hs = [a.register_forward_hook(lambda mod, args, kw, outp, i=i: hook(mod, args, kw, outp, i), with_kwargs=True) for i, a in enumerate(layers)]
        for name, x in data:
            xx = x[:, T0 - ctx:T0]                                             # [C, ctx]
            pt = torch.tensor(xx, device=dev).view(xx.shape[0], -1, PATCH)     # [C, N, 16]
            mu = pt.mean(-1); sd = pt.std(-1); sp = (pt[..., 1:] - pt[..., :-1]).abs().mean(-1)
            jump = torch.cat([torch.zeros_like(mu[:, :1]), (mu[:, 1:] - mu[:, :-1]).abs()], 1)
            cur['sig'] = torch.stack([sp, sd, jump], -1)
            m(past_values=torch.tensor(xx, device=dev).T[None])
        for h in hs: h.remove()
        for i, A in sorted(acc.items()):
            row = dict(ckpt=ck, ctx=ctx, layer=i, pos=layers[i].pos)
            for kk, v in A.items():
                v = torch.cat(v); v = v[~v.isnan()]
                if v.numel() == 0: continue
                row[kk] = float(v.mean()) if kk.startswith(('b_', 'bb_', 'past', 'fut', 'dist', 'near', 'last', 'ent')) else float(v.median())
                if kk.startswith('rF_') or kk == 'rho_bbb': row[kk + '_pos'] = float((v > 0).float().mean())
            wr.writerow(row); print({k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in row.items() if k_ != 'ckpt'}, flush=True)


def attn_stats(P, D):
    N = P.shape[-1]; ent = -(P * P.clamp_min(1e-30).log()).sum(-1) / math.log(N)
    return {'past': (P * (D > 0)).sum(-1).mean(-1), 'fut': (P * (D < 0)).sum(-1).mean(-1), 'dist': (P * D.abs()).sum(-1).mean(-1),
            'near16': (P * (D.abs() <= 16)).sum(-1).mean(-1), 'ent': ent.mean(-1), 'last_near16': (P[:, -1] * (D[-1].abs() <= 16)).sum(-1)}


def o_wb(a, x, pc):
    bsz, N, _ = x.shape; H, E = a.num_heads, a.head_dim; hp = x if pc is None else pc
    return torch.nn.functional.normalize(a._shape(a.path_wb_proj(hp), -1, bsz).view(bsz * H, N, E), dim=-1)


if __name__ == '__main__':
    out = sys.argv[1]; data = systems(); rows = []
    keys = None
    class W:
        def writerow(self, r): rows.append(r)
    for ck in sys.argv[2:]: analyse(ck, data, W())
    keys = sorted({k for r in rows for k in r}, key=lambda s: (s not in ('ckpt', 'ctx', 'layer', 'pos'), s))
    with open(out + '_what.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in rows]
