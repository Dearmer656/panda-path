"""PAT-342: what transform do the learned Lie transitions A_t = exp(Q_t^T m_t Q_t), m_t = dt_t (K_t + diag(lambda_t)), of an
early Panda checkpoint do?  Same inputs as pat336/props.py (official held-out trajectories, first file per system, all
channels, window x[:, T0 - ctx : T0], load seed 99). One row per (ckpt, ctx, layer with Lie transport); pooled over heads /
(sample, channel) sequences; median unless stated. Step t = 1..N-1 (A_0 is never used).

ROT  rotation per step: the skew part dt K (r = 4) has eigenvalues +-i theta_1, +-i theta_2 (theta_1 >= theta_2, rad / patch);
     rot_max = theta_1, rot_min = theta_2; share of steps with theta_1 > 0.1 rad; period 2 pi / theta_1 in patches.
STR  stretch per step: log-rates dt lambda_a (a = 1..r; lambda in (-1, 0.1), so dt lambda in (-2, 0.2));
     share contracting (< -0.05), ~0 (|.| <= 0.05), expanding (> 0.05); lam_min / lam_max = smallest / largest rate per step.
     cumulative log-stretch over d steps along the walk (sum of the per-step most-contracting / most-expanding rate,
     an upper bound of the log singular-value range of the product when Q_t is constant), d = 16, N - 1.
DT   dt = 2 sigmoid(.) in (0, 2).
SUB  subspace persistence ||Q_t Q_{t-1}^T||_F^2 / r (1 = same r-dim subspace every step, random in 64 dims: r / 64 = .0625);
     share of the subspace in the top-4 principal subspace of the keys of the same sequence-head (random .0625);
     |cos| of the subspace with k_t and with q_t (norm of the projection of the unit vector, random sqrt(r / 64) = .25).
SIG  Spearman over patches, within each sequence, of the head-mean theta_1 / mean dt lambda / dt with patch speed
     mean |x_{s+1} - x_s|; mean over sequences and share > 0.
LOG  score magnitude: max |S| and max |S0| (S0 = q k^T without transport) per sequence-head; ratio; attention mass on the
     past / future, mean |i - j|, entropy with and without transport.
usage: python -m pat342.what_lie OUT CKPT [CKPT ...]   (env P336_CTXS, P336_NSYS as props; P336_LIE_RANK must match)"""
import sys, os, csv, math, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat336.props import systems, load, transport, T0, CTXS
from pat340.what import spearman_rows, attn_stats
dev = torch.device('cuda'); PATCH = 16


def ucosQ(Q, v):
    """Q [n, N, r, E] orthonormal rows, v [n, N, E] -> |projection of unit v onto span Q| [n, N]."""
    v = torch.nn.functional.normalize(v, dim=-1); return (Q @ v[..., None]).squeeze(-1).norm(dim=-1)


@torch.no_grad()
def analyse(ck, data, rows):
    p = load(ck); m = p.model; layers = [l.temporal_self_attn for l in m.model.encoder.layers]
    print(f'== {ck}  temporal_pos = {[a.pos for a in layers]}', flush=True)
    for ctx in CTXS:
        m.config.context_length = ctx; acc = {}; cur = {}

        def hook(mod, args, kw, outp, i):
            a = layers[i]
            if not a.pos.startswith('lie'): return
            x = args[0] if args else kw['hidden_states']; o = transport(a, x, kw.get('path_ctx'))
            A = acc.setdefault(i, {}); ap = lambda kk, v: A.setdefault(kk, []).append(v.flatten().float().cpu())
            q, k, Q, S, S0 = o['q'], o['k'], o['Q'][:, 1:], o['S'], o['S0']; n, N, E = q.shape; H = a.num_heads; r = Q.shape[2]
            mm = o['m'][:, 1:].float(); Kp = (mm - mm.transpose(-1, -2)) / 2; Sp = (mm + mm.transpose(-1, -2)) / 2
            th = torch.linalg.eigvals(Kp).imag.abs().sort(-1, descending=True).values[..., ::2]   # [n, N-1, r/2] theta_1 >= theta_2
            ap('rot_max', th[..., 0]); ap('rot_min', th[..., -1]); ap('s_rot_gt.1', (th[..., 0] > .1).float())
            ap('period_max', 2 * math.pi / th[..., 0].clamp_min(1e-6))
            lr = torch.linalg.eigvalsh(Sp)                                    # dt lambda (diag in the Q basis) [n, N-1, r]
            ap('lam', lr); ap('s_contr', (lr < -.05).float()); ap('s_zero', (lr.abs() <= .05).float()); ap('s_expand', (lr > .05).float())
            ap('lam_min', lr[..., 0]); ap('lam_max', lr[..., -1]); ap('dt', o['dt'][:, 1:])
            for d in sorted({16, N - 1}):
                ap(f'cum_min{d}', lr[..., 0][:, :d].sum(-1)); ap(f'cum_max{d}', lr[..., -1][:, :d].sum(-1))
            ap('sub_persist', (Q[:, 1:] @ Q[:, :-1].transpose(-1, -2)).pow(2).sum((-1, -2)) / r)
            kc = k - k.mean(1, keepdim=True); U = torch.linalg.svd(kc, full_matrices=False)[2][:, :4]   # [n, 4, E]
            ap('sub_in_k4', (Q @ U.transpose(1, 2)[:, None]).pow(2).sum((-1, -2)) / r)
            ap('cos_k', ucosQ(Q, k[:, 1:])); ap('cos_q', ucosQ(Q, q[:, 1:]))
            ap('orth_err', o['orth'])
            ap('maxS', S.abs().amax((1, 2))); ap('maxS0', S0.abs().amax((1, 2)))
            ap('ratio_maxS', S.abs().amax((1, 2)) / S0.abs().amax((1, 2)).clamp_min(1e-12))
            ii = torch.arange(N, device=x.device); D = ii[:, None] - ii[None]
            for nm, PP in (('', torch.softmax(S.float(), -1)), ('0', torch.softmax(S0.float(), -1))):
                for kk, v in attn_stats(PP, D).items(): ap(kk + nm, v)
            sig = cur['sig'][:, 1:, 0]                                        # speed [bsz, N-1]; sequences batch-major, heads inner
            for nm, v in (('rot', th[..., 0]), ('lam', lr.mean(-1)), ('dt', o['dt'][:, 1:])):
                vh = v.reshape(-1, H, N - 1).mean(1); ap('rS_' + nm, spearman_rows(vh, sig))
        hs = [a.register_forward_hook(lambda mod, args, kw, outp, i=i: hook(mod, args, kw, outp, i), with_kwargs=True) for i, a in enumerate(layers)]
        for name, x in data:
            xx = x[:, T0 - ctx:T0]; pt = torch.tensor(xx, device=dev).view(xx.shape[0], -1, PATCH)
            cur['sig'] = (pt[..., 1:] - pt[..., :-1]).abs().mean(-1)[..., None]
            m(past_values=torch.tensor(xx, device=dev).T[None])
        for h in hs: h.remove()
        for i, A in sorted(acc.items()):
            row = dict(ckpt=ck, ctx=ctx, layer=i, pos=layers[i].pos)
            for kk, v in A.items():
                v = torch.cat(v); v = v[~v.isnan()]
                if v.numel() == 0: continue
                row[kk] = float(v.mean()) if kk.startswith(('s_', 'past', 'fut', 'dist', 'near', 'last', 'ent')) else float(v.median())
                if kk in ('maxS', 'ratio_maxS'): row[kk + '_p99'] = float(v.quantile(.99)) if v.numel() < 2 ** 24 else float('nan')
                if kk == 'lam': row['lam_p05'], row['lam_p95'] = float(v.quantile(.05)), float(v.quantile(.95))
                if kk.startswith('rS_'): row[kk + '_pos'] = float((v > 0).float().mean())
            rows.append(row); print({k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in row.items() if k_ != 'ckpt'}, flush=True)


if __name__ == '__main__':
    out = sys.argv[1]; data = systems(); rows = []
    for ck in sys.argv[2:]: analyse(ck, data, rows)
    keys = sorted({k for r in rows for k in r}, key=lambda s: (s not in ('ckpt', 'ctx', 'layer', 'pos'), s))
    with open(out + '_what_lie.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in rows]
