"""PAT-342 review tasks 2 + 3 (Linear PAT-335 review 035861eb): no-retraining interventions on a trained Lie-KS checkpoint and
exact stretch of the ordered products. Same inputs as pat336/props.py (official held-out trajectories, first file per system,
all channels, window x[:, T0 - ctx : T0], load seed 99).

Interventions (env P342_LIE_INTV, applied inside the model forward so all downstream layers see them; see patchtst.py):
  none | k0 (K = 0) | s0 (Lambda = 0) | id (A_t = I, plain q k^T) | perm (one fixed permutation of steps 1..N-1, applied jointly
  to basis and gen, i.e. Q_t, K_t, Lambda_t, dt_t move together).
Per (ckpt, intervention, ctx, Lie layer): attention mass and mean logit in |i - j| bins 0, 1, 2-4, 5-16, 17-64, 65+.
For intervention none only (task 3):
  SVD  log sigma_max / log sigma_min of the ordered ambient-space product Phi(j0 + d <- j0) = A_{j0+d} ... A_{j0+1} in fp64, for
       d = 1, 4, 16, 31, 63, 127 (< N - j0), starts j0 in {0, N/4, N/2}, on 32 sequence-heads per system and layer; the anticausal
       (adjoint) product is Phi^T with the same singular values.
  RAW  percentiles (1, 50, 99) of the raw generator outputs: omega (skew entries), lambda_raw (pre-tanh), dt_raw (pre-sigmoid);
       share lambda_raw > 0 (expansion branch); max |Q Q^T - I|.
  HEAD per-head median of dt*lambda contraction share (< -0.05), expansion share (> 0.05), theta_1 (rad / patch).
usage: python -m pat342.intv_attn OUT CKPT [CKPT ...]   (env P342_INTVS=none,k0,s0,id,perm  P336_CTXS  P336_NSYS)"""
import sys, os, csv, math, json, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat336.props import systems, load, qk, T0, CTXS
from panda.patchtst.lie_pos import lie_factors, lowrank_scores, LMAX
dev = torch.device('cuda'); PATCH = 16
INTVS = os.environ.get('P342_INTVS', 'none,k0,s0,id,perm').split(',')
BINS = [(0, 0), (1, 1), (2, 4), (5, 16), (17, 64), (65, 10 ** 9)]


def lie_inputs(a, x):
    bsz, N, _ = x.shape; H, E, r = a.num_heads, a.head_dim, a.lie_rank
    basis = a.lie_basis_proj(x).view(bsz, N, H, r, E).transpose(1, 2).reshape(bsz * H, N, r, E)
    gen = a.lie_gen_proj(x).view(bsz, N, H, -1).transpose(1, 2).reshape(bsz * H, N, -1)
    return basis, gen


def scores(a, x, intv):
    """same computation as the model forward under intervention intv (fp32)."""
    q, k, _, _ = qk(a, x); q, k = q.float(), k.float(); S0 = q @ k.transpose(1, 2)
    if intv == 'id': return S0, S0, None
    basis, gen = lie_inputs(a, x); basis, gen = basis.float(), gen.float(); N = x.shape[1]
    if intv == 'perm':
        g = torch.Generator().manual_seed(int(os.environ.get('P342_PERM_SEED', 0)))
        pi = torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(N - 1, generator=g)]).to(x.device); basis, gen = basis[:, pi], gen[:, pi]
    mode = {'k0': 's', 's0': 'k'}.get(intv, a.lie_mode)
    Q, G = lie_factors(basis, gen, a.lie_rank, mode)
    return lowrank_scores(q, k, Q, G), S0, (Q, G, gen)


@torch.no_grad()
def analyse(ck, data, rows):
    p = load(ck); m = p.model; layers = [l.temporal_self_attn for l in m.model.encoder.layers]
    print(f'== {ck}  temporal_pos = {[a.pos for a in layers]}', flush=True)
    for intv in INTVS:
        os.environ['P342_LIE_INTV'] = '' if intv == 'none' else intv
        for ctx in CTXS:
            m.config.context_length = ctx; acc = {}

            def hook(mod, args, kw, outp, i):
                a = layers[i]
                if not a.pos.startswith('lie'): return
                x = args[0] if args else kw['hidden_states']; S, S0, extra = scores(a, x, intv)
                A = acc.setdefault(i, {}); ap = lambda kk, v: A.setdefault(kk, []).append(v.flatten().float().cpu())
                n, N, _ = S.shape; ii = torch.arange(N, device=x.device); D = (ii[:, None] - ii[None]).abs(); P = torch.softmax(S, -1)
                for lo, hi in BINS:
                    msk = (D >= lo) & (D <= hi)
                    if not msk.any(): continue
                    ap(f'mass_{lo}', (P * msk).sum(-1).mean(-1)); ap(f'logit_{lo}', (S * msk).sum((1, 2)) / msk.sum())
                if intv != 'none' or extra is None: return
                Q, G, gen = extra; r = a.lie_rank; nk = r * (r - 1) // 2; H = a.num_heads
                om, lr, dr = gen[:, 1:, :nk], gen[:, 1:, nk:nk + r], gen[:, 1:, -1]
                for nm, v in (('om', om), ('lamraw', lr), ('dtraw', dr)):
                    vv = v.flatten(); vv = vv[torch.randperm(vv.numel(), device=vv.device)[:200000]]
                    for qq in (.01, .5, .99): ap(f'raw_{nm}_p{int(qq * 100)}', vv.quantile(qq)[None])
                ap('raw_lam_pos', (lr > 0).float().mean()[None])
                ap('orth_max', (Q @ Q.transpose(-1, -2) - torch.eye(r, device=x.device)).abs().amax()[None])
                # per-head generator statistics
                tl = torch.tanh(lr); lam = torch.where(tl > 0, LMAX * tl, tl); dt = 2 * torch.sigmoid(dr)
                dl = dt[..., None] * lam                                       # [n, N-1, r]
                iu = torch.triu_indices(r, r, 1, device=x.device); K = torch.zeros(*om.shape[:-1], r, r, device=x.device)
                K[..., iu[0], iu[1]] = om; K = K - K.transpose(-1, -2)
                th = torch.linalg.eigvals(dt[..., None, None] * K).imag.abs().amax(-1)
                hv = lambda v: v.reshape(-1, H, *v.shape[1:]).transpose(0, 1).reshape(H, -1)
                ap('head_contr', hv((dl < -.05).float()).mean(-1)); ap('head_expand', hv((dl > .05).float()).mean(-1)); ap('head_rot', hv(th).median(-1).values)
                # exact ordered products in fp64
                E = Q.shape[-1]; sel = torch.arange(0, n, max(1, n // 32), device=x.device)[:32]
                Qs, Gs = Q[sel].double(), G[sel].double()
                At = torch.eye(E, dtype=torch.float64, device=x.device) + Qs.transpose(-1, -2) @ Gs @ Qs   # A_t, t = 0..N-1
                for j0 in sorted({0, N // 4, N // 2}):
                    Phi = torch.eye(E, dtype=torch.float64, device=x.device).expand(len(sel), E, E).clone()
                    for d in range(1, N - j0):
                        Phi = At[:, j0 + d] @ Phi
                        if d in (1, 4, 16, 31, 63, 127):
                            sv = torch.linalg.svdvals(Phi); ap(f'lsmax_d{d}', sv[:, 0].log()); ap(f'lsmin_d{d}', sv[:, -1].log())
            hs = [a.register_forward_hook(lambda mod, args, kw, outp, i=i: hook(mod, args, kw, outp, i), with_kwargs=True) for i, a in enumerate(layers)]
            for name, x in data:
                xx = x[:, T0 - ctx:T0]; m(past_values=torch.tensor(xx, device=dev).T[None])
            for h in hs: h.remove()
            for i, A in sorted(acc.items()):
                row = dict(ckpt=ck, intv=intv, ctx=ctx, layer=i)
                for kk, v in A.items():
                    v = torch.cat(v)
                    if kk.startswith('head_'): row[kk] = json.dumps([round(float(t), 3) for t in v.view(-1, layers[i].num_heads).mean(0)])
                    elif kk.startswith('raw_') or kk == 'orth_max': row[kk] = float(v.amax()) if kk == 'orth_max' else float(v.mean())
                    else: v = v[~v.isnan()]; row[kk] = float(v.mean()) if kk.startswith(('mass', 'logit')) else float(v.median())
                    if kk.startswith('ls'): row[kk + '_p05'], row[kk + '_p95'] = float(v.quantile(.05)), float(v.quantile(.95))
                rows.append(row); print({k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in row.items() if k_ != 'ckpt'}, flush=True)
    os.environ['P342_LIE_INTV'] = ''


if __name__ == '__main__':
    out = sys.argv[1]; data = systems(); rows = []
    for ck in sys.argv[2:]: analyse(ck, data, rows)
    keys = sorted({k for r in rows for k in r}, key=lambda s: (s not in ('ckpt', 'intv', 'ctx', 'layer'), s))
    with open(out + '_intv.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in rows]
