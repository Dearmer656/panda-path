"""PAT-342: cumulative rotation of the learned Lie transport over long distances (does the q-k transport reach rotations
never seen at the training length?). Same inputs as pat336/props.py (official held-out trajectories, first file per
system, all channels, window x[:, T0 - ctx : T0], load seed 99), ctx 512 (N = 32, the training length) and 2048 (N = 128).

Per (ckpt, ctx, transport layer; Lie or, as control, PaTH H_t = I - beta w w^T), on 32 sequence-heads per system, starts j0 in {0, N/4}, distances d in {1, 4, 16, 31, 63, 127}
(< N - j0), with the exact ordered ambient product Phi_d = A_{j0+d} ... A_{j0+1} in fp64:
  polar   rotation of the directions that survive: Phi_d = U S V^T, over the singular pairs with sigma_i >= SMIN = 0.1 the largest
          angle(u_i, v_i) in [0, pi] (where an input direction v_i ends up; a flip counts as pi). The full polar factor U V^T is not
          used: on near-null directions (PaTH beta ~ 1, Lie log sigma_min ~ -23 at d = 127) it is numerical noise.
  nkeep   number of surviving singular pairs
  net     angle between the key k_{j0} and Phi_d k_{j0} (in [0, pi]; the direction change the transport applies to that key)
  path    unwrapped angle: sum over steps of the angle between Phi_{s-1} k and Phi_s k (path length of the key's direction on
          the sphere; keeps growing past pi, so it tells whether long distances reach phases unseen at d <= 31)
  sumth   sum over steps of the per-step largest rotation rate theta_1(dt K) (upper bound of the rotation; 0 for PaTH)
Statistics: median, p05, p95, share of polar > pi / 2.
usage: python -m pat342.rot_cum OUT CKPT [CKPT ...]   (env P336_NSYS as props; P336_LIE_RANK=4)"""
import sys, os, csv, math, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat336.props import systems, load, qk, T0
from pat342.intv_attn import lie_inputs
from panda.patchtst.lie_pos import lie_factors
dev = torch.device('cuda'); SMIN = 0.1; DS = (1, 4, 16, 31, 63, 127); CTXS = (512, 2048)


def ang(a, b):
    return torch.arccos((torch.nn.functional.normalize(a, dim=-1) * torch.nn.functional.normalize(b, dim=-1)).sum(-1).clamp(-1, 1))


@torch.no_grad()
def analyse(ck, data, rows):
    p = load(ck); m = p.model; layers = [l.temporal_self_attn for l in m.model.encoder.layers]
    print(f'== {ck}  temporal_pos = {[a.pos for a in layers]}', flush=True)
    for ctx in CTXS:
        m.config.context_length = ctx; acc = {}

        def hook(mod, args, kw, outp, i):
            a = layers[i]
            if not a.pos.startswith(('lie', 'path')): return
            x = args[0] if args else kw['hidden_states']; A = acc.setdefault(i, {})
            ap = lambda kk, v: A.setdefault(kk, []).append(v.flatten().double().cpu())
            _, k, _, _ = qk(a, x); bsz, N, _ = x.shape; H, E = a.num_heads, a.head_dim; n = bsz * H
            sel = torch.arange(0, n, max(1, n // 32), device=x.device)[:32]; ks = k[sel].double(); I = torch.eye(E, dtype=torch.float64, device=x.device)
            if a.pos.startswith('lie'):
                basis, gen = lie_inputs(a, x); Q, G = lie_factors(basis.float(), gen.float(), a.lie_rank, a.lie_mode); r = Q.shape[2]
                Qs, Gs = Q[sel].double(), G[sel].double(); At = I + Qs.transpose(-1, -2) @ Gs @ Qs          # A_t, t = 0..N-1
                # per-step largest rotation rate theta_1 of dt K (rebuild K, dt as in lie_factors)
                nk = r * (r - 1) // 2; g = gen[sel].double(); om, dt = g[..., :nk], 2 * torch.sigmoid(g[..., -1])
                iu = torch.triu_indices(r, r, 1, device=x.device); K = torch.zeros(*om.shape[:-1], r, r, dtype=torch.float64, device=x.device)
                K[..., iu[0], iu[1]] = om; K = K - K.transpose(-1, -2)
                th = torch.linalg.eigvals(dt[..., None, None] * K).imag.abs().amax(-1)                # [32, N]
            else:                                                                                     # PaTH H_t = I - beta_t w_t w_t^T (as in the forward)
                w = torch.nn.functional.normalize(a._shape(a.w_proj(x), -1, bsz).view(n, -1, E), dim=-1)[sel].double()
                beta = (2 * torch.sigmoid(a.beta_proj(x))).transpose(1, 2).reshape(n, -1)[sel].double()
                At = I - beta[..., None, None] * w[..., :, None] * w[..., None, :]; th = torch.zeros(len(sel), N, dtype=torch.float64, device=x.device)
            for j0 in sorted({0, N // 4}):
                Phi = torch.eye(E, dtype=torch.float64, device=x.device).expand(len(sel), E, E).clone()
                k0 = ks[:, j0]; v = k0.clone(); path = torch.zeros(len(sel), dtype=torch.float64, device=x.device); sth = path.clone()
                for d in range(1, N - j0):
                    Phi = At[:, j0 + d] @ Phi; vn = Phi @ k0[..., None]; vn = vn[..., 0]
                    path = path + ang(v, vn); v = vn; sth = sth + th[:, j0 + d]
                    if d in DS:
                        U, Sv, Vh = torch.linalg.svd(Phi); keep = Sv >= SMIN                         # surviving singular pairs
                        a_i = torch.arccos((U * Vh.transpose(-1, -2)).sum(-2).clamp(-1, 1))            # angle(u_i, v_i) per pair
                        pol = torch.where(keep, a_i, torch.full_like(a_i, -1.)).amax(-1); pol[~keep.any(-1)] = float('nan')
                        ap(f'polar_d{d}', pol); ap(f'nkeep_d{d}', keep.double().sum(-1)); ap(f'net_d{d}', ang(k0, v))
                        ap(f'path_d{d}', path); ap(f'sumth_d{d}', sth)
        hs = [a.register_forward_hook(lambda mod, args, kw, outp, i=i: hook(mod, args, kw, outp, i), with_kwargs=True) for i, a in enumerate(layers)]
        for name, x in data:
            xx = x[:, T0 - ctx:T0]; m(past_values=torch.tensor(xx, device=dev).T[None])
        for h in hs: h.remove()
        for i, A in sorted(acc.items()):
            row = dict(ckpt=ck, ctx=ctx, layer=i)
            for kk, v in A.items():
                v = torch.cat(v); v = v[~v.isnan()]
                row[kk] = float(v.median()); row[kk + '_p05'], row[kk + '_p95'] = float(v.quantile(.05)), float(v.quantile(.95))
                if kk.startswith('polar'): row[kk + '_gt90'] = float((v > math.pi / 2).double().mean())
            rows.append(row); print({k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in row.items() if k_ != 'ckpt' and not k_.endswith('p05')}, flush=True)


if __name__ == '__main__':
    out = sys.argv[1]; data = systems(); rows = []
    for ck in sys.argv[2:]: analyse(ck, data, rows)
    keys = sorted({k for r in rows for k in r}, key=lambda s: (s not in ('ckpt', 'ctx', 'layer'), s))
    with open(out + '_rotcum.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in rows]
