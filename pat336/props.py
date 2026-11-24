"""PAT-335/336 v2: learned-transport properties of a Panda checkpoint, per layer / head / context length.

Fixes over v1: dispatch on EACH layer's own `pos` (v1 used layers[0].pos, so rope1_* checkpoints had no transport
diagnostics); RoPE is applied to q / k exactly as in the module; the transition at patch 0 is never used by the score walk
(S[i, j] uses A_{j+1} .. A_i), so it is excluded from params / growth / commutators; every layer's recomputed attention
is checked against the module's own attention probabilities (output_attentions=True) -> `match` column.

Inputs: official held-out trajectories (data/final_{base40,skew40}/test_zeroshot), first trajectory file of each system
(all channels jointly), base40 systems all + skew40 systems evenly sub-sampled to P336_NSYS in total; context window
x[:, T0 - ctx : T0] with T0 = 3072, so ctx 512 / 1024 / 2048 share their most recent part. Model loaded with load seed 99.

Per layer (and per head for the parameter / growth rows):
  rel       |S - q k^T|_F / |q k^T|_F with q, k after RoPE (0 = transport does nothing); rope_rel = |S_rope - S_nope| / |S_nope|
  attention normalised entropy, mean |i - j|, max |logit|, non-finite rate of S
  path      beta (and path2 anticausal beta_b) quantiles, share < .05 / > 1.95
  lie       dt, lambda, rotation |dt K|_F / sqrt 2 quantiles; orthogonality error ||Q Q^T - I||_F of the Gram-Schmidt rows
  growth    log sigma_max and log |det| of the fp64 product A_{N-1} ... A_1 (sigma via svdvals of the explicit product)
  comm      adjacent commutators C_t = ||[X_t, X_{t-1}]||_F (path X = H_t = I - beta w w^T; lie X = M_t = Q^T m Q),
            scale-normalised C_t / (||X_t - c I|| ||X_{t-1} - c I||) (c = 1 path, 0 lie; zero norms -> excluded), and the
            same with the patch order randomly permuted within each sequence (time-permuted control)
Writes OUT_props.csv (layer / head rows), OUT_comm.csv (per system x layer commutator means) and prints a summary.
Flags (fixed before looking): non-finite scores; max |logit| > 1e3 (added after the lie_s smoke showed finite blow-up); match error > 1e-3; transport rel < 0.02 in every transport layer;
  > 90 % of a gate at one bound; orthogonality error > 1e-3; entropy collapsed (< .05) / uniform (> .995) in every layer.
usage: python -m pat336.props OUT CKPT [CKPT ...]   (env P336_CTXS=512,1024,2048  P336_NSYS=96)"""
import sys, os, math, csv, torch
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from panda.patchtst.path_pos import bidir_path_scores
from panda.patchtst.lie_pos import lie_factors, lowrank_scores, LMAX
from panda.patchtst import patchtst as PT
dev = torch.device('cuda')
CTXS = [int(c) for c in os.environ.get('P336_CTXS', '512,1024,2048').split(',')]
NSYS = int(os.environ.get('P336_NSYS', 96)); T0 = 3072; CAP = 20000


def systems():
    from gluonts.dataset.common import FileDataset
    base = sorted((ROOT / 'data/final_base40/test_zeroshot').iterdir()); skew = sorted((ROOT / 'data/final_skew40/test_zeroshot').iterdir())
    ns = max(NSYS - len(base), 0); skew = [skew[i] for i in np.linspace(0, len(skew) - 1, ns).round().astype(int)] if ns else []
    out = []
    for d in base + skew:
        f = sorted(d.glob('*.arrow'), key=lambda p: int(p.name.split('_')[0]))[0]
        x = np.asarray(list(FileDataset(path=f, freq='h', one_dim_target=False))[0]['target'], dtype=np.float32)
        out.append((d.name, x))
    return out


def load(ck):
    import transformers
    from panda.patchtst.pipeline import PatchTSTPipeline
    from pat318.exact_load import exact_load
    transformers.set_seed(99)
    p = PatchTSTPipeline.from_pretrained(mode='predict', pretrain_path=str(ck), device_map='cuda', torch_dtype=torch.float32)
    exact_load(p.model, str(ck)); p.model.eval(); return p


def sub(x):
    x = x.flatten().float(); return x[torch.randperm(x.numel(), device=x.device)[:CAP]].cpu() if x.numel() > CAP else x.cpu()


def qk(a, x):
    bsz, N, _ = x.shape; H, E = a.num_heads, a.head_dim
    q = a._shape(a.q_proj(x) * a.scaling, -1, bsz).view(bsz * H, N, E); k = a._shape(a.k_proj(x), -1, bsz).view(bsz * H, N, E)
    q0, k0 = q, k
    if a.use_rope:
        k, q = PT.apply_p_rope_to_qk(k, q, a.get_seq_pos(N, k.device, k.dtype), E, a.max_wavelength, a.rope_percent)
    return q, k, q0, k0


def transport(a, x, pc):
    """returns dict with scores S and the per-step transition factors (for growth / commutators) of this layer."""
    bsz, N, _ = x.shape; H, E = a.num_heads, a.head_dim; ps = (bsz * H, N, E)
    q, k, q0, k0 = qk(a, x); out = {'S0': q @ k.transpose(1, 2), 'q': q, 'k': k}
    if a.use_rope: out['Snr'] = q0 @ k0.transpose(1, 2)
    pos = a.pos
    if 'path' in pos:
        hp = x if pc is None else pc
        w = torch.nn.functional.normalize(a._shape(a.w_proj(hp), -1, bsz).view(*ps), dim=-1)
        be = 2 * torch.sigmoid(a.beta_proj(hp)).transpose(1, 2).reshape(bsz * H, N)
        wb = bb = None
        if pos == 'path2':
            wb = torch.nn.functional.normalize(a._shape(a.path_wb_proj(hp), -1, bsz).view(*ps), dim=-1)
            bb = 2 * torch.sigmoid(a.path_sb_proj(hp)).transpose(1, 2).reshape(bsz * H, N); out['beta_b'] = bb
        out['S'] = bidir_path_scores(q, k, w, be, wb, bb); out['kind'] = 'path'; out['w'] = w; out['beta'] = be
    elif pos.startswith('lie'):
        r = a.lie_rank
        basis = a.lie_basis_proj(x).view(bsz, N, H, r, E).transpose(1, 2).reshape(bsz * H, N, r, E)
        gen = a.lie_gen_proj(x).view(bsz, N, H, -1).transpose(1, 2).reshape(bsz * H, N, -1)
        Q, G = lie_factors(basis, gen, r, a.lie_mode); out['S'] = lowrank_scores(q, k, Q, G); out['kind'] = 'lie'
        nk = r * (r - 1) // 2; tl = torch.tanh(gen[..., nk:nk + r]); lam = torch.where(tl > 0, LMAX * tl, tl)
        dt = 2 * torch.sigmoid(gen[..., -1]); om = gen[..., :nk]
        if a.lie_mode == 's': om = torch.zeros_like(om)
        if a.lie_mode == 'k': lam = torch.zeros_like(lam)
        iu = torch.triu_indices(r, r, 1, device=x.device); Km = torch.zeros(*gen.shape[:-1], r, r, device=x.device)
        Km[..., iu[0], iu[1]] = om; Km = Km - Km.transpose(-1, -2)
        out.update(Q=Q, G=G, dt=dt, lam=lam, rot=dt * om.norm(dim=-1), m=dt[..., None, None] * (Km + torch.diag_embed(lam)))
        out['orth'] = (Q @ Q.transpose(-1, -2) - torch.eye(r, device=x.device)).norm(dim=(-1, -2))
    else:
        out['S'] = out['S0']; out['kind'] = 'none'
    return out


def factors64(o, idx):
    """explicit fp64 E x E transition matrices X_t (t = 1..N-1) for sequence-heads idx; path: H_t, lie: A_t and M_t."""
    if o['kind'] == 'path':
        w = o['w'][idx, 1:].double(); b = o['beta'][idx, 1:].double(); E = w.shape[-1]
        A = torch.eye(E, dtype=torch.float64, device=w.device) - b[..., None, None] * w[..., :, None] * w[..., None, :]
        return A, A, 1.0
    Q = o['Q'][idx, 1:].double(); E = Q.shape[-1]
    A = torch.eye(E, dtype=torch.float64, device=Q.device) + Q.transpose(-1, -2) @ o['G'][idx, 1:].double() @ Q
    M = Q.transpose(-1, -2) @ o['m'][idx, 1:].double() @ Q
    return A, M, 0.0


def comm_stats(X, c, perm):
    """X [n, T, E, E]; adjacent pairs along T (or permuted order). returns raw [n, T-1], normalised [n, T-1] (nan where norm 0)."""
    if perm is not None: X = X[:, perm]
    C = (X[:, 1:] @ X[:, :-1] - X[:, :-1] @ X[:, 1:]).norm(dim=(-1, -2))
    I = torch.eye(X.shape[-1], dtype=X.dtype, device=X.device) * c
    nx = (X - I).norm(dim=(-1, -2)); den = nx[:, 1:] * nx[:, :-1]
    return C, torch.where(den > 1e-12, C / den.clamp_min(1e-30), torch.full_like(C, float('nan')))


@torch.no_grad()
def analyse(ck, data, wr, wc):
    p = load(ck); m = p.model; layers = [l.temporal_self_attn for l in m.model.encoder.layers]
    poss = [a.pos for a in layers]; print(f'== {ck}  temporal_pos per layer = {poss}', flush=True); flags = []
    for ctx in CTXS:
        m.config.context_length = ctx; acc = {i: {} for i in range(len(layers))}; cur = {}
        def hook(mod, args, kw, outp, i):
            if cur.get('busy'): return
            x = args[0] if args else kw['hidden_states']; pc = kw.get('path_ctx'); a = layers[i]
            o = transport(a, x, pc); S, S0 = o['S'], o['S0']; N = S.shape[-1]; A = acc[i]; ap = lambda kk, v: A.setdefault(kk, []).append(v)
            P = torch.softmax(S, -1); cur['busy'] = True; Pm = mod(x, path_ctx=pc, output_attentions=True)[1].reshape(P.shape); cur['busy'] = False
            ap('match', (P - Pm).abs().amax()[None].cpu()); ii = torch.arange(N, device=x.device)
            ap('rel', ((S - S0).norm(dim=(1, 2)) / S0.norm(dim=(1, 2)).clamp_min(1e-12)).cpu())
            if 'Snr' in o: ap('rope_rel', ((S0 - o['Snr']).norm(dim=(1, 2)) / o['Snr'].norm(dim=(1, 2)).clamp_min(1e-12)).cpu())
            ap('ent', (-(P * P.clamp_min(1e-30).log()).sum(-1).mean(-1) / math.log(N)).cpu())
            ap('dist', (P * (ii[:, None] - ii[None]).abs()).sum(-1).mean(-1).cpu())
            ap('maxlog', S.abs().amax((1, 2)).cpu()); ap('nonfin', (~torch.isfinite(S)).float().mean((1, 2)).cpu())
            H = a.num_heads; hd = lambda v: v.view(-1, H, *v.shape[1:]).transpose(0, 1).reshape(H, -1)   # [H, *] per head
            for kk in ('beta', 'beta_b', 'dt', 'rot', 'orth'):
                if kk in o: ap(kk, torch.stack([sub(h) for h in hd(o[kk][:, 1:].contiguous())]) if kk != 'orth' else sub(o[kk][:, 1:]))
            if 'lam' in o: ap('lam', torch.stack([sub(h) for h in hd(o['lam'][:, 1:].contiguous())]))
            if o['kind'] != 'none':
                n = S.shape[0]; idx = torch.arange(0, n, max(1, n // 64), device=x.device)[:64]      # <= 64 sequence-heads per batch
                Af, X, c = factors64(o, idx)
                Pr = torch.eye(Af.shape[-1], dtype=torch.float64, device=x.device).expand(len(idx), -1, -1).clone()
                for t in range(Af.shape[1]): Pr = Af[:, t] @ Pr
                ap('growth', torch.linalg.svdvals(Pr)[:, 0].log().cpu()); ap('heads_g', (idx % H).cpu())
                ap('logdet', torch.linalg.slogdet(Pr)[1].cpu())
                perm = torch.randperm(X.shape[1], device=x.device)
                Ca, Na = comm_stats(X, c, None); Cp, Np = comm_stats(X, c, perm)
                for kk, v in (('comm', Ca), ('comm_n', Na), ('comm_p', Cp), ('comm_pn', Np)): ap(kk, v.flatten().float().cpu())
                sysid = cur['sys'][(idx // H) // cur['C']]
                for s_, a_, b_ in zip(sysid.tolist(), Na.float().nanmean(-1).tolist(), Np.float().nanmean(-1).tolist()):
                    A.setdefault('sys', {}).setdefault(s_, []).append((a_, b_))
        hs = [a.register_forward_hook(lambda mod, args, kw, outp, i=i: hook(mod, args, kw, outp, i), with_kwargs=True) for i, a in enumerate(layers)]
        by = {}
        for si, (name, x) in enumerate(data): by.setdefault(x.shape[0], []).append((si, x[:, T0 - ctx:T0]))
        for C, lst in by.items():
            for b0 in range(0, len(lst), max(1, 64 // C)):
                chunk = lst[b0:b0 + max(1, 64 // C)]; cur['C'] = C
                cur['sys'] = torch.tensor([si for si, _ in chunk], device=dev).repeat_interleave(1)
                ctxt = torch.tensor(np.stack([xx for _, xx in chunk]), device=dev).transpose(1, 2)
                m(past_values=ctxt)
        for h in hs: h.remove()
        rels, ents = [], []
        for i, A in acc.items():
            pos = poss[i]; cat = lambda kk: torch.cat([t.flatten() for t in A[kk]]) if kk in A else None
            qs = lambda v: [round(float(z), 4) for z in torch.quantile(v.float()[~v.isnan()][:2000000], torch.tensor([.1, .5, .9]))] if v is not None and (~v.isnan()).any() else None
            row = dict(ckpt=ck, ctx=ctx, layer=i, pos=pos, head='all')
            for kk in ('rel', 'rope_rel', 'ent', 'dist', 'maxlog', 'nonfin', 'growth', 'logdet', 'comm', 'comm_n', 'comm_p', 'comm_pn', 'orth'):
                v = cat(kk)
                if v is not None:
                    q3 = qs(v); row[kk + '_q10'], row[kk + '_q50'], row[kk + '_q90'] = q3 if q3 else (None,) * 3
                    row[kk + '_mean'] = float(v[~v.isnan()].mean()) if (~v.isnan()).any() else None; row[kk + '_max'] = float(v[~v.isnan()].max()) if (~v.isnan()).any() else None
            row['match_max'] = float(cat('match').max())
            for kk in ('beta', 'beta_b', 'dt', 'lam', 'rot'):
                if kk in A:
                    v = torch.cat(A[kk], 1); q3 = qs(v.flatten()); row[kk + '_q10'], row[kk + '_q50'], row[kk + '_q90'] = q3
                    lo, hi = {'beta': (.05, 1.95), 'beta_b': (.05, 1.95), 'dt': (.05, 1.95), 'lam': (-.95, .95 * LMAX), 'rot': (None, None)}[kk]
                    if lo is not None:
                        row[kk + '_lo'] = float((v < lo).float().mean()); row[kk + '_hi'] = float((v > hi).float().mean())
                        if max(row[kk + '_lo'], row[kk + '_hi']) > .9: flags.append(f'ctx{ctx} L{i} {kk} saturated')
                    for h in range(v.shape[0]):
                        hq = qs(v[h]); wr.writerow(dict(ckpt=ck, ctx=ctx, layer=i, pos=pos, head=h, **{kk + '_q10': hq[0], kk + '_q50': hq[1], kk + '_q90': hq[2]}))
            if 'growth' in A:
                g, hg = cat('growth'), cat('heads_g')
                for h in sorted(set(hg.tolist())):
                    hq = qs(g[hg == h]); wr.writerow(dict(ckpt=ck, ctx=ctx, layer=i, pos=pos, head=h, growth_q10=hq[0], growth_q50=hq[1], growth_q90=hq[2]))
            wr.writerow(row)
            for s_, lst in A.get('sys', {}).items():
                arr = np.array(lst); wc.writerow(dict(ckpt=ck, ctx=ctx, layer=i, pos=pos, system=data[s_][0], comm_n=np.nanmean(arr[:, 0]), comm_pn=np.nanmean(arr[:, 1])))
            if row['nonfin_max'] > 0: flags.append(f'ctx{ctx} L{i} non-finite')
            if row['maxlog_max'] > 1e3: flags.append(f'ctx{ctx} L{i} logit blow-up (max |logit| {row["maxlog_max"]:.1e})')
            if row['match_max'] > 1e-3: flags.append(f'ctx{ctx} L{i} recomputed attention != module ({row["match_max"]:.1e})')
            if row.get('orth_max') is not None and row['orth_max'] > 1e-3: flags.append(f'ctx{ctx} L{i} orthogonality error {row["orth_max"]:.1e}')
            ents.append(row['ent_mean'])
            if pos not in ('rope', 'nope'): rels.append(row['rel_q50'])
            f = lambda kk: row.get(kk + '_q50'); g = lambda kk: f'{row[kk + "_q50"]:.3g}' if row.get(kk + '_q50') is not None else '-'
            line = f'  ctx{ctx} L{i} {pos:6s} match {row["match_max"]:.1e} rel {g("rel")} ent {row["ent_mean"]:.3f} dist {row["dist_mean"]:.2f} max|logit| {row["maxlog_max"]:.1f}'
            if 'growth_q50' in row: line += f' logσmax q50/max {g("growth")}/{row["growth_max"]:.2f} comm_n adj {row["comm_n_mean"]:.3g} perm {row["comm_pn_mean"]:.3g}'
            for kk in ('beta', 'beta_b', 'dt', 'lam', 'rot'):
                if kk + '_q50' in row: line += f' {kk} [{row[kk + "_q10"]},{row[kk + "_q50"]},{row[kk + "_q90"]}]'
            if 'orth_max' in row: line += f' orth_max {row["orth_max"]:.1e}'
            print(line, flush=True)
        if rels and max(rels) < .02: flags.append(f'ctx{ctx} transport unused in every transport layer')
        if max(ents) < .05 or min(ents) > .995: flags.append(f'ctx{ctx} attention collapsed / uniform in every layer')
    print('  FLAGS: ' + ('; '.join(flags) if flags else 'none'), flush=True)


if __name__ == '__main__':
    out = sys.argv[1]; data = systems(); print(f'{len(data)} systems, T0 {T0}, ctx {CTXS}', flush=True)
    keys = ['ckpt', 'ctx', 'layer', 'pos', 'head'] + [f'{k}_{s}' for k in ('rel', 'rope_rel', 'ent', 'dist', 'maxlog', 'nonfin', 'growth', 'logdet', 'comm', 'comm_n', 'comm_p', 'comm_pn', 'orth')
                                                     for s in ('q10', 'q50', 'q90', 'mean', 'max')] + ['match_max'] + [f'{k}_{s}' for k in ('beta', 'beta_b', 'dt', 'lam', 'rot') for s in ('q10', 'q50', 'q90', 'lo', 'hi')]
    with open(out + '_props.csv', 'w', newline='') as f1, open(out + '_comm.csv', 'w', newline='') as f2:
        wr = csv.DictWriter(f1, keys); wr.writeheader(); wc = csv.DictWriter(f2, ['ckpt', 'ctx', 'layer', 'pos', 'system', 'comm_n', 'comm_pn']); wc.writeheader()
        for ck in sys.argv[2:]: analyse(ck, data, wr, wc); f1.flush(); f2.flush()
