"""PAT-336: learned-transport properties of a Panda checkpoint (temporal_pos path / path_shared / lie), per layer.
Inputs: 512-step contexts of 256 test instances (pat321/data/test.pt, all channels jointly, seed 99 load). Forward hooks
capture each temporal attention's input; quantities are recomputed from the module's own weights.
Reported per layer (pooled over heads, patches, sequences):
  params   path : beta quantiles, share beta < 0.05 (~identity) / > 1.95 (~full reflection)
           lie  : dt quantiles, share dt < 0.05 / > 1.95; lambda quantiles, share at the bounds (< -0.95 / > 0.95 LMAX);
                  rotation size |dt K|_F / sqrt 2 quantiles (radians per patch)
  transport rel = |S - q k^T|_F / |q k^T|_F (0 = acts like NoPE)
  growth   log ||A_31 ... A_1 v|| for random unit v (path is <= 0 by construction; lie may grow by <= 31 * 2 * LMAX)
  attention normalised entropy H / log N, mean |i - j| under the softmax, max |logit|, all finite
Flags ("abnormal", fixed before looking): any non-finite; rel < 0.02 in every layer (transport unused);
  > 90 % of a gate at one bound in a layer (saturated); mean entropy < 0.05 or > 0.995 in every layer (collapsed / uniform).
usage: python -m pat336.props CKPT [CKPT ...]"""
import sys, math, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'pat321')); sys.path.insert(0, str(ROOT))
import fc_dysts as FC
from panda.patchtst.path_pos import bidir_path_scores
from panda.patchtst.lie_pos import lie_factors, lowrank_scores, LMAX
dev = torch.device('cuda'); torch.manual_seed(0)


def contexts(n=256):
    d = torch.load(ROOT / 'pat321' / 'data' / 'test.pt', weights_only=False)
    S, rows, t, ids = d['S'], d['rows'], d['t'], d['ids']; out = []
    for i in range(min(n, d['n_inst'])):
        qi = torch.nonzero(ids == i)[:, 0]; tt = int(t[qi[0]])
        if tt >= 512: out.append(S[rows[qi]][:, tt - 512:tt])
    by = {}
    for c in out: by.setdefault(c.shape[0], []).append(c)
    return [torch.stack(v).transpose(1, 2) for v in by.values()]          # list of [b, 512, C]


def q_(x, p): return [round(float(v), 3) for v in torch.quantile(x.flatten().float()[torch.randperm(x.numel(), device=x.device)[:200000]], torch.tensor(p, device=x.device))]


@torch.no_grad()
def analyse(ck):
    p = FC.load_backbone(ROOT / ck, dev); m = p.model
    layers = [l.temporal_self_attn for l in m.model.encoder.layers]; pos = layers[0].pos
    cap = {i: [] for i in range(len(layers))}
    hs = [a.register_forward_hook(lambda mod, args, kw, out, i=i: cap[i].append((args[0] if args else kw['hidden_states'], kw.get('path_ctx'))), with_kwargs=True) for i, a in enumerate(layers)]
    for c in contexts(): p.predict(context=c.to(dev), prediction_length=128, verbose=False)
    for h in hs: h.remove()
    print(f'== {ck}  (temporal_pos = {pos})', flush=True); flags = []; rels, ents = [], []
    for i, lst in cap.items():
        A = {}
        a = layers[i]
        for x, pc in lst[:4]:
            bsz, N, _ = x.shape; H, E = a.num_heads, a.head_dim
            q = a._shape(a.q_proj(x) * a.scaling, -1, bsz).view(bsz * H, N, E); k = a._shape(a.k_proj(x), -1, bsz).view(bsz * H, N, E)
            S0 = q @ k.transpose(1, 2); v = torch.nn.functional.normalize(torch.randn(bsz * H, E, device=dev), dim=-1)
            if 'path' in pos:
                hp = x if pc is None else pc
                w = torch.nn.functional.normalize(a._shape(a.w_proj(hp), -1, bsz).view(bsz * H, N, E), dim=-1)
                be = 2 * torch.sigmoid(a.beta_proj(hp)).transpose(1, 2).reshape(bsz * H, N)
                S = bidir_path_scores(q, k, w, be)
                A.setdefault('beta', []).append(be)
                for s in range(N): v = v - be[:, s, None] * (w[:, s] * v).sum(-1, keepdim=True) * w[:, s]
            elif pos == 'lie':
                r = a.lie_rank
                basis = a.lie_basis_proj(x).view(bsz, N, H, r, E).transpose(1, 2).reshape(bsz * H, N, r, E)
                gen = a.lie_gen_proj(x).view(bsz, N, H, -1).transpose(1, 2).reshape(bsz * H, N, -1)
                Q, G = lie_factors(basis, gen, r); S = lowrank_scores(q, k, Q, G)
                nk = r * (r - 1) // 2; tl = torch.tanh(gen[..., nk:nk + r]); lam = torch.where(tl > 0, LMAX * tl, tl)
                dt = 2 * torch.sigmoid(gen[..., -1]); om = gen[..., :nk]
                A.setdefault('dt', []).append(dt); A.setdefault('lam', []).append(lam); A.setdefault('rot', []).append(dt * om.norm(dim=-1))
                for s in range(N): v = v + (Q[:, s].transpose(1, 2) @ (G[:, s] @ (Q[:, s] @ v[..., None])))[..., 0]
            else:
                S = S0
            P = torch.softmax(S, -1); ii = torch.arange(N, device=dev)
            A.setdefault('rel', []).append(((S - S0).norm(dim=(1, 2)) / S0.norm(dim=(1, 2)).clamp_min(1e-12)))
            A.setdefault('ent', []).append(-(P * P.clamp_min(1e-30).log()).sum(-1).mean(-1) / math.log(N))
            A.setdefault('dist', []).append((P * (ii[:, None] - ii[None]).abs()).sum(-1).mean(-1))
            A.setdefault('maxlog', []).append(S.abs().amax((1, 2))); A.setdefault('growth', []).append(v.norm(dim=-1).log())
            A.setdefault('fin', []).append(torch.isfinite(S).all()[None].float())
        A = {kk: torch.cat([t.flatten() for t in vv]) for kk, vv in A.items()}
        line = f'  L{i}: rel {q_(A["rel"], [.1, .5, .9])}  ent {float(A["ent"].mean()):.3f}  dist {float(A["dist"].mean()):.2f}  max|logit| {float(A["maxlog"].max()):.1f}  growth(31) {q_(A["growth"], [.1, .5, .9])}'
        if 'beta' in A:
            b = A['beta']; line += f'  beta {q_(b, [.1, .5, .9])} <.05 {float((b < .05).float().mean()):.2f} >1.95 {float((b > 1.95).float().mean()):.2f}'
            if max(float((b < .05).float().mean()), float((b > 1.95).float().mean())) > .9: flags.append(f'L{i} beta saturated')
        if 'dt' in A:
            d_, l_ = A['dt'], A['lam']; sh = {'dt<.05': (d_ < .05), 'dt>1.95': (d_ > 1.95), 'lam<-.95': (l_ < -.95), 'lam>.95max': (l_ > .95 * LMAX)}
            line += f'  dt {q_(d_, [.1, .5, .9])} lam {q_(l_, [.1, .5, .9])} rot {q_(A["rot"], [.1, .5, .9])} ' + ' '.join(f'{kk} {float(vv.float().mean()):.2f}' for kk, vv in sh.items())
            flags += [f'L{i} {kk} saturated' for kk, vv in sh.items() if float(vv.float().mean()) > .9]
        if float(A['fin'].min()) < 1: flags.append(f'L{i} non-finite')
        print(line, flush=True); rels.append(float(A['rel'].median())); ents.append(float(A['ent'].mean()))
    if pos != 'nope' and pos != 'rope' and max(rels) < .02: flags.append('transport unused in every layer')
    if max(ents) < .05 or min(ents) > .995: flags.append('attention collapsed / uniform in every layer')
    print('  FLAGS: ' + ('; '.join(flags) if flags else 'none'), flush=True)


if __name__ == '__main__':
    for ck in sys.argv[1:]: analyse(ck)
