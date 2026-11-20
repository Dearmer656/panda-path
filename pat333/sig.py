"""PAT-333 signal-space retrieval batch (top-1 copy unless noted; errors in ctx-window-normalised units; per-trajectory median;
paired bootstrap seed 0). Criteria fixed before results:
 E1 thesis test (ctx 512, whole history, hidden states = PCA-64 last-layer Panda patch states stored in test.pt):
    same candidate set (patch ends t - 16k), same copy readout; only the key differs.
    sig16 (Euclid on the signal window) vs hid_cos / hid_eu (hidden state); *_a = only window-aligned patches (same
    within-window position as the query, full context). Thesis supported if sig beats every hidden arm at all H (CI < 0).
 E2 joint-state keys: Panda is channel-independent; the dynamical state is the full vector. joint = one analog time per
    instance, distance summed over channels (D = D_FULL[H]); joint96 = same with D = 96. GO if joint beats eu_parrot
    (per-channel) on sMAPE at all H (CI < 0).
 E3 match-confidence gate between backbone f and joint retrieval r, dn = joint min distance / n_channels:
    gate_hard p = r if dn < tau else f; gate_soft p = a r + (1 - a) f, a = exp(-dn / kappa); fuse_const p = al f + (1 - al) r.
    tau / kappa / al chosen on val median MSE. GO if gate (better of hard / soft on val) beats f AND joint on sMAPE and
    MSE (CI excludes 0) and beats fuse_const on MSE.
P333_HIST=full: whole own history (s + H <= t); ctx: window and future inside [t - CTX, t), backbone = truncated PaTH (last 512).
usage: P321_LQ=96 P321_OUT=pat321/data P333_HIST=full python pat333/sig.py
       P321_LQ=96 P321_CTX=2048 P321_OUT=pat321/data_ctx2048_path P333_HIST=ctx python pat333/sig.py"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in ('pat327', 'pat329', 'pat330'): sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(ROOT))
import numpy as np, torch
import fc_corr as C  # noqa: E402
from mse_eval import per_traj, D_FULL  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); CTX = FC.CTX; MODE = os.environ.get('P333_HIST', 'full'); CLIP = 1000.0
O = ROOT / 'pat333' / 'out'; TAG = f'ctx{CTX}_{MODE}'


def backbone(name, d):
    if MODE == 'full': return d['f']
    if name == 'test': return torch.load(ROOT / 'pat330' / 'out' / f'trunc_ctx{CTX}.pt')['preds']['PaTH_trunc512']
    fp = O / f'valbb_{TAG}.pt'
    if fp.exists(): return torch.load(fp)
    from trunc import forecast, CK
    p = FC.load_backbone(ROOT / CK['PaTH'], dev); p.model.config.context_length = 512
    f = forecast(p, d); torch.save(f, fp); del p; torch.cuda.empty_cache(); return f


def chunks(d, maxrows=96):
    ids = d['ids']; out, cur, n = [], [], 0
    for i in range(d['n_inst']):
        q = torch.nonzero(ids == i)[:, 0]
        if cur and n + len(q) > maxrows: out.append(torch.cat(cur)); cur, n = [], 0
        cur.append(q); n += len(q)
    if cur: out.append(torch.cat(cur))
    return out


def copy(Z, s, H):
    ar = torch.arange(len(Z), device=dev)
    return Z[ar[:, None], s.clamp(0, Z.shape[1] - H)[:, None] + torch.arange(H, device=dev)].clamp(-CLIP, CLIP)


def okmask(s, t, D, H):
    ok = (s[None] + H <= t[:, None]) & (s[None] >= D)
    if MODE == 'ctx': ok &= s[None] - D >= (t - CTX)[:, None]
    return ok


@torch.no_grad()
def run_set(name, H, hidden):
    d = torch.load(FC.OUT / f'{name}.pt'); S = d['S'].to(dev); fall = backbone(name, d)
    P = {}; Y, DN = [], []
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - CTX + torch.arange(CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10)
        Z = (X - cm[:, None]) / cs[:, None]
        Y.append(Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)].cpu())
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        nch = torch.zeros(ni, device=dev).index_add_(0, loc, torch.ones(B, device=dev))
        A = {'f': (f, torch.ones(B, dtype=torch.bool, device=dev))}
        for tag, D in (('', D_FULL[H]), ('96', 96)):
            s = torch.arange(D, T + 1, device=dev)
            q = Z[ar[:, None], t[:, None] + torch.arange(-D, 0, device=dev)]
            dist = (Z.unfold(1, D, 1) - q[:, None]).square().mean(-1)                      # window ending at s
            ok = okmask(s, t, D, H)
            if tag == '':
                dm = dist.masked_fill(~ok, float('inf')); A['eu_parrot'] = (copy(Z, s[dm.argmin(1)], H), torch.isfinite(dm.min(1).values))
            J = torch.zeros(ni, len(s), device=dev).index_add_(0, loc, dist.masked_fill(~ok, 0.0))
            J = J.masked_fill(~ok[torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)], float('inf'))
            jm, ja = J.min(1); okj = torch.isfinite(jm)[loc]
            A[f'joint{tag}'] = (copy(Z, s[ja][loc], H), okj)
            if tag == '': DN.append(torch.where(okj, (jm / nch)[loc], torch.full_like(okj, float('inf'), dtype=torch.float)).cpu())
        if hidden:
            h, hv = d['h'][idx].to(dev).float(), d['hv'][idx].to(dev)
            k = torch.arange(FC.KMAX, device=dev); e = t[:, None] - 16 * k[None]; D = D_FULL[H]
            okh = hv & (16 * k[None] >= H) & (e >= D)
            q = Z[ar[:, None], t[:, None] + torch.arange(-D, 0, device=dev)]
            Wc = Z[ar[:, None, None], e.clamp(min=D)[..., None] + torch.arange(-D, 0, device=dev)]
            sd = (Wc - q[:, None]).square().mean(-1)
            hq = h[:, 0]; hc = torch.nn.functional.cosine_similarity(h, hq[:, None], dim=-1); he = (h - hq[:, None]).square().sum(-1)
            for al, okk in (('', okh), ('_a', okh & (k[None] % FC.NPW == 0))):
                for nm, sc in (('sig16', sd), ('hid_cos', -hc), ('hid_eu', he)):
                    dm = sc.masked_fill(~okk, float('inf')); j = dm.argmin(1)
                    A[nm + al] = (copy(Z, e.gather(1, j[:, None])[:, 0], H), torch.isfinite(dm.min(1).values))
        for m, (p, okm) in A.items(): P.setdefault(m, []).append(torch.where(okm[:, None], p, f).cpu())
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    P = {m: torch.cat(v)[inv] for m, v in P.items()}
    return d, P, torch.cat(Y)[inv], torch.cat(DN)[inv]


def errs(p, y, d):
    return {'smape': per_traj(((p - y).abs() / (p.abs() + y.abs() + 1e-10)).mean(1), d['ids'], d['inst']),
            'mse': per_traj((p - y).square().mean(1), d['ids'], d['inst'])}


def mix(P, DN, kind, v):
    f, r = P['f'], P['joint']
    if kind == 'hard': a = (DN < v).float()
    elif kind == 'soft': a = torch.exp(-DN / v)
    else: a = torch.full_like(DN, 1 - v)
    a = torch.nan_to_num(a, nan=0.0)[:, None]
    return a * r + (1 - a) * f


def cmp(T, x, ref, rng):
    o = []
    for met, sc in (('smape', 100), ('mse', 1)):
        d0, lo, hi, _ = C.boot(sc * T[x][met].numpy(), sc * T[ref][met].numpy(), rng); o.append(f'{met} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
    return ' | '.join(o)


def run(H):
    hidden = MODE == 'full' and CTX == 512
    dv, Pv, Yv, DNv = run_set('val', H, False); dt, Pt, Yt, DNt = run_set('test', H, hidden)
    q = torch.quantile(DNv[torch.isfinite(DNv)], torch.linspace(0.05, 1.0, 20)).tolist()
    grids = {'hard': q + [float('inf')], 'soft': list(np.logspace(-3, 1, 25)), 'const': list(np.round(np.arange(0, 1.01, 0.1), 1))}
    pick = {}
    for kind, g in grids.items():
        sc = [float(errs(mix(Pv, DNv, kind, v), Yv, dv)['mse'].median()) for v in g]; j = int(np.argmin(sc)); pick[kind] = (g[j], sc[j])
    for kind, (v, _) in pick.items(): Pt[f'gate_{kind}' if kind != 'const' else 'fuse_const'] = mix(Pt, DNt, kind, v)
    gbest = 'gate_hard' if pick['hard'][1] <= pick['soft'][1] else 'gate_soft'
    T = {m: errs(p, Yt, dt) for m, p in Pt.items()}
    vf = float(errs(Pv['f'], Yv, dv)['mse'].median()); vj = float(errs(Pv['joint'], Yv, dv)['mse'].median())
    print(f'== {TAG} H{H}: val MSE f {vf:.4f} joint {vj:.4f} | picks ' + ', '.join(f'{k} {v:.4g} ({s:.4f})' for k, (v, s) in pick.items()) + f' -> {gbest}', flush=True)
    for m in T: print(f'   test {m:12s} sMAPE {100 * float(T[m]["smape"].median()):6.2f}  MSE {float(T[m]["mse"].median()):.4f}', flush=True)
    rng = np.random.default_rng(0)
    pairs = [('joint', 'eu_parrot'), ('joint96', 'eu_parrot'), ('joint', 'f'), (gbest, 'f'), (gbest, 'joint'), (gbest, 'eu_parrot'), (gbest, 'fuse_const')]
    if hidden: pairs += [(f'sig16{a}', f'{h}{a}') for a in ('', '_a') for h in ('hid_cos', 'hid_eu')] + [('sig16', 'eu_parrot')]
    for x, ref in pairs: print(f'   {x:12s} - {ref:12s} ' + cmp(T, x, ref, rng), flush=True)
    torch.save({'T': T, 'pick': pick, 'DNt': DNt}, O / f'sig_{TAG}_H{H}.pt')


if __name__ == '__main__':
    O.mkdir(exist_ok=True)
    for H in (128, 256, 512): run(H)
