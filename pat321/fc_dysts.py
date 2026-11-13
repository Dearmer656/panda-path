"""PAT-321: FreshCast on a frozen PatchTST (Panda without channel attention) on dysts, and FreshCast with a PaTH-transported
similarity term.

Protocol (Panda official eval): test windows are exactly those of the official run (pat321/dump_test.py records the
sampled origins); context 512, one 512-step forecast (PatchTST rolls out 4 x 128 with a sliding context), sMAPE on
context-standardised values at L = 128 / 256 / 512 (prefixes), median over the 9347 test windows.

FreshCast (arXiv 2610.07834) adapted to chaotic trajectories, per channel, H in {128, 256, 512} (one scorer per H,
as in the paper):
  memory      the trajectory's own observed history [0, t) (the frozen forecaster sees only [t - 512, t))
  query       the last LQ = 96 observed steps
  candidates  every window ending at s with LQ <= s and s + H <= t (causal admissibility); the N = 200 with the highest
              shape cosine are kept, each with a value and a shape readout (no seasonal analogs: no period)
  descriptors value flag, cosine, log age, level gap, scale ratio, + 3 history statistics (acf lag 16, acf lag 96,
              log mean squared difference) -> NZ = 8; query statistics (mean / sd / max top cosine + history stats)
  scorer      relational kernel regression as in pat320/fc.py (linear + MLP score, inverse temperature)
  loss        min(||r - y||^2 / (H eps_q), 20), eps_q = MSE of the untrained rule (mean of the top-20 shape readouts)
  combination y = f + w_H (r - f), w_H = clip(<y - f, r - f> / <r - f, r - f>, 0, 1) on held-out TRAINING systems
Arms:
  fc    FreshCast as above
  bil   fc + gamma * phi_k(cand)^T phi_q(query)            (learned shape similarity, no transport; control)
  path  fc + gamma * phi_k(cand)^T P_{t,s} phi_q(query)    P = H_{k_s - 1} ... H_0, H_k = I - beta_k w_k w_k^T,
        w_k = normalise(A h_k), beta_k = 2 sigmoid(b . h_k); h_k = frozen backbone last-layer state of patch k
        (k = patches back from t; states from backbone windows ending at t, t - 512, ...; PCA-64)
All values are standardised by the query's context (t - 512, t] statistics, as the official metric does.
usage: python pat321/fc_dysts.py --stage prep|train|test ..."""
import argparse, copy, glob, json, math, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
LQ, NSH, PATCH, CTX, T_LEN = 96, 200, 16, 512, 4096
HS = (128, 256, 512)
KMAX = (T_LEN - LQ) // PATCH + 1
MAX_TEST = int(os.environ.get('P321_MAX_TEST', 0))
SD_FLOOR, CLIP, C_MAX = 1e-2, 1000.0, 20.0
EPOCHS, BATCH, LR, WD, EMA = int(os.environ.get('P321_EPOCHS', 30)), 256, 2e-3, 1e-2, 0.995
NZ, NU, DK, DH = 8, 6, 32, 64
OUT = Path(os.environ.get('P321_OUT', ROOT / 'pat321' / 'data'))


# ------------------------------------------------------------------ data
def load_traj(path):
    from gluonts.dataset.common import FileDataset
    return [np.asarray(e['target'], dtype=np.float32) for e in FileDataset(path=Path(path), freq='h', one_dim_target=False)]


def test_instances(dump):
    trajs, inst = [], []
    with open(dump, 'rb') as f:
        while True:
            try:
                ts, idx = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                break
            if MAX_TEST and len(inst) >= MAX_TEST:
                break
            trajs.append(ts)
            inst += [(len(trajs) - 1, int(t)) for t in idx]
    return trajs, inst


def train_split(seed=0, frac_val=0.05):
    files = sorted(glob.glob(str(ROOT / 'data/final_base40/train/*/*.arrow')) + glob.glob(str(ROOT / 'data/final_skew40/train/*/*.arrow')))
    systems = sorted({Path(f).parent.parent.name + '/' + Path(f).parent.name for f in files})
    rng = np.random.default_rng(seed)
    val_sys = set(rng.choice(systems, size=max(1, int(frac_val * len(systems))), replace=False).tolist())
    tr = [f for f in files if Path(f).parent.parent.name + '/' + Path(f).parent.name not in val_sys]
    va = [f for f in files if Path(f).parent.parent.name + '/' + Path(f).parent.name in val_sys]
    return tr, va, sorted(val_sys)


def sample_instances(files, n_traj, per_traj, seed):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(files), size=min(n_traj, len(files)), replace=False)
    trajs, inst = [], []
    for i in pick:
        for ts in load_traj(files[i]):
            if ts.shape[-1] != T_LEN or not np.isfinite(ts).all():
                continue
            trajs.append(ts)
            for t in rng.integers(CTX, T_LEN - CTX + 1, size=per_traj):
                inst.append((len(trajs) - 1, int(t)))
            break
    return trajs, inst


# ------------------------------------------------------------------ backbone
def load_backbone(ckpt, dev):
    import transformers
    from panda.patchtst.pipeline import PatchTSTPipeline
    from pat318.exact_load import exact_load
    transformers.set_seed(99)
    p = PatchTSTPipeline.from_pretrained(mode='predict', pretrain_path=ckpt, device_map=str(dev), torch_dtype=torch.float32)
    exact_load(p.model, ckpt)
    p.model.eval()
    return p


@torch.no_grad()
def states_window(p, X, t, dev):
    """raw last-layer patch states [b, C, KMAX, 512] (zeros where no aligned backbone window exists)."""
    Hs = torch.zeros(X.shape[0], X.shape[1], KMAX, 512, dtype=torch.float16, device=dev)
    for m in range(T_LEN // CTX):
        end = t - m * CTX; ok = end - CTX >= 0
        if not ok.any():
            break
        bi = ok.nonzero()[:, 0]
        win = torch.stack([X[b, :, end[b] - CTX:end[b]] for b in bi.tolist()]).transpose(1, 2)
        hid = p.model.model(past_values=win).last_hidden_state                              # [b', C, 32, d]
        for kk in range(32):
            k = m * 32 + kk
            if k < KMAX:
                Hs[bi, :, k] = hid[:, :, 31 - kk].half()
    return Hs


def chunks_by_dim(trajs, inst, bs):
    order = sorted(range(len(inst)), key=lambda i: (trajs[inst[i][0]].shape[0], i))
    out, cur = [], []
    for i in order:
        if cur and (len(cur) == bs or trajs[inst[cur[-1]][0]].shape[0] != trajs[inst[i][0]].shape[0]):
            out.append(cur); cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def backbone_pass(p, trajs, inst, dev, want_f, pca, bs=128):
    """per instance (in `inst` order): f [C, 512] (official rollout, raw units), PCA-projected patch states
    [C, KMAX, DH] fp16 and their validity [C, KMAX]."""
    F_, Hp, Hv = [None] * len(inst), [None] * len(inst), [None] * len(inst)
    for ch in chunks_by_dim(trajs, inst, bs):
        C = trajs[inst[ch[0]][0]].shape[0]
        X = torch.stack([torch.from_numpy(trajs[inst[i][0]]) for i in ch]).to(dev)        # [b, C, T]
        t = torch.tensor([inst[i][1] for i in ch], device=dev)
        if want_f:
            ctx = torch.stack([X[b, :, t[b] - CTX:t[b]] for b in range(len(ch))]).transpose(1, 2)
            pr = p.predict(context=ctx, prediction_length=512, limit_prediction_length=False, sliding_context=True, verbose=False)
            fb = pr.reshape(len(ch), 512, C).transpose(1, 2).float().cpu()
        Hs = states_window(p, X, t, dev)
        hv = Hs.float().abs().sum(-1) > 0
        hp = ((Hs.float() - pca[0]) @ pca[1]).half() * hv[..., None]
        for b, i in enumerate(ch):
            Hp[i], Hv[i] = hp[b].cpu(), hv[b].cpu()
            if want_f:
                F_[i] = fb[b]
    return F_, Hp, Hv


@torch.no_grad()
def fit_pca(p, trajs, inst, dev, n=512):
    X_, t_ = [], []
    vecs = []
    for ch in chunks_by_dim(trajs, inst[:n], 128):
        X = torch.stack([torch.from_numpy(trajs[inst[i][0]]) for i in ch]).to(dev)
        t = torch.tensor([inst[i][1] for i in ch], device=dev)
        win = X[torch.arange(len(ch), device=dev)[:, None, None], torch.arange(X.shape[1], device=dev)[None, :, None],
                (t[:, None, None] - CTX + torch.arange(CTX, device=dev)[None, None])].transpose(1, 2)
        vecs.append(p.model.model(past_values=win).last_hidden_state.reshape(-1, 512).float())
    hs = torch.cat(vecs)
    mean = hs.mean(0); _, _, Vt = torch.linalg.svd(hs - mean, full_matrices=False)
    return mean, Vt[:DH].T.contiguous()


# ------------------------------------------------------------------ FreshCast pieces (per query = (series row, t))
class Series:
    """Flat per-channel series S [n_series, T] with sliding LQ-window statistics."""
    def __init__(self, S):
        self.S = S
        z = torch.zeros(len(S), 1, dtype=torch.float64, device=S.device)
        Sd = S.double()
        c1, c2 = torch.cat([z, Sd.cumsum(1)], 1), torch.cat([z, Sd.square().cumsum(1)], 1)
        mu = torch.full((len(S), T_LEN + 1), float('nan'), dtype=torch.float64, device=S.device); sd = mu.clone()
        m = (c1[:, LQ:] - c1[:, :-LQ]) / LQ
        mu[:, LQ:] = m; sd[:, LQ:] = ((c2[:, LQ:] - c2[:, :-LQ]) / LQ - m.square()).clamp_min(0).sqrt()
        self.mu, self.sd = mu.float(), sd.float()        # stats of S[:, s - LQ:s] at index s


def hist_desc(S, t):
    """acf lag 16, acf lag 96, log mean squared diff on [0, t) per query; S rows already gathered [B, T]."""
    B = len(S); ar = torch.arange(T_LEN, device=S.device)[None]
    m = (ar < t[:, None]).float()
    n = m.sum(1)
    mean = (S * m).sum(1) / n
    xc = (S - mean[:, None]) * m
    var = xc.square().sum(1) / n

    def acf(l):
        return (xc[:, l:] * xc[:, :-l]).sum(1) / ((n - l) * var).clamp_min(1e-8)
    d = (S[:, 1:] - S[:, :-1]) * m[:, 1:]
    msd = d.square().sum(1) / (n - 1)
    return torch.stack([acf(16), acf(96), torch.log(msd / var.clamp_min(1e-8) + 1e-6)], -1)


@torch.no_grad()
def build(se, rows, t, H):
    """candidates for queries (rows, t): s [B, 2N], val [B, 2N], z [B, 2N, NZ], u [B, NU], valid [B, 2N], ctx stats."""
    S, dev = se.S, se.S.device
    B = len(rows)
    X = S[rows]                                                                      # [B, T]
    mq, sq = se.mu[rows, t], se.sd[rows, t]
    kq = torch.arange(-LQ, 0, device=dev)
    qc = X.gather(1, t[:, None] + kq[None]) - mq[:, None]                            # [B, LQ]
    dot = F.conv1d(X[None], qc[:, None], groups=B)[0]                                # [B, T - LQ + 1], start index
    s_all = torch.arange(LQ, T_LEN + 1, device=dev)                                  # window end s = start + LQ
    sc = se.sd[rows][:, LQ:]
    cos = dot / (LQ * sc * sq[:, None]).clamp_min(1e-12)
    ok = (s_all[None] + H <= t[:, None]) & (sc > 1e-6) & (sq[:, None] > 1e-6)
    cos = cos.masked_fill(~ok, -math.inf)
    k = min(NSH, cos.shape[1])
    top, ti = cos.topk(k, 1)
    vs = torch.isfinite(top)
    s = torch.where(vs, s_all[ti], t[:, None] - H).clamp_min(LQ)
    top = torch.where(vs, top, torch.zeros_like(top))
    s2 = torch.cat([s, s], 1); val = torch.cat([torch.ones_like(vs), torch.zeros_like(vs)], 1)
    valid = torch.cat([vs, vs], 1); cs = torch.cat([top, top], 1)
    ctx_m = X.gather(1, t[:, None] - CTX + torch.arange(CTX, device=dev)[None]).mean(1)
    ctx_s = X.gather(1, t[:, None] - CTX + torch.arange(CTX, device=dev)[None]).std(1, unbiased=False).clamp_min(1e-10)
    sqf = (sq / ctx_s).clamp_min(SD_FLOOR)[:, None]
    mc, scc = se.mu[rows[:, None], s2], se.sd[rows[:, None], s2]
    gap = torch.log1p(((mc - mq[:, None]) / ctx_s[:, None]).abs() / sqf).clamp_max(10)
    ratio = torch.log((scc / ctx_s[:, None]).clamp_min(SD_FLOOR) / sqf).clamp(-10, 10)
    hd = hist_desc(X, t)
    z = torch.stack([val.float(), cs, torch.log1p((t[:, None] - s2).float() / LQ), gap, ratio], -1)
    z = torch.cat([z, hd[:, None].expand(-1, s2.shape[1], -1)], -1)
    n = vs.float().sum(1).clamp_min(1)
    m1 = (top * vs).sum(1) / n
    sdv = (((top - m1[:, None]) * vs).square().sum(1) / n).sqrt()
    u = torch.cat([torch.stack([m1, sdv, top.max(1).values], -1), hd], -1)
    return dict(s=s2, val=val, z=z, u=u, valid=valid, cm=ctx_m, cs=ctx_s, any=vs.any(1))


def readout(se, rows, t, b, H):
    """candidate continuations [B, 2N, H] in context-standardised units."""
    S, dev = se.S, se.S.device
    kh = torch.arange(H, device=dev)
    s = b['s']
    cont = S[rows[:, None, None], s[..., None] + kh]
    mc, sc = se.mu[rows[:, None], s][..., None], se.sd[rows[:, None], s].clamp_min(1e-10)[..., None]
    mq, sq = se.mu[rows, t][:, None, None], se.sd[rows, t][:, None, None]
    R = torch.where(b['val'][..., None], cont, mq + sq / sc * (cont - mc))
    return ((R - b['cm'][:, None, None]) / b['cs'][:, None, None]).clamp(-CLIP, CLIP)


def shapes(se, rows, t, b):
    """normalised LQ-window shapes of the query [B, LQ] and of the N distinct candidates [B, N, LQ]."""
    S, dev = se.S, se.S.device
    kq = torch.arange(-LQ, 0, device=dev)
    N = b['s'].shape[1] // 2; s = b['s'][:, :N]
    q = S[rows[:, None], t[:, None] + kq]
    q = (q - se.mu[rows, t][:, None]) / se.sd[rows, t].clamp_min(1e-6)[:, None]
    c = S[rows[:, None, None], s[..., None] + kq]
    c = (c - se.mu[rows[:, None], s][..., None]) / se.sd[rows[:, None], s].clamp_min(1e-6)[..., None]
    return q, c


class Scorer(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.lin = nn.Linear(NZ, 1)
        self.mlp = nn.Sequential(nn.Linear(NZ, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.b0 = nn.Parameter(torch.zeros(()))
        self.bnet = nn.Sequential(nn.Linear(NU, 16), nn.GELU(), nn.Linear(16, 1))
        for k, n in (('zm', NZ), ('zs', NZ), ('um', NU), ('us', NU)):
            self.register_buffer(k, torch.zeros(n) if k.endswith('m') else torch.ones(n))
        if arm in ('bil', 'path'):
            self.phq = nn.Sequential(nn.Linear(LQ, 64), nn.GELU(), nn.Linear(64, DK))
            self.phk = nn.Sequential(nn.Linear(LQ, 64), nn.GELU(), nn.Linear(64, DK))
            self.gamma = nn.Parameter(torch.tensor(0.1))
        if arm == 'path':
            self.wp = nn.Linear(DH, DK)
            self.bp = nn.Linear(DH, 1)

    def transport(self, q, h, hv):
        """V[:, k] = H_{k-1} ... H_0 q for k = 0..KMAX-1."""
        w = F.normalize(self.wp(h.float()), dim=-1)                                  # [B, K, DK]
        beta = 2 * torch.sigmoid(self.bp(h.float()))[..., 0] * hv                     # [B, K]
        V = [q]; v = q
        for k in range(h.shape[1] - 1):
            v = v - (beta[:, k, None] * (w[:, k] * v).sum(-1, keepdim=True)) * w[:, k]
            V.append(v)
        return torch.stack(V, 1)

    def forward(self, b, qs=None, cs=None, t=None, h=None, hv=None):
        zn = (b['z'] - self.zm) / self.zs
        sc = self.lin(zn)[..., 0] + self.mlp(zn)[..., 0]
        if self.arm in ('bil', 'path'):
            fq, fk = self.phq(qs), self.phk(cs)                                       # [B, DK], [B, N, DK]
            if self.arm == 'path':
                V = self.transport(fq, h, hv)
                N = cs.shape[1]
                ks = ((t[:, None] - b['s'][:, :N]) // PATCH).clamp(0, V.shape[1] - 1)
                fqk = V.gather(1, ks[..., None].expand(-1, -1, DK))
                e = (fk * fqk).sum(-1)
            else:
                e = (fk * fq[:, None]).sum(-1)
            sc = sc + self.gamma * torch.cat([e, e], 1) / math.sqrt(DK)
        beta = torch.exp(self.b0 + self.bnet((b['u'] - self.um) / self.us)[..., 0].clamp(-2, 2))
        return torch.softmax((beta[:, None] * sc).masked_fill(~b['valid'], -math.inf), -1)


# ------------------------------------------------------------------ query tables
def queries(trajs, inst):
    """flatten instances into per-channel queries: series matrix rows, origin, instance id."""
    rows, ts, ids, S = [], [], [], []
    base = {}
    for j in sorted({j for j, _ in inst}):
        base[j] = len(S); S += [trajs[j][c] for c in range(trajs[j].shape[0])]
    for i, (j, t) in enumerate(inst):
        for c in range(trajs[j].shape[0]):
            rows.append(base[j] + c); ts.append(t); ids.append(i)
    return np.stack(S), np.array(rows), np.array(ts), np.array(ids)


def run_prep(a, dev):
    p = load_backbone(a.ckpt, dev)
    tr_files, va_files, val_sys = train_split()
    sets = dict(train=sample_instances(tr_files, a.n_train_traj, a.per_traj_train, seed=1),
                val=sample_instances(va_files, a.n_val_traj, a.per_traj, seed=2),
                test=test_instances(a.dump))
    pca = fit_pca(p, *sets['train'], dev)
    torch.save(dict(mean=pca[0].cpu(), V=pca[1].cpu()), OUT / 'pca.pt')
    for name in ('train', 'val', 'test'):
        trajs, inst = sets[name]; t0 = time.time()
        assert all(tr.shape[-1] == T_LEN for tr in trajs)
        f, hp, hv = backbone_pass(p, trajs, inst, dev, want_f=(name != 'train'), pca=pca)
        S, rows, tt, ids = queries(trajs, inst)
        cidx = np.concatenate([np.arange(trajs[j].shape[0]) for j, _ in inst])
        d = dict(S=torch.from_numpy(S), rows=torch.from_numpy(rows), t=torch.from_numpy(tt), ids=torch.from_numpy(ids),
                 h=torch.stack([hp[i][c] for i, c in zip(ids, cidx)]), hv=torch.stack([hv[i][c] for i, c in zip(ids, cidx)]),
                 inst=inst, n_inst=len(inst))
        if name != 'train':
            d['f'] = torch.stack([f[i][c] for i, c in zip(ids, cidx)])                   # raw units [nq, 512]
        torch.save(d, OUT / f'{name}.pt')
        print(f'[prep] {name}: {len(inst)} instances, {len(rows)} queries, {time.time() - t0:.0f}s', flush=True)
    json.dump(dict(val_systems=val_sys), open(OUT / 'split.json', 'w'))


# ------------------------------------------------------------------ train / test
def load_set(name, dev):
    d = torch.load(OUT / f'{name}.pt')
    se = Series(d['S'].to(dev))
    return d, se


def batch_inputs(se, d, idx, H, dev, arm):
    rows, t = d['rows'][idx].to(dev), d['t'][idx].to(dev)
    b = build(se, rows, t, H)
    extra = {}
    if arm in ('bil', 'path'):
        qs, cs = shapes(se, rows, t, b); extra = dict(qs=qs, cs=cs, t=t)
        if arm == 'path':
            extra.update(h=d['h'][idx].to(dev), hv=d['hv'][idx].to(dev).float())
    return rows, t, b, extra


def target(se, rows, t, b, H):
    kh = torch.arange(H, device=rows.device)
    y = se.S[rows[:, None], t[:, None] + kh]
    return (y - b['cm'][:, None]) / b['cs'][:, None]


def untrained_eps(se, d, sel, H, dev):
    """eps_q = MSE of the untrained rule (mean of the top-20 shape readouts); drops queries without candidates."""
    eps, zs, us, keep = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(sel), 512):
            idx = sel[i:i + 512]
            rows, t, b, _ = batch_inputs(se, d, idx, H, dev, 'fc')
            R = readout(se, rows, t, b, H); y = target(se, rows, t, b, H)
            N = R.shape[1] // 2
            un = R[:, N:N + 20].mean(1)                                               # shape readouts of top-20
            eps.append(((un - y).square().mean(1)).clamp_min(1e-3).cpu())
            keep.append(b['any'].cpu())
            zs.append(b['z'][b['valid']][::7].cpu()); us.append(b['u'].cpu())
    eps = torch.cat(eps); keep = torch.cat(keep)
    return sel[keep], eps[keep], zs, us


@torch.no_grad()
def eval_loss(model, se, d, sel, eps, H, dev, arm, bs=512):
    """training objective on held-out queries (clipped normalised MSE)."""
    model.eval(); tot = 0.0
    for i in range(0, len(sel), bs):
        idx = sel[i:i + bs]
        rows, t, b, ex = batch_inputs(se, d, idx, H, dev, arm)
        R = readout(se, rows, t, b, H); y = target(se, rows, t, b, H)
        r = (model(b, **ex)[..., None] * R).sum(1)
        tot += float(((r - y).square().mean(1) / eps[i:i + bs].to(dev)).clamp_max(C_MAX).sum())
    model.train()
    return tot / len(sel)


def run_train(a, dev):
    d, se = load_set('train', dev)
    H, arm = a.H, a.arm
    nq = len(d['rows']); g = torch.Generator().manual_seed(a.seed)
    sel = torch.randperm(nq, generator=g)[:a.n_q]
    # eps_q and normaliser statistics
    sel, eps, zs, us = untrained_eps(se, d, sel, H, dev)
    dv, sv = load_set('val', dev)
    vsel, veps, _, _ = untrained_eps(sv, dv, torch.arange(len(dv['rows'])), H, dev)
    torch.manual_seed(a.seed)
    model = Scorer(arm).to(dev)
    zv, uv = torch.cat(zs), torch.cat(us)
    model.zm.copy_(zv.mean(0)); model.zs.copy_(zv.std(0).clamp_min(1e-3)); model.um.copy_(uv.mean(0)); model.us.copy_(uv.std(0).clamp_min(1e-3))
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=a.wd)
    n_step = math.ceil(len(sel) / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * n_step)
    hist, vhist, best = [], [], (math.inf, -1, None)
    print(f'[train] arm {arm} H {H}: {len(sel)} queries, {n_step} steps/epoch', flush=True)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(sel), generator=g); tot = 0.0; t0 = time.time()
        for i in range(n_step):
            pi = perm[i * BATCH:(i + 1) * BATCH]; idx = sel[pi]
            rows, t, b, ex = batch_inputs(se, d, idx, H, dev, arm)
            R = readout(se, rows, t, b, H); y = target(se, rows, t, b, H)
            r = (model(b, **ex)[..., None] * R).sum(1)
            loss = ((r - y).square().mean(1) / eps[pi].to(dev)).clamp_max(C_MAX).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(EMA).add_(pm, alpha=1 - EMA)
            tot += float(loss.detach())
        hist.append(tot / n_step)
        vl = eval_loss(ema, sv, dv, vsel, veps, H, dev, arm) if a.tag else float('nan'); vhist.append(vl)
        if a.tag and vl < best[0]:
            best = (vl, ep, copy.deepcopy(ema.state_dict()))
        print(f'[train] arm {arm} H {H} epoch {ep} loss {hist[-1]:.4f} val {vl:.4f} {time.time() - t0:.0f}s' + (f' gamma {float(model.gamma):.3f}' if arm != 'fc' else ''), flush=True)
    if a.tag:
        print(f'[train] arm {arm} H {H} wd {a.wd}: best val {best[0]:.4f} at epoch {best[1]}', flush=True)
        torch.save(dict(state=best[2], hist=hist, vhist=vhist, best_val=best[0], best_ep=best[1], wd=a.wd), OUT / f'scorer_{arm}_H{H}_s{a.seed}{a.tag}.pt')
    else:
        torch.save(dict(state=ema.state_dict(), hist=hist), OUT / f'scorer_{arm}_H{H}_s{a.seed}.pt')


@torch.no_grad()
def predict_set(name, arm, H, seed, dev, bs=256, fname=None):
    d, se = load_set(name, dev)
    model = Scorer(arm).to(dev); model.load_state_dict(torch.load(OUT / (fname or f'scorer_{arm}_H{H}_s{seed}.pt'))['state']); model.eval()
    rs, ys, fs = [], [], []
    for i in range(0, len(d['rows']), bs):
        idx = torch.arange(i, min(i + bs, len(d['rows'])))
        rows, t, b, ex = batch_inputs(se, d, idx, H, dev, arm)
        f = (d['f'][idx, :H].to(dev) - b['cm'][:, None]) / b['cs'][:, None]
        if b['any'].any():
            R = readout(se, rows, t, b, H)
            r = (model(b, **ex)[..., None] * R).sum(1)
            r = torch.where(b['any'][:, None], r, f)
        else:
            r = f
        rs.append(r.cpu()); ys.append(target(se, rows, t, b, H).cpu()); fs.append(f.cpu())
    return d, torch.cat(rs), torch.cat(ys), torch.cat(fs)


def smape_traj(pred, y, ids, inst):
    """official aggregation: sMAPE (dysts, scaled) of each window over (steps, channels), averaged over the windows of
    a trajectory (compute_metrics with batch_axis = 0); one value per test trajectory (9347)."""
    e = ((pred - y).abs() / (pred.abs() + y.abs() + 1e-10)).double().mean(1)
    n = len(inst)
    s = torch.zeros(n, dtype=torch.float64).index_add_(0, ids, e)
    c = torch.zeros(n, dtype=torch.float64).index_add_(0, ids, torch.ones_like(e))
    per_win = 100 * s / c
    traj = torch.tensor([j for j, _ in inst])
    nt = int(traj.max()) + 1
    st = torch.zeros(nt, dtype=torch.float64).index_add_(0, traj, per_win)
    ct = torch.zeros(nt, dtype=torch.float64).index_add_(0, traj, torch.ones_like(per_win))
    return (st / ct)[ct > 0]


def run_test(a, dev):
    res = dict(arms={}, w={})
    for H in [int(x) for x in a.Hs.split(',')]:
        for arm in a.arms.split(','):
            fname = None
            if a.tags:   # model selection: weight decay and epoch with the lowest held-out (val) loss
                cands = [(torch.load(OUT / f'scorer_{arm}_H{H}_s{a.seed}{tg}.pt')['best_val'], tg) for tg in a.tags.split(',')]
                bv, tg = min(cands); fname = f'scorer_{arm}_H{H}_s{a.seed}{tg}.pt'
                ck = torch.load(OUT / fname); res.setdefault('sel', {})[f'{arm}_H{H}'] = dict(tag=tg, wd=ck['wd'], ep=ck['best_ep'], val_loss=bv, all={t: float(v) for v, t in cands})
                print(f'[select] {arm} H{H}: {tg} wd {ck["wd"]} epoch {ck["best_ep"]} val loss {bv:.4f} | ' + ' '.join(f'{t}:{v:.4f}' for v, t in cands), flush=True)
            dv, rv, yv, fv = predict_set('val', arm, H, a.seed, dev, fname=fname)
            dl = rv - fv; err = yv - fv
            w = float(((err * dl).sum() / dl.square().sum().clamp_min(1e-12)).clamp(0, 1))
            res['w'][f'{arm}_H{H}'] = w
            dt, rt, yt, ft = predict_set('test', arm, H, a.seed, dev, fname=fname)
            n = dt['n_inst']; ids = dt['ids']
            base = smape_traj(ft, yt, ids, dt['inst']); comb = smape_traj(ft + w * (rt - ft), yt, ids, dt['inst']); ronly = smape_traj(rt, yt, ids, dt['inst'])
            vb = smape_traj(fv, yv, dv['ids'], dv['inst']); vc = smape_traj(fv + w * (rv - fv), yv, dv['ids'], dv['inst'])
            res['arms'][f'{arm}_H{H}'] = dict(test_base=float(base.median()), test_comb=float(comb.median()), test_r=float(ronly.median()),
                                              test_base_mean=float(base.mean()), test_comb_mean=float(comb.mean()),
                                              val_base=float(vb.median()), val_comb=float(vc.median()), w=w,
                                              test_win_frac=float((comb < base).double().mean()))
            print(f'[test] {arm} H{H}: w {w:.3f} | test sMAPE base {base.median():.3f} comb {comb.median():.3f} r-only {ronly.median():.3f} '
                  f'| val base {vb.median():.3f} comb {vc.median():.3f}', flush=True)
            torch.save(dict(base=base, comb=comb, r=ronly), OUT / f'inst_{arm}_H{H}_s{a.seed}{a.out_tag}.pt')
    json.dump(res, open(OUT / f'results_s{a.seed}{a.out_tag}.json', 'w'), indent=1)


def run_check(a, dev):
    """the base forecaster on the dumped windows must reproduce the official metric (34.93 / 46.85 / 57.37)."""
    d = torch.load(OUT / 'test.pt'); se = Series(d['S'].to(dev))
    for H in HS:
        rows, t = d['rows'].to(dev), d['t'].to(dev)
        kc = t[:, None] - CTX + torch.arange(CTX, device=dev)[None]
        cx = se.S[rows[:, None], kc]; cm = cx.mean(1); cs = cx.std(1, unbiased=False).clamp_min(1e-10)
        y = se.S[rows[:, None], t[:, None] + torch.arange(H, device=dev)[None]]
        f = d['f'][:, :H].to(dev)
        s = smape_traj(((f - cm[:, None]) / cs[:, None]).cpu(), ((y - cm[:, None]) / cs[:, None]).cpu(), d['ids'], d['inst'])
        print(f'[check] L{H}: base median {s.median():.4f} mean {s.mean():.4f} n {len(s)}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prep', 'check', 'train', 'test'])
    ap.add_argument('--ckpt', default=str(ROOT / 'pat318/ckpt/p318ut_rope_s99_st100000-1/checkpoint-final'))
    ap.add_argument('--dump', default=str(OUT / 'test_dump.pkl'))
    ap.add_argument('--n_train_traj', type=int, default=6000); ap.add_argument('--n_val_traj', type=int, default=300)
    ap.add_argument('--per_traj', type=int, default=6); ap.add_argument('--per_traj_train', type=int, default=2)
    ap.add_argument('--arm', default='fc'); ap.add_argument('--arms', default='fc,bil,path')
    ap.add_argument('--H', type=int, default=128); ap.add_argument('--n_q', type=int, default=32000)
    ap.add_argument('--seed', type=int, default=99); ap.add_argument('--Hs', default='128,256,512')
    ap.add_argument('--wd', type=float, default=WD); ap.add_argument('--tag', default='')
    ap.add_argument('--tags', default=''); ap.add_argument('--out_tag', default='')
    a = ap.parse_args()
    dev = torch.device('cuda')
    OUT.mkdir(parents=True, exist_ok=True)
    {'prep': run_prep, 'check': run_check, 'train': run_train, 'test': run_test}[a.stage](a, dev)
