"""PAT-331 pilot: memory attention inside a frozen Panda (RoPE ckpt), backbone window always 512 steps, ctx 2048 origins, H128.
Memory of a query (one channel; Panda has no channel attention, so channels are independent) = patches of the same channel
inside [t - 2048, t) whose next 128 steps end <= t:
  windows m = 1, 2, 3 end at t - 512 m (all 32 patches), window 0 = the current window (patches k <= 23).
  key   = [frozen layer-LB state of the patch in its own 512 window ; (loc_m - loc_0) / scale_0, log(scale_m / scale_0), age]
  value = [raw next 128 steps after the patch end, scaled by the current window (loc_0, scale_0) ; same 3 features]
Memory layer = 8-head cross-attention current states -> memory, zero-init output projection, residual; inserted after LB
frozen layers (LB = 6 or 8), then the remaining frozen layers and the frozen head. Only the memory layer is trained.
usage: P321_LQ=96 P321_CTX=2048 P330_BLOCKS=6 P330_HIST=ctx P321_OUT=pat330/fair2048 python pat331/mem.py --lb 6 [--check]"""
import argparse, math, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--lb', type=int, default=6); ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--ntrain', type=int, default=96000); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327')); sys.path.insert(0, str(ROOT / 'pat329')); sys.path.insert(0, str(ROOT))
import numpy as np, torch
import torch.nn as nn
import fc_corr as C  # noqa: E402
from mse_eval import per_traj  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); assert FC.CTX == 2048
W, PL, H, NM, D = 512, 16, 128, 3, 512
CK = ROOT / 'pat318/ckpt/p318ut_rope_s99_st100000-1/checkpoint-final'
torch.manual_seed(99)


def load_bb():
    p = FC.load_backbone(CK, dev); p.model.config.context_length = W
    for q in p.model.parameters(): q.requires_grad_(False)
    return p.model


def enc(m, x, upto, h=None, start=0):
    """x [B, 512] raw -> (h after `upto` layers [B, 1, 32, D], loc [B], scale [B]); or continue from h at layer `start`."""
    if h is None:
        s, loc, sc = m.model.scaler(x[..., None], torch.ones_like(x[..., None]))
        h = m.model.encoder.embedder(m.model.patchifier(s)); loc, sc = loc[:, 0, 0], sc[:, 0, 0]
    else:
        loc = sc = None
    for L in m.model.encoder.layers[start:upto]: h = L(hidden_state=h)[0]
    return h, loc, sc


class Mem(nn.Module):
    def __init__(self):
        super().__init__()
        self.nq, self.nk = nn.LayerNorm(D), nn.LayerNorm(D)
        self.q = nn.Linear(D, D); self.k = nn.Linear(D + 3, D); self.v = nn.Linear(H + 3, D); self.o = nn.Linear(D, D)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def forward(self, h, hm, fk, fut, valid):
        B, N = hm.shape[:2]; nh, dh = 8, D // 8
        q = self.q(self.nq(h[:, 0])).view(B, -1, nh, dh).transpose(1, 2)                        # [B, nh, 32, dh]
        k = self.k(torch.cat([self.nk(hm), fk], -1)).view(B, N, nh, dh).transpose(1, 2)
        v = self.v(torch.cat([fut, fk], -1)).view(B, N, nh, dh).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        att = torch.softmax(s.masked_fill(~valid[:, None, None], -1e9), -1)
        return h + self.o((att @ v).transpose(1, 2).reshape(B, -1, D))[:, None]


@torch.no_grad()
def memory(m, S, rows, t, mrows, mt, h0, loc0, sc0, lb):
    """memory entries from (mrows, mt) (= (rows, t) normally; permuted for the random-memory control), scaled by the
    current window (loc0, sc0). Returns hm [B, N, D], features [B, N, 3], futures [B, N, H], valid [B, N], ends [B, N]."""
    B = len(rows); ar = torch.arange(B, device=dev); X = S[mrows]
    HM, FK, FU, V, E = [], [], [], [], []
    for mi in range(NM + 1):
        e = mt - W * mi
        if mi == 0 and mrows is rows and mt is t: hm, lm, sm = h0[:, 0], loc0, sc0
        else:
            x = X[ar[:, None], e[:, None] - W + torch.arange(W, device=dev)]; hh, lm, sm = enc(m, x, lb); hm = hh[:, 0]
        ends = e[:, None] - W + PL * (torch.arange(W // PL, device=dev)[None] + 1)               # patch end positions
        ok = (ends + H <= mt[:, None]) & (e[:, None] - W >= 0)
        fu = X[ar[:, None, None], (ends.clamp(max=X.shape[1] - H)[..., None] + torch.arange(H, device=dev))]
        fu = ((fu - loc0[:, None, None]) / sc0[:, None, None]).clamp(-10, 10)
        fk = torch.stack([((lm - loc0) / sc0).clamp(-10, 10)[:, None].expand_as(ends), torch.log(sm / sc0).clamp(-10, 10)[:, None].expand_as(ends),
                          (mt[:, None] - ends).float() / FC.CTX], -1)
        HM.append(hm); FK.append(fk); FU.append(fu); V.append(ok); E.append(ends)
    return torch.cat(HM, 1), torch.cat(FK, 1), torch.cat(FU, 1), torch.cat(V, 1), torch.cat(E, 1)


def forward(m, mem, S, rows, t, lb, perm_rows=None, perm_t=None):
    ar = torch.arange(len(rows), device=dev)
    x = S[rows][ar[:, None], t[:, None] - W + torch.arange(W, device=dev)]
    with torch.no_grad(): h, loc, sc = enc(m, x, lb)
    if mem is not None:
        mr, mt = (rows, t) if perm_rows is None else (perm_rows, perm_t)
        hm, fk, fu, v, _ = memory(m, S, rows, t, mr, mt, h, loc, sc, lb)
        h = mem(h, hm, fk, fu, v)
    h, _, _ = enc(m, None, len(m.model.encoder.layers), h=h, start=lb)
    yh = m.head(h)[:, :, 0]                                                                       # scaled units [B, 128]
    return yh, loc, sc


def batch_loss(m, mem, se, d, idx, lb):
    rows, t = d['rows'][idx].to(dev), d['t'][idx].to(dev)
    yh, loc, sc = forward(m, mem, se.S, rows, t, lb)
    y = (se.S[rows[:, None], t[:, None] + torch.arange(H, device=dev)] - loc[:, None]) / sc[:, None]
    return (yh - y).square().mean()


@torch.no_grad()
def predict(m, mem, name, lb, perm=False):
    """raw-unit H128 forecasts per query, plus the retrieval (smap_0.01, fair2048 scorer) in ctx-normalised units."""
    d, se = FC.load_set(name, dev); n = len(d['rows']); out = torch.zeros(n, H)
    pi = torch.randperm(n, generator=torch.Generator().manual_seed(0)) if perm else None
    for i in range(0, n, 64):
        idx = torch.arange(i, min(i + 64, n)); rows, t = d['rows'][idx].to(dev), d['t'][idx].to(dev)
        pr, pt = (d['rows'][pi[idx]].to(dev), d['t'][pi[idx]].to(dev)) if perm else (None, None)
        yh, loc, sc = forward(m, mem, se.S, rows, t, lb, pr, pt); out[idx] = (yh * sc[:, None] + loc[:, None]).cpu()
    return out


def ctx_err(pred_raw, d, se):
    """per-trajectory sMAPE / MSE with the ctx-2048 window statistics (as fair_cmp / trunc.py)."""
    S, rows, t = d['S'], d['rows'], d['t']
    Wc = S[rows[:, None], t[:, None] - FC.CTX + torch.arange(FC.CTX)[None]]; cm, cs = Wc.mean(1), Wc.std(1, unbiased=False).clamp_min(1e-10)
    y = (S[rows[:, None], t[:, None] + torch.arange(H)[None]] - cm[:, None]) / cs[:, None]
    p = (pred_raw - cm[:, None]) / cs[:, None]
    return p, y


def check():
    m = load_bb(); mem = Mem().to(dev); d, se = FC.load_set('test', dev)
    idx = torch.arange(0, 512); rows, t = d['rows'][idx].to(dev), d['t'][idx].to(dev)
    with torch.no_grad():
        y0, loc, sc = forward(m, None, se.S, rows, t, a.lb); y1, _, _ = forward(m, mem, se.S, rows, t, a.lb)
        raw = (y0 * sc[:, None] + loc[:, None]).cpu()
        ref = torch.load(ROOT / 'pat330' / 'out' / 'trunc_ctx2048.pt')['preds']['RoPE_trunc512'][:512, :H]
        err = float((raw - ref).abs().max() / ref.abs().max()); assert err < 1e-3, f'univariate path != pipeline predict ({err})'
        assert torch.allclose(y0, y1), 'zero-init memory changes the output'
        h, l0, s0 = enc(m, se.S[rows][torch.arange(512, device=dev)[:, None], t[:, None] - W + torch.arange(W, device=dev)], a.lb)
        _, _, _, v, E = memory(m, se.S, rows, t, rows, t, h, l0, s0, a.lb)
        assert ((E + H <= t[:, None]) | ~v).all() and v.sum(1).min() > 0, 'leakage / empty memory'
    print(f'check ok: univariate == pipeline (rel {err:.2e}), zero-init identity, no leakage, memory size {int(v.sum(1).float().mean())}', flush=True)


def fit():
    m = load_bb(); mem = Mem().to(dev); d, se = FC.load_set('train', dev); dv, sv = FC.load_set('val', dev)
    sel = torch.randperm(len(d['rows']), generator=torch.Generator().manual_seed(99))[:a.ntrain]
    vsel = torch.randperm(len(dv['rows']), generator=torch.Generator().manual_seed(98))[:8192]
    opt = torch.optim.AdamW(mem.parameters(), lr=5e-4, weight_decay=1e-2)
    steps = a.epochs * math.ceil(len(sel) / 128); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=steps, pct_start=0.05)

    def vloss():
        with torch.no_grad(): return float(np.mean([float(batch_loss(m, mem, sv, dv, vsel[i:i + 128], a.lb)) for i in range(0, len(vsel), 128)]))
    best = (vloss(), {k: v.clone() for k, v in mem.state_dict().items()}); print(f'[mem lb{a.lb}] val loss at init (= backbone) {best[0]:.4f}', flush=True)
    for ep in range(a.epochs):
        t0 = time.time(); tot = 0.0; perm = sel[torch.randperm(len(sel), generator=torch.Generator().manual_seed(ep))]
        for j, i in enumerate(range(0, len(perm), 128)):
            l = batch_loss(m, mem, se, d, perm[i:i + 128], a.lb); opt.zero_grad(); l.backward(); opt.step(); sch.step(); tot += float(l)
            if j % 200 == 0: print(f'   ep {ep} step {j} loss {tot / (j + 1):.4f} {time.time() - t0:.0f}s', flush=True)
        vl = vloss(); print(f'[mem lb{a.lb}] ep {ep} train {tot / (j + 1):.4f} val {vl:.4f} {time.time() - t0:.0f}s', flush=True)
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in mem.state_dict().items()})
    mem.load_state_dict(best[1]); torch.save(best[1], ROOT / 'pat331' / 'out' / f'mem_lb{a.lb}.pt'); return m, mem


if __name__ == '__main__':
    (ROOT / 'pat331' / 'out').mkdir(exist_ok=True)
    if a.check: check(); sys.exit()
    m, mem = fit(); res = {}
    for name in ('val', 'test'):
        res[name] = {'mem': predict(m, mem, name, a.lb), 'mem_rand': predict(m, mem, name, a.lb, perm=True)}
        if a.lb == 6: res[name]['bb'] = predict(m, None, name, a.lb)
    torch.save(res, ROOT / 'pat331' / 'out' / f'pred_lb{a.lb}.pt'); print('saved', flush=True)
