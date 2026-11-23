"""PAT-334: which difference between paper-version parroting (parrot_ctx) and our single-scale top-1 (S1) gives the gain?
Same Z / candidate code as mh.py (ctx window [t - CTX, t), match length D = 16 = parrot_ctx's val choice), 1-NN, no valid
match -> Panda f. 2 x 2 factorial:
  P0 : per channel, periodic continuation (paper version: anchor a <= t - 1, copy Z[a + (h mod (t - a))])
  P1 : per channel, full observed continuation (a + H <= t, copy Z[a + h])
  P2 : joint channels (distances summed over the instance's channels, one anchor), periodic
  P3 : joint channels, full continuation  (= S1 when S1's val pick is l = 16)
Reported (descriptive): P1 - P0 (continuation), P2 - P0 (joint), P3 - P0 (both), P3 - P1, P3 - P2; sanity P0 vs saved
parrot_ctx (pat329), P3 vs ablate S1 when its pick is l = 16; share of P0 anchors with t - a < H (needs periodic fill).
usage: same env as mh.py + P334_H"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import mh as M
from mh import G, FC, dev, chunks, backbone, TAG, MODE, qerr
from mse_eval import per_traj  # noqa: E402
O = ROOT / 'pat334' / 'out'; D = 16


@torch.no_grad()
def run(H):
    d = torch.load(FC.OUT / 'test.pt'); S = d['S'].to(dev); fall = backbone('test', d)
    E, short = {}, []
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        hz = torch.arange(H, device=dev); y = Z[ar[:, None], t[:, None] + hz]
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)
        a, dist, _ = M.cand(Z, t, H, D, 0, False)
        inwin = (a - D)[None] >= (t - G.CTX)[:, None]
        okF = inwin & (a[None] + H <= t[:, None]); okP = inwin & (a[None] <= (t - 1)[:, None])
        for nm, joint, ok, full in (('P0', False, okP, False), ('P1', False, okF, True), ('P2', True, okP, False), ('P3', True, okF, True)):
            dd, oo = dist, ok
            if joint:
                dd = torch.zeros(ni, len(a), device=dev).index_add_(0, loc, dist.masked_fill(~ok, 0.0))[loc]; oo = ok[rep][loc]
            dm = dd.masked_fill(~oo, float('inf')); v, j = dm.min(1); has = torch.isfinite(v); ab = a[j]
            ix = ab[:, None] + (hz[None] if full else hz[None] % (t - ab).clamp_min(1)[:, None])
            p = torch.where(has[:, None], Z[ar[:, None], ix.clamp(max=T - 1)], f)
            s_, m_ = qerr(p, y); E.setdefault(nm, ([], []))[0].append(s_.cpu()); E[nm][1].append(m_.cpu())
            if nm == 'P0': short.append(((t - ab) < H).float().cpu())
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    A = {k: {'smape': per_traj(torch.cat(s)[inv], d['ids'], d['inst']), 'mse': per_traj(torch.cat(m)[inv], d['ids'], d['inst'])}
         for k, (s, m) in E.items()}
    sv = torch.load(ROOT / 'pat329' / 'out' / f'mse_ctx{G.CTX}_lq96_H{H}.pt', weights_only=False)['parrot_ctx@LQ96']
    ab_ = torch.load(O / f'ablate_{TAG}_H{H}.pt', weights_only=False)
    dP = A['P0']['smape'] - sv['smape']
    msg = f'S1 pick {ab_["pick"]["S1"]}'
    if ab_['pick']['S1'] == 'S1|16': msg += f', |P3 - S1| max {float((A["P3"]["smape"] - ab_["A"]["S1"]["smape"]).abs().max()):.1e}'
    print(f'== {TAG} parrot ablation H{H} (D = {D}): sanity P0 vs saved parrot_ctx median {100 * float(sv["smape"].median()):.2f} -> '
          f'{100 * float(A["P0"]["smape"].median()):.2f}, max |diff| {float(dP.abs().max()):.1e}; {msg}; '
          f'P0 anchors needing periodic fill {float(torch.cat(short).mean()):.3f}', flush=True)
    print('   test ' + '  '.join(f'{k} {100 * float(v["smape"].median()):.2f}/{float(v["mse"].median()):.4f}' for k, v in A.items()), flush=True)
    rng = np.random.default_rng(0)
    for x, rf, lab in (('P1', 'P0', 'continuation'), ('P2', 'P0', 'joint channels'), ('P3', 'P0', 'both'), ('P3', 'P1', 'joint | full'), ('P3', 'P2', 'full | joint')):
        o = [f'{met} %+.3f [%+.3f, %+.3f]' % G.C.boot(sc * A[x][met].numpy(), sc * A[rf][met].numpy(), rng)[:3] for met, sc in (('smape', 100), ('mse', 1))]
        print(f'   {x} - {rf} {lab:15s} ' + ' | '.join(o), flush=True)
    torch.save({'A': A}, O / f'parrot_abl_{TAG}_H{H}.pt')


if __name__ == '__main__':
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else (128, 256, 512)): run(H)
