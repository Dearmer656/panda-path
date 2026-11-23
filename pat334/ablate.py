"""PAT-334 ablation ladder: where does the gain over context parroting come from? (ctx2048 fair setting; also runs ctx512)
Same retrieval code as mh.py (scale family heads (l, 0), l in 16..96, same mode as the setting's headline M2, NMS radius 32,
readout Z[a, a + H), no valid analog -> Panda f). 2 x 2 factorial {single scale, 5 scales} x {top-1, top-k} + parroting + Gbt:
  P   : parrot_ctx (paper version, PAT-329) [ctx512: parrot_full]
  S1  : single scale, top-1                 val picks l
  Sk  : single scale, top-k weighted        val picks (l, u/w, k)
  M1  : 5 scales, top-1 per head, head mixing tau in TAUS         val picks tau
  Mk  : 5 scales, top-k = M2 (key from mh_{TAG}_H{H}.pt; sanity vs mh A)
  G   : Gbt (gbt_{TAG}_H{H}.pt)
Reported (descriptive; a step counts when its paired-bootstrap CI on the median difference excludes 0, sMAPE and MSE):
  S1 - P (implementation), M1 - S1 (multi-scale @ top-1), Sk - S1 (averaging @ single), Mk - Sk (multi-scale @ top-k),
  Mk - M1 (averaging @ multi), G - Mk (combination), Mk - P, G - P.  Also the val-chosen l / tau and the head share of M1.
usage: same env as mh.py + P334_H"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat334')); sys.path.insert(0, str(ROOT / 'pat333'))
import numpy as np, torch
import mh as M
from mh import G, FC, dev, chunks, backbone, TAG, MODE, futs, qerr
from reanchor import parse
from mse_eval import per_traj  # noqa: E402
O = ROOT / 'pat334' / 'out'; LS = (16, 32, 48, 64, 96)


@torch.no_grad()
def run_set(name, H, cfg):
    mode = cfg[0]
    d = torch.load(FC.OUT / f'{name}.pt'); S = d['S'].to(dev); fall = backbone(name, d)
    E, hsel = {}, []

    def put(k, p, y):
        s, m = qerr(p, y); E.setdefault(k, ([], []))[0].append(s.cpu()); E[k][1].append(m.cpu())
    for idx in chunks(d):
        rows, t, ids = d['rows'][idx].to(dev), d['t'][idx].to(dev), d['ids'][idx].to(dev)
        X = S[rows]; B, T = X.shape; ar = torch.arange(B, device=dev)
        win = X[ar[:, None], t[:, None] - G.CTX + torch.arange(G.CTX, device=dev)]
        cm, cs = win.mean(1), win.std(1, unbiased=False).clamp_min(1e-10); Z = (X - cm[:, None]) / cs[:, None]
        y = Z[ar[:, None], t[:, None] + torch.arange(H, device=dev)]
        f = (fall[idx, :H].to(dev) - cm[:, None]) / cs[:, None]
        loc = torch.unique(ids, return_inverse=True)[1]; ni = int(loc.max()) + 1
        rep = torch.zeros(ni, dtype=torch.long, device=dev).index_copy_(0, loc, ar)
        HD = []
        for l in LS:
            js, vs, dn = M.head(Z, t, H, l, 0, False, mode, loc, rep, ni)
            val = torch.isfinite(vs); fu = futs(Z, js, H); v1 = vs[:, :1]
            ww = torch.exp(-(vs - v1) / v1.abs().clamp_min(1e-8)).masked_fill(~val, 0.0)
            HD.append((fu, val, ww, torch.where(val[:, 0], dn, torch.full_like(dn, float('inf')))))
        PH = {}
        for wn in ('u', 'w'):
            for k in M.KS:
                ph = []
                for li, (fu, val, ww, dn) in enumerate(HD):
                    w = val[:, :k].float() if wn == 'u' else ww[:, :k]
                    p = (w[..., None] * fu[:, :k]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12); ph.append(p)
                    if k == 1 and wn == 'u': put(f'S1|{LS[li]}', torch.where(val[:, 0, None], p, f), y)
                    put(f'Sk|{LS[li]}|{wn}{k}', torch.where(val[:, 0, None], p, f), y)
                PH[(wn, k)] = torch.stack(ph, 1)
        has = torch.stack([h[1][:, 0] for h in HD], 1); dns = torch.stack([h[3] for h in HD], 1); anyh = has.any(1)

        def mix(tau, ph):
            if tau == 'uni': a = has.float()
            elif tau == 'best': a = torch.nn.functional.one_hot(dns.argmin(1), len(HD)).float() * has.float()
            else: a = torch.softmax((-dns / tau).masked_fill(~has, float('-inf')), 1).nan_to_num(0.0)
            return torch.where(anyh[:, None], (a[..., None] * ph).sum(1) / a.sum(1, keepdim=True).clamp_min(1e-12), f)
        for tau in M.TAUS: put(f'M1|{tau}', mix(tau, PH[('u', 1)]), y)
        _, _, wn, k, tau = cfg; put('Mk', mix(tau, PH[(wn, k)]), y)
        hsel.append(torch.where(anyh, dns.argmin(1), -1).cpu())
    order = torch.cat(chunks(d)); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    T = {k: {'smape': per_traj(torch.cat(s)[inv], d['ids'], d['inst']), 'mse': per_traj(torch.cat(m)[inv], d['ids'], d['inst'])}
         for k, (s, m) in E.items()}
    return T, torch.cat(hsel)[inv]


def med(x): return 100 * float(x['smape'].median())


def run(H):
    pk = torch.load(O / f'mh_{TAG}_H{H}.pt', weights_only=False); hm = pk['hm']; key = pk['pick'][f'{hm}_M2'][0]; cfg = parse(key)
    assert cfg[1] == 'scale', key
    V, _ = run_set('val', H, cfg); T, hsel = run_set('test', H, cfg)
    pick = {g: min([k for k in V if k.startswith(g + '|')], key=lambda k: med(V[k])) for g in ('S1', 'Sk', 'M1')}
    A = {'P': None, 'S1': T[pick['S1']], 'Sk': T[pick['Sk']], 'M1': T[pick['M1']], 'Mk': T['Mk']}
    dM = float((A['Mk']['smape'] - pk['A'][f'{hm}_M2']['smape']).abs().max())
    b9 = torch.load(ROOT / 'pat329' / 'out' / (f'mse_lq96_H{H}.pt' if MODE == 'full' else f'mse_ctx{G.CTX}_lq96_H{H}.pt'), weights_only=False)
    A['P'] = b9['parrot_full@LQ96' if MODE == 'full' else 'parrot_ctx@LQ96']
    A['G'] = torch.load(O / f'gbt_{TAG}_H{H}.pt', weights_only=False)['A']['G']
    for k_, v in A.items(): assert len(v['smape']) == len(A['Mk']['smape']), k_
    sh = torch.bincount(hsel[hsel >= 0], minlength=len(LS)).float(); sh = sh / sh.sum()
    print(f'== {TAG} ablation H{H}: M2 = {key}; sanity |Mk - mh M2| {dM:.1e}; picks S1 {pick["S1"]}  Sk {pick["Sk"]}  M1 {pick["M1"]}; '
          f'P = {"parrot_full" if MODE == "full" else "parrot_ctx"}', flush=True)
    print('   best-head share (test queries) ' + '  '.join(f'l{l} {float(s):.3f}' for l, s in zip(LS, sh)), flush=True)
    print('   test ' + '  '.join(f'{k_} {med(v):.2f}/{float(v["mse"].median()):.4f}' for k_, v in A.items()), flush=True)
    assert dM < 1e-4
    rng = np.random.default_rng(0)
    for x, rf, lab in (('S1', 'P', 'implementation'), ('M1', 'S1', 'multi-scale @ top-1'), ('Sk', 'S1', 'averaging @ single'),
                       ('Mk', 'Sk', 'multi-scale @ top-k'), ('Mk', 'M1', 'averaging @ multi'), ('G', 'Mk', 'combination'),
                       ('Mk', 'P', 'retrieval total'), ('G', 'P', 'all')):
        o = [f'{met} %+.3f [%+.3f, %+.3f]' % G.C.boot(sc * A[x][met].numpy(), sc * A[rf][met].numpy(), rng)[:3] for met, sc in (('smape', 100), ('mse', 1))]
        print(f'   {x:2s} - {rf:2s} {lab:22s} ' + ' | '.join(o), flush=True)
    torch.save({'A': A, 'pick': pick, 'key': key, 'hsel': hsel, 'share': sh}, O / f'ablate_{TAG}_H{H}.pt')


if __name__ == '__main__':
    for H in ([int(os.environ['P334_H'])] if os.environ.get('P334_H') else (128, 256, 512)): run(H)
