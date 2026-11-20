"""PAT-330 truncated-backbone baseline: at the ctx 1024 / 2048 test origins, the frozen Panda backbones (PaTH, RoPE) see only
the last 512 steps (their training length, no length extrapolation). Errors normalised by the ctx-window statistics, as in
fair_cmp.py, so numbers are paired with the ctx-restricted retrieval (val-selected method from valsel_fair{ctx}.pt).
Criterion (fixed before results): the long-ctx retrieval win holds at (ctx, H) if the paired median sMAPE difference vs the
better truncated backbone has 95% CI upper end < 0; MSE reported alongside. Sanity: the full-ctx PaTH f stored in test.pt must
reproduce the earlier PaTH numbers.
usage: P321_CTX=1024 P321_OUT=pat321/data_ctx1024_path python pat330/trunc.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pat327')); sys.path.insert(0, str(ROOT / 'pat329')); sys.path.insert(0, str(ROOT))
import numpy as np, torch
import fc_corr as C  # noqa: E402
from mse_eval import per_traj  # noqa: E402
FC = C.FC; dev = torch.device('cuda'); CTX, TR = FC.CTX, 512
CK = {'PaTH': 'pat318/ckpt/p318ut_path_s99_st100000-0/checkpoint-final', 'RoPE': 'pat318/ckpt/p318ut_rope_s99_st100000-1/checkpoint-final'}


@torch.no_grad()
def forecast(p, d):
    """per query [nq, 512] raw units: backbone on the last TR steps of its instance (all channels jointly)."""
    S, rows, t, ids = d['S'], d['rows'], d['t'], d['ids']
    out = torch.zeros(len(rows), 512)
    by = {}
    for i in range(d['n_inst']): by.setdefault(int((ids == i).sum()), []).append(i)
    qi = [torch.nonzero(ids == i)[:, 0] for i in range(d['n_inst'])]
    for nc, insts in by.items():
        for k in range(0, len(insts), 64):
            ch = insts[k:k + 64]
            ctx = torch.stack([S[rows[qi[i]]][:, int(t[qi[i][0]]) - TR:int(t[qi[i][0]])] for i in ch]).transpose(1, 2).to(dev)
            pr = p.predict(context=ctx, prediction_length=512, limit_prediction_length=False, sliding_context=True, verbose=False)
            pr = pr.reshape(len(ch), 512, nc).transpose(1, 2).float().cpu()
            for b, i in enumerate(ch): out[qi[i]] = pr[b]
    return out


def errors(pred, d, H):
    S, rows, t = d['S'], d['rows'], d['t']
    kc = t[:, None] - CTX + torch.arange(CTX)[None]; W = S[rows[:, None], kc]
    cm, cs = W.mean(1), W.std(1, unbiased=False).clamp_min(1e-10)
    y = (S[rows[:, None], t[:, None] + torch.arange(H)[None]] - cm[:, None]) / cs[:, None]
    p = (pred[:, :H] - cm[:, None]) / cs[:, None]
    sm = ((p - y).abs() / (p.abs() + y.abs() + 1e-10)).mean(1); ms = (p - y).square().mean(1)
    return {'smape': per_traj(sm, d['ids'], d['inst']), 'mse': per_traj(ms, d['ids'], d['inst'])}


if __name__ == '__main__':
    d = torch.load(FC.OUT / 'test.pt'); FC.CTX = CTX
    preds = {}
    for n, ck in CK.items():
        p = FC.load_backbone(ROOT / ck, dev); p.model.config.context_length = TR
        preds[f'{n}_trunc512'] = forecast(p, d); del p; torch.cuda.empty_cache()
    preds['PaTH_full'] = d['f']
    R = torch.load(ROOT / 'pat330' / 'out' / f'valsel_fair{CTX}.pt'); res = {}
    for H in (128, 256, 512):
        rng = np.random.default_rng(0)
        E = {n: errors(v, d, H) for n, v in preds.items()}
        val = {m: float(v['smape'].median()) for m, v in R[H]['val'].items()}; best = min(val, key=val.get); te = R[H]['test'][best]
        res[H] = E | {'retrieval': te}
        print(f'== ctx {CTX} H{H}: retrieval ({best}) sMAPE {100 * float(te["smape"].median()):.2f} MSE {float(te["mse"].median()):.3f}', flush=True)
        for n, e in E.items():
            o = []
            for met, sc in (('smape', 100), ('mse', 1)):
                d0, lo, hi, _ = C.boot(sc * te[met].numpy(), sc * e[met].numpy(), rng)
                o.append(f'{met} {sc * float(e[met].median()):.3f} d {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]')
            print(f'   vs {n:14s} ' + ' | '.join(o), flush=True)
    torch.save({H: {n: {k: v for k, v in e.items()} for n, e in r.items()} for H, r in res.items()} | {'preds': preds}, ROOT / 'pat330' / 'out' / f'trunc_ctx{CTX}.pt')
