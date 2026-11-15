"""PAT-318 reference: DynaMix (Hemmer & Durstewitz, NeurIPS 2025; official weights DurstewitzLab/dynamix, code
DurstewitzLab/DynaMix-python @39cc555), zero-shot on the exact official Panda test windows (pat321/data/test_dump.pkl:
9347 trajectories x 6 origins), context 512, horizon 512.

Two settings, both with the authors' notebook recipes:
  uni   each channel forecast alone (= the PatchTST-univariate setting): dynamix-6d, pos_embedding, standardize
        (example_evaluation_TS.ipynb)
  multi all channels jointly: dynamix-3d if dim <= 3, dynamix-6d if dim <= 6, zero_embedding, standardize
        (example_evaluation_DSR.ipynb); trajectories with dim > 6 are forecast per channel by DynaMix itself
Metric = official: values standardised by the context's mean / std, sMAPE averaged over (steps, channels) per window,
over the 6 windows per trajectory, median over trajectories, at L = 128 / 256 / 512 (prefixes); also MAE / MSE.
usage: python pat318/dynamix_eval.py --mode uni|multi"""
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'DynaMix-python' / 'src'))
from dynamix.model.forecaster import DynaMixForecaster  # noqa: E402
from dynamix.utilities.utilities import load_hf_model  # noqa: E402

CTX, HOR, LS = 512, 512, (128, 256, 512)


def load_dump(path):
    recs = []
    with open(path, 'rb') as f:
        while True:
            try:
                recs.append(pickle.load(f))
            except EOFError:
                break
    return recs


def metrics(pred, y, cx):
    """pred, y [C, H]; cx [C, CTX] -> per-L (smape, mae, mse) of one window, official standardisation."""
    m, s = cx.mean(1, keepdims=True), cx.std(1, keepdims=True) + 1e-10
    p, t = (pred - m) / s, (y - m) / s
    out = {}
    for L in LS:
        a, b = p[:, :L], t[:, :L]
        out[L] = (100 * np.mean(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-10)), np.mean(np.abs(a - b)), np.mean((a - b) ** 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='uni', choices=['uni', 'multi'])
    ap.add_argument('--dump', default=str(ROOT / 'pat321/data/test_dump.pkl'))
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--max_traj', type=int, default=0)
    ap.add_argument('--out', default=str(ROOT / 'pat318/eval/dynamix'))
    a = ap.parse_args()
    dev = torch.device('cuda')
    recs = load_dump(a.dump)
    if a.max_traj:
        recs = recs[:a.max_traj]
    dims = np.array([r[0].shape[0] for r in recs])
    print(f'[dynamix] {len(recs)} trajectories, dims {dict(zip(*np.unique(dims, return_counts=True)))}', flush=True)
    models = {k: DynaMixForecaster(load_hf_model(f'dynamix-{k}-alrnn-v1.0').to(dev).eval()) for k in ('3d', '6d')}
    # jobs: (traj j, window w, channels, model key, embedding)
    jobs = []
    for j, (ts, idx) in enumerate(recs):
        C = ts.shape[0]
        for w, t in enumerate(idx):
            if a.mode == 'uni':
                for c in range(C):
                    jobs.append((j, w, (c,), '6d', 'pos_embedding'))
            else:
                key = '3d' if C <= 3 else '6d'
                jobs.append((j, w, tuple(range(C)), key, 'zero_embedding'))
    preds = {}
    t0 = time.time()
    groups = {}
    for jb in jobs:
        groups.setdefault((jb[3], jb[4], len(jb[2])), []).append(jb)
    for (key, emb, nc), lst in groups.items():
        for i in range(0, len(lst), a.bs):
            chunk = lst[i:i + a.bs]
            ctx = np.stack([recs[j][0][list(ch), recs[j][1][w] - CTX:recs[j][1][w]].T for j, w, ch, _, _ in chunk], 1)  # [CTX, S, nc]
            with torch.no_grad():
                out = models[key].forecast(context=torch.tensor(ctx, dtype=torch.float32, device=dev), horizon=HOR,
                                           preprocessing_method=emb, standardize=True, fit_nonstationary=False)
            out = out.float().cpu().numpy()  # [HOR, S, nc]
            for k, (j, w, ch, _, _) in enumerate(chunk):
                for q, c in enumerate(ch):
                    preds[(j, w, c)] = out[:, k, q]
            print(f'[dynamix] {key} {emb} nc={nc}: {min(i + a.bs, len(lst))}/{len(lst)} {time.time() - t0:.0f}s', flush=True)
    per_traj = {L: {'smape': [], 'mae': [], 'mse': []} for L in LS}
    nonfinite = 0
    for j, (ts, idx) in enumerate(recs):
        C = ts.shape[0]; acc = {L: [] for L in LS}
        for w, t in enumerate(idx):
            pred = np.stack([preds[(j, w, c)] for c in range(C)])
            if not np.isfinite(pred).all():
                nonfinite += 1; pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            mm = metrics(pred, ts[:, t:t + HOR], ts[:, t - CTX:t])
            for L in LS:
                acc[L].append(mm[L])
        for L in LS:
            v = np.array(acc[L]).mean(0)
            for k, name in enumerate(('smape', 'mae', 'mse')):
                per_traj[L][name].append(float(v[k]))
    res = {f'{n}_L{L}': float(np.median(per_traj[L][n])) for L in LS for n in ('smape', 'mae', 'mse')}
    res.update(n_traj=len(recs), nonfinite_windows=nonfinite, mode=a.mode)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=res, per_traj=per_traj), open(Path(a.out) / f'{a.mode}.json', 'w'))
    print('[dynamix] ' + ' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in res.items()), flush=True)


if __name__ == '__main__':
    main()
