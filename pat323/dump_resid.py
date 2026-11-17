"""PAT-323 step 1 (inference only): rolling-origin residuals of the frozen PaTH checkpoint on test_zeroshot systems.
Checkpoint p318ut_path_s99_st100000 (PatchTST-univariate PaTH, seed 99), ctx 512, native prediction 128 (one forward,
official PatchTSTPipeline.predict). Per system: first 9 trajectories by sample index (eval_utils order), split A/B/V
round-robin (traj 0,3,6 -> A; 1,4,7 -> B; 2,5,8 -> V). Origins t = 512, 544, ..., 3968 (stride 32).
Residual e[t, h] = yhat[t + h | t] - y[t + h], each channel divided by the trajectory's std over x[0:512] (causal, fixed
per trajectory). Saved leads h = 1..128 subsampled at H_SAVE. Also the official-style sMAPE L128 (context-standardised)
per window as a sanity check against the official median (34.40).
usage: python pat323/dump_resid.py OUT.pkl [max_systems]"""
import sys, pickle, time, numpy as np, torch, transformers
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panda.utils.eval_utils import get_eval_data_dict
from panda.patchtst.pipeline import PatchTSTPipeline
from pat318.exact_load import exact_load
CK = 'pat318/ckpt/p318ut_path_s99_st100000-0/checkpoint-final'
CTX, PRED, STRIDE, NTRAJ = 512, 128, 32, 9
H_SAVE = [1, 8, 16, 32, 64, 96, 128]
D = get_eval_data_dict(['data/final_base40/test_zeroshot', 'data/final_skew40/test_zeroshot'], num_subdirs=None, one_dim_target=False)
FILES = {}
for d in ['data/final_base40/test_zeroshot', 'data/final_skew40/test_zeroshot']:   # same file order as get_eval_data_dict
    for sd in Path(d).iterdir():
        if sd.is_dir(): FILES.setdefault(sd.name, []).extend(str(f) for f in sd.glob('*'))
FILES = {k: sorted(v, key=lambda x: int(Path(x).stem.split('_')[0])) for k, v in FILES.items()}
names = sorted(D)[: int(sys.argv[2])] if len(sys.argv) > 2 else sorted(D)
transformers.set_seed(99); p = PatchTSTPipeline.from_pretrained(mode='predict', pretrain_path=CK); exact_load(p.model, CK)
p.model.cuda().eval()
out = dict(ckpt=CK, ctx=CTX, stride=STRIDE, h=H_SAVE, split=['A', 'B', 'V'] * 3, systems={})
t0 = time.time()
for si, name in enumerate(names):
    fl = D[name][:NTRAJ]
    if len(fl) < NTRAJ:
        print('skip', name, len(fl)); continue
    X = np.stack([next(iter(f))['target'] for f in fl]).astype(np.float32)        # [9, Dim, T]
    T = X.shape[2]; origins = np.arange(CTX, T - PRED + 1, STRIDE)
    scale = X[:, :, :CTX].std(2) + 1e-8                                              # [9, Dim]
    ctx = np.stack([X[:, :, t - CTX:t] for t in origins], 1)                         # [9, O, Dim, CTX]
    n, O, Dm, _ = ctx.shape
    flat = torch.tensor(ctx.reshape(n * O, Dm, CTX).transpose(0, 2, 1).copy())
    preds = []
    for i in range(0, len(flat), 1024):
        preds.append(p.predict(flat[i:i + 1024].cuda(), PRED, verbose=False)[:, 0].float().cpu())   # [b, PRED, Dim]
    yhat = torch.cat(preds).numpy().reshape(n, O, PRED, Dm)
    y = np.stack([X[:, :, t:t + PRED] for t in origins], 1).transpose(0, 1, 3, 2)   # [9, O, PRED, Dim]
    E = (yhat - y) / scale[:, None, None, :]
    m, s = ctx.mean(3), ctx.std(3) + 1e-10                                            # official standardisation
    a, b = (yhat - m[:, :, None]) / s[:, :, None], (y - m[:, :, None]) / s[:, :, None]
    smape = 100 * np.mean(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-10), axis=(2, 3))   # [9, O]
    out['systems'][name] = dict(dim=Dm, origins=origins, files=FILES[name][:NTRAJ], scale=scale,
                                E=E[:, :, [h - 1 for h in H_SAVE]].astype(np.float32), smape=smape.astype(np.float32))
    if si % 20 == 0:
        print(f'{si}/{len(names)} {name} dim {Dm} O {O} smape med {np.median(smape):.2f} {time.time() - t0:.0f}s', flush=True)
pickle.dump(out, open(sys.argv[1], 'wb'))
allsm = np.concatenate([v['smape'].ravel() for v in out['systems'].values()])
print('systems', len(out['systems']), 'windows', allsm.size, 'smape L128 median', np.median(allsm), flush=True)
