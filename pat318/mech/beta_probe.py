"""PAT-318: what do the trained PaTH transitions do? (PatchTST-univariate path / rope_path, seed 99, inference only)
H_t = I - beta_t w_t w_t^T, eigenvalue 1-beta_t along w_t (beta 0 = identity, 1 = projection, 2 = pure reflection).
Per layer (heads pooled and per head), on official test windows (context 512 = 32 patches):
  beta distribution; |1-beta| (contraction along w); variance split of beta (across heads / across sequences / within seq)
  cos(w_t, w_{t+1}) (two reflections with angle theta between planes = rotation by 2 theta)
  cumulative transport P_{i<-j} = H_i..H_{j+1} applied to the real k_j, by distance d: cos(P k, k), |P k| / |k|
  how much PaTH changes the logits: corr over (i, j) of path score vs plain q.k
  link to dynamics: within-sequence Spearman of beta_t with the patch's phase speed (mean |dx| of z-scored channel)
one trajectory per system (all 480 systems, 6 windows each); usage: python pat318/mech/beta_probe.py N_SUBDIRS ARM [ARM ...]"""
import sys, json, numpy as np, torch, transformers
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from panda.dataset import MultivariateTimeSeriesDataset
from panda.utils.eval_utils import get_eval_data_dict
from panda.patchtst.pipeline import PatchTSTPipeline
import panda.patchtst.patchtst as PT
from pat318.exact_load import exact_load
torch.set_num_threads(16)
CTX, NH, DH, PL = 512, 8, 64, 16
D = get_eval_data_dict(['data/final_base40/test_zeroshot', 'data/final_skew40/test_zeroshot'], num_subdirs=int(sys.argv[1]))
W = []
for name, fl in D.items():
    fl = fl[:1]; ds = MultivariateTimeSeriesDataset(datasets=fl, probabilities=[1 / len(fl)] * len(fl), context_length=CTX, prediction_length=512,
                                       num_test_instances=6, window_style='sampled', window_stride=1, model_type='predict', mode='test')
    W += [e['past_values'].numpy() for e in ds]
print('windows', len(W), flush=True)
cap = []
orig = PT.bidir_path_scores
def rec(q, k, w, beta):
    s = orig(q, k, w, beta); cap.append((q.detach().float(), k.detach().float(), w.detach().float(), beta.detach().float(), s.detach().float())); return s
PT.bidir_path_scores = rec
def transport(k, w, beta):
    """k, w [B, N, E], beta [B, N] -> cos[d], ratio[d] for causal P_{i<-j}, d = i - j = 1..N-1 (mean over B, j)."""
    B, N, E = k.shape; cs = np.zeros(N); rt = np.zeros(N); cnt = np.zeros(N)
    for j in range(N - 1):
        v = k[:, j].clone(); k0 = k[:, j]
        for t in range(j + 1, N):
            v = v - beta[:, t, None] * w[:, t] * (w[:, t] * v).sum(-1, keepdim=True)
            d = t - j
            cs[d] += torch.nn.functional.cosine_similarity(v, k0, dim=-1).sum().item()
            rt[d] += (v.norm(dim=-1) / (k0.norm(dim=-1) + 1e-8)).sum().item(); cnt[d] += B
    return cs[1:] / cnt[1:], rt[1:] / cnt[1:]
def spearman_rows(a, b):
    ra = a.argsort(1).argsort(1).astype(float); rb = b.argsort(1).argsort(1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1) + 1e-12)
OUT = {}
for arm in sys.argv[2:]:
    path = f'pat318/ckpt/p318ut_{arm}_s99_st100000-0/checkpoint-final'
    transformers.set_seed(99); p = PatchTSTPipeline.from_pretrained(mode='predict', pretrain_path=path); exact_load(p.model, path); m = p.model.eval()
    NL = len(m.model.encoder.layers)
    acc = {l: dict(beta=[], wcos=[], corr=[], sp=[], cs=[], rt=[]) for l in range(NL)}
    BS = 32
    with torch.no_grad():
        for i in range(0, len(W), BS):
            Bt = W[i:i + BS]; C = [b.shape[1] for b in Bt]
            P = [b for b in Bt if b.shape[1] == Bt[0].shape[1]]
            for cdim in sorted(set(C)):
                P = np.stack([b for b in Bt if b.shape[1] == cdim]); cap.clear()
                m(past_values=torch.tensor(P, dtype=torch.float32))
                z = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-8)          # [b, CTX, C]
                spd = np.abs(np.diff(z, axis=1, prepend=z[:, :1])).reshape(len(P), CTX // PL, PL, cdim).mean(2)  # [b, N, C]
                spd = np.repeat(spd.transpose(0, 2, 1).reshape(-1, CTX // PL), NH, 0)           # [b*C*NH, N] (bsz = b*C, then heads)
                assert len(cap) == NL, len(cap)
                for l, (q, k, w, beta, s) in enumerate(cap):
                    a = acc[l]; bnp = beta.numpy()
                    a['beta'].append(bnp.reshape(-1, NH, bnp.shape[-1]))                        # [b*C, NH, N]
                    a['wcos'].append((w[:, 1:] * w[:, :-1]).sum(-1).numpy().reshape(-1, NH, w.shape[1] - 1))
                    plain = q @ k.transpose(1, 2)
                    sc, pc = s - s.mean((1, 2), keepdim=True), plain - plain.mean((1, 2), keepdim=True)
                    a['corr'].append(((sc * pc).sum((1, 2)) / (sc.norm(dim=(1, 2)) * pc.norm(dim=(1, 2)) + 1e-8)).numpy().reshape(-1, NH))
                    a['sp'].append(spearman_rows(bnp, spd).reshape(-1, NH))
                    if i % (4 * BS) == 0:
                        cs, rt = transport(k, w, beta); a['cs'].append(cs); a['rt'].append(rt)
            print(f'{arm} {min(i + BS, len(W))}/{len(W)}', flush=True)
    res = {}
    for l in range(NL):
        a = acc[l]; b = np.concatenate(a['beta'])                                                   # [S, NH, N]
        tot = b.var(); head = b.mean((0, 2)).var(); seq = (b.mean(2) - b.mean((0, 2))).var(); within = b.var(2).mean()
        wc = np.concatenate(a['wcos']); cr = np.concatenate(a['corr']); sp = np.concatenate(a['sp'])
        cs = np.mean(a['cs'], 0); rt = np.mean(a['rt'], 0)
        res[l] = dict(beta_q=np.percentile(b, [5, 25, 50, 75, 95]).round(3).tolist(), beta_head_mean=b.mean((0, 2)).round(3).tolist(),
                      frac_lt01=float((b < 0.1).mean()), frac_gt19=float((b > 1.9).mean()), frac_09_11=float((np.abs(b - 1) < 0.1).mean()),
                      contraction_med=float(np.median(np.abs(1 - b))), var_split=dict(head=head / tot, seq=seq / tot, within=within / tot),
                      wcos_med=float(np.median(wc)), wcos_abs_med=float(np.median(np.abs(wc))), score_corr_med=float(np.median(cr)),
                      score_corr_head=np.median(cr, 0).round(3).tolist(), beta_speed_rho_mean=float(np.nanmean(sp)),
                      beta_speed_rho_head=np.nanmean(sp, 0).round(3).tolist(),
                      transport_cos_d=[float(cs[d - 1]) for d in (1, 2, 4, 8, 16, 31)], transport_ratio_d=[float(rt[d - 1]) for d in (1, 2, 4, 8, 16, 31)])
        r = res[l]
        print(f'{arm} L{l}: beta q5/25/50/75/95 {r["beta_q"]} | <0.1 {r["frac_lt01"]:.2f} ~1 {r["frac_09_11"]:.2f} >1.9 {r["frac_gt19"]:.2f} | '
              f'var head/seq/within {head / tot:.2f}/{seq / tot:.2f}/{within / tot:.2f} | cos(w_t,w_t+1) {r["wcos_med"]:+.2f} |abs| {r["wcos_abs_med"]:.2f} | '
              f'corr(path,qk) {r["score_corr_med"]:.3f} | rho(beta,speed) {r["beta_speed_rho_mean"]:+.3f} | '
              f'transport d1/2/4/8/16/31 cos {np.round(r["transport_cos_d"], 2).tolist()} norm {np.round(r["transport_ratio_d"], 2).tolist()}', flush=True)
        print(f'   head beta mean {r["beta_head_mean"]} | head corr(path,qk) {r["score_corr_head"]} | head rho(beta,speed) {r["beta_speed_rho_head"]}', flush=True)
    OUT[arm] = res
json.dump(OUT, open(Path(__file__).with_name('beta_probe.json'), 'w'), indent=1, default=float)
