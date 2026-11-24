"""PAT-335: association (not causation) between per-system latent commutators (pat336.props *_comm.csv) and per-system
forecast error (median sMAPE over that system's test samples, official eval dir of the SAME checkpoint and ctx).
Reports Spearman across systems per layer and for the layer mean, for comm_n (adjacent) and comm_n - comm_pn
(adjacent minus time-permuted control). usage: python -m pat336.assoc COMM_CSV CKPT CTX EVAL_DIR [L=128]"""
import sys, ast, numpy as np, pandas as pd
from scipy.stats import spearmanr
cf, ck, ctx, ed = sys.argv[1:5]; L = int(sys.argv[5]) if len(sys.argv) > 5 else 128
df = pd.read_csv(cf); df = df[(df.ckpt == ck) & (df.ctx == int(ctx))]
ev = pd.read_csv(f'{ed}/metrics_pred{L}.csv'); sm = {r.System: float(np.median(ast.literal_eval(r.smape))) for r in ev.itertuples()}
df = df[df.system.isin(sm)]; print(f'{ck} ctx{ctx} L{L}: {df.system.nunique()} systems matched')
def rep(g, tag):
    a = g.groupby('system')[['comm_n', 'comm_pn']].mean(); y = [sm[s] for s in a.index]
    r1, p1 = spearmanr(a.comm_n, y); r2, p2 = spearmanr(a.comm_n - a.comm_pn, y)
    print(f'  {tag}: adj-perm mean {float((a.comm_n - a.comm_pn).mean()):+.4f} (share systems adj<perm {float((a.comm_n < a.comm_pn).mean()):.2f}); '
          f'Spearman(comm_n, sMAPE) {r1:+.2f} (p {p1:.3f}); Spearman(adj-perm, sMAPE) {r2:+.2f} (p {p2:.3f})')
for l, g in df.groupby('layer'): rep(g, f'L{l}')
rep(df, 'layer-mean')
