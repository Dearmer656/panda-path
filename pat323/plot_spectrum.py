"""PAT-323: eigenvalue spectrum (fraction lambda_i / sum) and cumulative explained variance of the fit-set residual covariance,
median and IQR over systems, by system dimension D (3 and 4: the groups with >= 30 systems), one line per lead h.
usage: python pat323/plot_spectrum.py OUTDIR (reads OUTDIR/per_system.json, writes OUTDIR/spectrum.png)"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
res = json.load(open(f'{sys.argv[1]}/per_system.json')); leads = ['1', '16', '64', '128']
col = {'1': '#9ecae1', '16': '#4292c6', '64': '#2171b5', '128': '#08306b'}     # one sequential hue: lead = magnitude
fig, ax = plt.subplots(2, 2, figsize=(9, 6.6), constrained_layout=True)
for r, D in enumerate((3, 4)):
    for h in leads:
        E = np.array([x['eig'] for x in res[h] if x['D'] == D]); n = len(E); idx = np.arange(1, D + 1)
        for c, Y in enumerate((E, np.cumsum(E, 1))):
            med, lo, hi = np.median(Y, 0), np.percentile(Y, 25, 0), np.percentile(Y, 75, 0)
            ax[r, c].plot(idx, med, '-o', color=col[h], lw=2, ms=6, label=f'h = {h}')
            ax[r, c].fill_between(idx, lo, hi, color=col[h], alpha=0.12, lw=0)
    ax[r, 0].plot(idx, np.full(D, 1 / D), '--', color='#888', lw=1.2, label='isotropic 1/D')
    ax[r, 1].plot(idx, idx / D, '--', color='#888', lw=1.2, label='isotropic k/D')
    ax[r, 0].set_title(f'D = {D} ({n} systems): eigenvalue fraction λᵢ/Σλ', fontsize=11, loc='left')
    ax[r, 1].set_title(f'D = {D}: cumulative explained variance', fontsize=11, loc='left')
    for c in (0, 1):
        a = ax[r, c]; a.set_xticks(idx); a.set_xlabel('eigen-direction index'); a.set_ylim(0, 1.02)
        a.grid(axis='y', color='#ddd', lw=0.6); [a.spines[s].set_visible(False) for s in ('top', 'right')]
    ax[r, 0].legend(frameon=False, fontsize=9)
fig.suptitle('PaTH residual covariance on held-out dysts systems (median, IQR band), PAT-323', fontsize=12, x=0.01, ha='left')
fig.savefig(f'{sys.argv[1]}/spectrum.png', dpi=150)
