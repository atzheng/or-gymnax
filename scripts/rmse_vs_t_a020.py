"""RMSE vs. number of steps t for every estimator at A=0.2 (held-out seed 42, 50 envs).

usage: python scripts/rmse_vs_t_a020.py <dir> [out.png] [out.csv]
<dir> holds the same CSVs as scripts/box_a020_below.py. Estimates are saved every
10k steps; the row with steps=s is the estimate after s + 10k steps, so t runs 1e4 .. 5e5.
RMSE is against the true (long-run) ATE.
"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'rmse_vs_t_a020.png'
CSV = sys.argv[3] if len(sys.argv) > 3 else 'rmse_vs_t_a020.csv'
CHUNK = 10000
T = {'0.26': (0.587, 0.49), '0.175': (-0.537, -0.44), '0.15': (-1.225, -1.01), '0.1': (-2.799, -2.31)}
# (file kind, column, label, family)
E = [('fork', 'naive', 'Naive', 'naive'),
     ('tsri_mb1', 'tsri_tsrn', 'TSRN', 'tsr'),
     ('tsri_mb1', 'tsri_lr', 'TSR LR', 'tsr'),
     ('tsri_mb1', 'tsri_tsri1_b0.3679', 'TSRI-1 β=e⁻¹', 'tsr'),
     ('tsri_mb1', 'tsri_tsri2_b0.3679', 'TSRI-2 β=e⁻¹', 'tsr'),
     ('ope', 'ope_lstd_cf_r1e-08', 'OPE-LSTD ridge 1e-8', 'ope'),
     ('ope', 'ope_lstd_cf_cv', 'OPE-LSTD CV', 'ope'),
     ('ope', 'ope_td_cf_sel', 'OPE diff. TD', 'ope'),
     ('fork', 'lstd_paired', 'Pure LSTD (paired)', 'dq'),
     ('fork', 'lstd_tr0.5', 'LSTD-DQ trace λ=.5', 'dq'),
     ('fork', 'lstd_fork_H300_jk', 'Fork-DQ H=300 (jk)', 'dq'),
     ('fork', 'lstd_forkmc_H1000', 'Fork MC H=1000', 'dq'),
     ('fork', 'lstd_forktdL8h0_0.0001_H100', 'Fork-TD 8 modes', 'dq')]
STYLE = {'naive': ('#555555', ['-']), 'tsr': ('#e08a1e', ['-', '--', '-.', ':']),
         'ope': ('#2f78c4', ['-', '--', ':']), 'dq': ('#2e9e3e', ['-', '--', '-.', ':', (0, (5, 1, 1, 1, 1, 1))])}
DQCOL = ['#1b7837', '#d6007a', '#7b3294', '#00a3a3', '#8c510a']
FILES = {'fork': 'fork_pair0.2:{b}_seed42.csv', 'ope': 'ope_pair0.2:{b}.csv',
         'tsri_mb1': 'tsri_pair0.2:{b}_mb1.csv'}
TSHOW = [1e4, 3e4, 1e5, 3e5, 5e5]

rows = []
for b, (ate, pct) in T.items():
    for kind, fn in FILES.items():
        d = pd.read_csv(os.path.join(DATA, fn.format(b=b)))
        d['t'] = d.steps + CHUNK
        for k, c, lab, fam in E:
            if k != kind:
                continue
            g = d.groupby('t')[c]
            for t, x in g:
                x = x.dropna().values
                rows.append(dict(B=b, ate=ate, est=lab, family=fam, t=t, n=len(x),
                                 rmse=np.sqrt(np.mean((x - ate) ** 2)) if len(x) else np.nan,
                                 bias=x.mean() - ate if len(x) else np.nan))
R = pd.DataFrame(rows)
R.to_csv(CSV, index=False)

fig, axs = plt.subplots(1, 4, figsize=(26, 7.5), sharex=True)
for ax, (b, (ate, pct)) in zip(axs, T.items()):
    used = {f: 0 for f in STYLE}
    for _, _, lab, fam in E:
        r = R[(R.B == b) & (R.est == lab)].sort_values('t')
        col, ls = STYLE[fam]
        if fam == 'dq':  # DQ variants get distinct colors so they can be told apart
            col = DQCOL[used[fam] % len(DQCOL)]
        ax.plot(r.t, r.rmse, color=col, ls=ls[used[fam] % len(ls)], lw=1.8 if fam == 'dq' else 1.3,
                marker='o' if lab.startswith('LSTD-DQ') else None, markersize=3, label=lab)
        used[fam] += 1
    ax.set_xscale('log'); ax.set_yscale('log')
    for t in TSHOW:
        ax.axvline(t, color='gray', lw=0.5, ls=':')
    ax.axhline(abs(ate), color='tab:red', lw=1, ls='--')
    ax.text(1.05e4, abs(ate) * 1.08, '|ATE|', color='tab:red', fontsize=8)
    ax.set_title(f'A=0.2, B={b}: ATE {ate:+.2f} ({pct:+.2f}%)')
    ax.set_xlabel('steps t')
    ax.set_xticks(TSHOW); ax.set_xticklabels(['1e4', '3e4', '1e5', '3e5', '5e5'])
    ax.grid(True, which='major', axis='y', alpha=0.3)
axs[0].set_ylabel('RMSE vs true ATE (50 envs)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=7, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.suptitle('RMSE vs steps at A=0.2. Held-out seed 42, 50 envs. Estimates saved every 10k steps (first at t=1e4)')
fig.tight_layout(rect=(0, 0.08, 1, 0.97))
fig.savefig(OUT, dpi=120)

pd.set_option('display.width', 250)
for b in T:
    tab = R[(R.B == b) & R.t.isin(TSHOW)].pivot(index='est', columns='t', values='rmse')
    tab = tab.loc[[lab for _, _, lab, _ in E]]
    tab.columns = [f'{int(c):,}' for c in tab.columns]
    print(f'\nB={b} (ATE {T[b][0]:+.3f}): RMSE'); print(tab.round(2).to_string())
