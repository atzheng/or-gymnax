"""Boxplots of per-env ATE estimates for A=0.2: B=0.26 (+0.49%) and B below A (held-out seed 42, 50 envs).

usage: python scripts/box_a020_below.py <dir>
<dir> holds local copies of
  s3://research/dq/forkall/<job>/fork_pair0.2:{B}_seed42.csv   (Naive, LSTD-DQ, fork variants)
    For B=0.26: the fork/<job>/ seed-42 CSV with the lstd_forktd* columns of forktd3/<job>/ merged in on (env_id, trial)
  s3://research/dq/ope/<job>/ope_pair0.2:{B}.csv               (OPE)
  s3://research/dq/tsri/<job>/tsri_pair0.2:{B}_{ac0.5_al0.5,mb1}.csv  (TSR)
Top row: full scale. Bottom row: zoomed to the true ATE.
"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'box_a020_below.png'
T = {'0.26': (0.587, 0.49), '0.175': (-0.537, -0.44), '0.15': (-1.225, -1.01), '0.1': (-2.799, -2.31)}
# (file suffix, column, label, group)
E = [('fork', 'naive', 'Naive', 'naive'),
     ('tsri_mb1', 'tsri_tsrn', 'TSRN', 'tsr'),
     ('tsri_mb1', 'tsri_lr', 'TSR LR', 'tsr'),
     ('tsri_mb1', 'tsri_tsri1_b0.3679', 'TSRI-1\nβ=e⁻¹', 'tsr'),
     ('tsri_mb1', 'tsri_tsri2_b0.3679', 'TSRI-2\nβ=e⁻¹', 'tsr'),
     ('ope', 'ope_lstd_cf_r1e-08', 'OPE-LSTD\nridge 1e-8', 'ope'),
     ('ope', 'ope_lstd_cf_cv', 'OPE-LSTD\nCV', 'ope'),
     ('ope', 'ope_td_cf_sel', 'OPE diff.\nTD', 'ope'),
     ('fork', 'lstd_paired', 'Pure LSTD\n(paired)', 'dq'),
     ('fork', 'lstd_tr0.5', 'LSTD-DQ\ntrace λ=.5', 'dq'),
     ('fork', 'lstd_fork_H300_jk', 'Fork-DQ\nH=300 (jk)', 'dq'),
     ('fork', 'lstd_forkmc_H1000', 'Fork MC\nH=1000', 'dq'),
     ('fork', 'lstd_forktdL8h0_0.0001_H100', 'Fork-TD\n8 modes', 'dq')]
COL = {'naive': '#bbbbbb', 'tsr': '#f2c48d', 'ope': '#a6c8e8', 'dq': '#b5dfa8'}
FILES = {'fork': 'fork_pair0.2:{b}_seed42.csv', 'ope': 'ope_pair0.2:{b}.csv',
         'tsri_mb1': 'tsri_pair0.2:{b}_mb1.csv'}


def load(kind, b):
    f = os.path.join(DATA, FILES[kind].format(b=b))
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d[d.steps == d.steps.max()]


fig, axs = plt.subplots(2, 4, figsize=(28, 11), height_ratios=[1, 1.3])
for j, (b, (ate, pct)) in enumerate(T.items()):
    data = {k: load(k, b) for k in FILES}
    E_b = [e for e in E if data[e[0]] is not None]
    vals = [data[k][c].values for k, c, _, _ in E_b]
    for i, ax in enumerate(axs[:, j]):
        bp = ax.boxplot(vals, tick_labels=[l for _, _, l, _ in E_b], showmeans=True, patch_artist=True,
                        meanprops=dict(marker='D', markerfacecolor='k', markeredgecolor='k', markersize=4),
                        medianprops=dict(color='k'), flierprops=dict(markersize=3))
        for p, (_, _, _, g) in zip(bp['boxes'], E_b):
            p.set_facecolor(COL[g])
        ax.axhline(ate, color='tab:red', lw=1.5, label=f'true ATE {ate:+.2f}')
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.tick_params(axis='x', labelsize=7, rotation=50)
        if i == 0:
            ax.set_title(f'A=0.2, B={b}: ATE {ate:+.2f} ({pct:+.2f}%)')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.set_ylim(-4 * abs(ate) - 1.5, 4 * abs(ate) + 1.5)
            ax.set_title('zoomed', fontsize=9)
axs[0, 0].set_ylabel('estimate, full scale')
axs[1, 0].set_ylabel('estimate, zoomed')
handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[g]) for g in COL]
fig.legend(handles, ['Naive', 'TSR (market-balance design)', 'OPE', 'DQ variants'], loc='upper center',
           ncol=4, bbox_to_anchor=(0.5, 0.965))
fig.suptitle('ATE estimates at A=0.2: B=0.26 (above A) and B below A. Held-out seed 42, 50 envs (◆ = mean)', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT, dpi=120)
