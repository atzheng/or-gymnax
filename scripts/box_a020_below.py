import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys
DATA = sys.argv[1] if len(sys.argv) > 1 else 'forkall'  # local copy of s3://research/dq/forkall/<job>/ CSVs
T={'0.175':(-0.537,-0.44),'0.15':(-1.225,-1.01),'0.1':(-2.799,-2.31)}
E={'lstd_paired':'Pure LSTD\n(paired)','lstd_tr0.5':'LSTD-DQ\ntrace λ=0.5','lstd_fork_H300_jk':'Fork-DQ\nH=300 (jk)',
   'lstd_forkmc_H1000':'Fork MC\nH=1000','lstd_forktdL8h0_0.0001_H100':'Fork-TD\n8 modes'}
fig,axs=plt.subplots(1,3,figsize=(16,5.5))
for ax,(b,(ate,pct)) in zip(axs,T.items()):
    d=pd.read_csv(f'{DATA}/fork_pair0.2:{b}_seed42.csv'); d=d[d.steps==d.steps.max()]
    ax.boxplot([d[c] for c in E],tick_labels=list(E.values()),showmeans=True,meanprops=dict(marker='D',markerfacecolor='k',markeredgecolor='k',markersize=5))
    ax.axhline(ate,color='tab:red',lw=1.5,label=f'true ATE {ate:+.2f}'); ax.axhline(0,color='gray',ls=':',lw=1)
    ax.set_title(f'A=0.2, B={b}: ATE {ate:+.2f} ({pct:+.2f}%)\nNaive = {d.naive.mean():+.1f} (off scale)')
    ax.tick_params(axis='x',labelsize=8); ax.legend(loc='upper left',fontsize=8)
axs[0].set_ylabel('estimate (reward / request)')
fig.suptitle('DQ estimates, B below A — held-out seed 42, 50 envs (◆ = mean)')
fig.tight_layout(); fig.savefig('/root/or-gymnax/box_a020_below.png',dpi=130)
