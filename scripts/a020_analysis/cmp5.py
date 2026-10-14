import pandas as pd, numpy as np
T={'0.175':-0.537,'0.15':-1.225,'0.1':-2.799}
R=[]
for b,ate in T.items():
    for s in (0,42):
        d=pd.read_csv(f'forkall/fork_pair0.2:{b}_seed{s}.csv'); d=d[d.steps==d.steps.max()]
        for c in d.columns:
            if c in('env_id','steps','trial') or 'fkdiag' in c: continue
            x=d[c].values; m=x.mean(); sd=x.std(ddof=1)
            R.append(dict(B=b,seed=s,est=c,bias=100*(m-ate)/abs(ate),rmse=np.sqrt((m-ate)**2+sd**2),sign=np.mean(np.sign(x)==np.sign(ate))))
R=pd.DataFrame(R)
dev=R[R.seed==0].groupby('est').agg(maxbias=('bias',lambda v:np.abs(v).max()),minsign=('sign','min'),meanrmse=('rmse','mean')).sort_values('maxbias')
print("DEV (seed 0) ranking by worst |bias| across the 3 settings"); print(dev.head(15).round(2).to_string())
def cell(r): return f"{r.rmse:5.2f} / {r.bias:+5.0f}% / {100*r.sign:3.0f}%"
pick=['naive','lstd_paired','lstd_tr0.5','lstd_tr0.8','lstd_paired_jk','lstd_fork_H300','lstd_fork_H1000','lstd_forkmc_H1000','lstd_forktdL8h0_0.0001_H100','lstd_forktdL16h0_0.0001_H100']+list(dev.index[:5])
pick=list(dict.fromkeys(pick))
for s in (42,0):
    print(f"\nseed {s}: rmse / bias / right sign")
    t=R[R.seed==s].set_index(['est','B'])
    for e in pick: print(f"{e:32s}"+"  ".join(cell(t.loc[(e,b)]) for b in T))
