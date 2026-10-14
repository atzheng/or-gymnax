import pandas as pd, numpy as np
p0=pd.read_csv('ate4/ate_A0.2_p0_E128.csv').groupby('env').reward.mean()
for b in ('0.26','0.385','0.49'):
    d=pd.read_csv(f'ate4/ate_pair0.2:{b}_E128.csv').groupby(['p','env']).reward.mean().unstack(0)
    ate=d[1.0]-p0; dq=(d[0.7]-d[0.3])/0.4; se=lambda x: x.std()/np.sqrt(len(x))
    print(f"B={b} f(A)={p0.mean():.3f} ATE={ate.mean():+.3f}±{se(ate):.3f} ({100*ate.mean()/p0.mean():+.2f}%)  DQ={dq.mean():+.3f}±{se(dq):.3f}  lam(.3)-lam(0)={(d[0.3]-p0).mean()/.3:+.2f}/unit")
