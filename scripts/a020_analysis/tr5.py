import pandas as pd, numpy as np
p0=pd.read_csv('ate4/ate_A0.2_p0_E128.csv').groupby('env').reward.mean()
for b in ('0.175','0.15','0.1','0.05'):
    d=pd.read_csv(f'ate4/ate_pair0.2:{b}_E128.csv').groupby(['p','env']).reward.mean().unstack(0)
    ate=d[1.0]-p0; dq=(d[0.7]-d[0.3])/0.4; se=lambda x: x.std()/np.sqrt(len(x))
    print(f"B={b} ATE={ate.mean():+.3f}±{se(ate):.3f} ({100*ate.mean()/p0.mean():+.2f}%)  DQ={dq.mean():+.3f}±{se(dq):.3f}")
