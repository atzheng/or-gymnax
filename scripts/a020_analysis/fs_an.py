import pandas as pd, glob, re, numpy as np
F={}
for f in glob.glob('fscan/f_s*_E128.csv'):
    s=float(re.search(r'f_s([\d.]+)_',f).group(1)); F[s]=pd.read_csv(f).groupby('env').reward.mean()
for A in (0.15,0.175,0.2,0.21):
    print(f"A={A} f={F[A].mean():.3f}")
    for s in sorted(F):
        d=F[s]-F[A]; print(f"  {s:.3f} f={F[s].mean():8.3f}  ATE={d.mean():+7.3f}±{d.std()/np.sqrt(len(d)):.3f}  {100*d.mean()/F[A].mean():+6.2f}%")
