import sys, numpy as np, pandas as pd
f, ate = sys.argv[1], float(sys.argv[2])
d = pd.read_csv(f); last = d[d.steps == d.steps.max()]
cols = [c for c in last.columns if c not in ("env_id", "steps", "trial") and "fkdiag" not in c]
rows = []
for c in cols:
    x = last[c].values; m, s = x.mean(), x.std(ddof=1); se = s / np.sqrt(len(x))
    rows.append(dict(est=c, mean=m, sd=s, bias_pct=100 * (m - ate) / abs(ate),
                     ci=f"[{100*(m-1.96*se-ate)/abs(ate):.0f}, {100*(m+1.96*se-ate)/abs(ate):.0f}]",
                     rmse=np.sqrt((m - ate) ** 2 + s ** 2), sign=np.mean(np.sign(x) == np.sign(ate))))
r = pd.DataFrame(rows).sort_values("rmse")
pd.set_option("display.width", 200)
print(f"n={len(last)} ATE={ate}"); print(r.round(2).to_string(index=False))
diag = [c for c in last.columns if "fkdiag" in c]
if diag: print(last[diag].mean().round(3).to_string())
