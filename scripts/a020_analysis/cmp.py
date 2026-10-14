import sys, pandas as pd, numpy as np
d, seed = sys.argv[1], sys.argv[2]
T = {"0.26": 0.587, "0.385": -1.057, "0.49": -6.108}
rows = {}
for b, ate in T.items():
    x = pd.read_csv(f"{d}/fork_pair0.2:{b}_seed{seed}.csv"); x = x[x.steps == x.steps.max()]
    for c in x.columns:
        if c in ("env_id", "steps", "trial") or "fkdiag" in c: continue
        v = x[c].values; m, s = v.mean(), v.std(ddof=1)
        rows.setdefault(c, {})[b] = (m, s, 100 * (m - ate) / abs(ate), np.sqrt((m - ate) ** 2 + s ** 2), np.mean(np.sign(v) == np.sign(ate)))
out = []
for c, r in rows.items():
    o = {"est": c}
    for b in T:
        m, s, bp, rm, sg = r[b]; o[f"{b}"] = f"{m:6.2f}±{s:4.2f} {bp:+5.0f}% r{rm:5.2f} s{sg:.2f}"
    o["maxbias"] = max(abs(r[b][2]) for b in T); o["sumrmse"] = sum(r[b][3] for b in T)
    o["relrmse"] = sum(r[b][3] / abs(T[b]) for b in T)
    out.append(o)
o = pd.DataFrame(out).sort_values(sys.argv[3] if len(sys.argv) > 3 else "relrmse")
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 40)
print(o.head(int(sys.argv[4]) if len(sys.argv) > 4 else 30).to_string(index=False))
