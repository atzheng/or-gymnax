import pandas as pd, glob, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = pd.read_csv("f_curve.csv").set_index("s").f.to_dict()
for f in glob.glob("fscan/f_s*_E128.csv"):
    s = float(re.search(r"f_s([\d.]+)_", f).group(1)); c[s] = pd.read_csv(f).groupby("env").reward.mean().mean()
s = np.array(sorted(c)); f = np.array([c[x] for x in s])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, lim in zip(ax, [(0, 0.8), (0.1, 0.55)]):
    m = (s >= lim[0]) & (s <= lim[1]); a.plot(s[m], f[m], "o-", ms=3)
    a.axvline(0.2, color="k", ls="--", lw=1); a.text(0.2, f[m].min(), " A=0.2", va="bottom")
    for b, lab in [(0.26, "+0.49%"), (0.385, "−0.87%"), (0.49, "−5.05%")]:
        a.axvline(b, color="C3", ls=":", lw=1); a.text(b, f[m].max(), f" {lab}", color="C3", va="top", fontsize=8)
    a.set_xlabel("savings threshold s"); a.set_ylabel("average reward f(s)")
ax[1].set_title("zoom: A on the rising side of the peak")
plt.tight_layout(); plt.savefig("/root/or-gymnax/f_curve_a020.png", dpi=120)
