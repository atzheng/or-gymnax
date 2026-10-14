"""Paper protocol for hyperparameters: pick the value minimizing MSE vs the true ATE on 10
trajectories, then evaluate on the remaining 40 (held-out seed 42, 50 envs = 25 envs x 2 trials).

usage: python tune10.py <dir> <prefix> <B:ATE,...> <col1,col2,...> [n_tune=10]
Columns are the candidate hyperparameter values of one estimator family.
Tuning trajectories: the first n_tune (trial, env_id) pairs in sorted order.
"""
import sys, os, numpy as np, pandas as pd

d, prefix, settings, cols = sys.argv[1:5]
n_tune = int(sys.argv[5]) if len(sys.argv) > 5 else 10
cols = cols.split(',')
rows = []
for kv in settings.split(','):
    b, ate = kv.split(':'); ate = float(ate)
    x = pd.read_csv(os.path.join(d, prefix.format(b=b)))
    x = x[x.steps == x.steps.max()].sort_values(['trial', 'env_id']).reset_index(drop=True)
    tune, ev = x.iloc[:n_tune], x.iloc[n_tune:]
    mse = {c: np.mean((tune[c] - ate) ** 2) for c in cols}
    pick = min(mse, key=mse.get)
    for c in cols:
        e = ev[c]
        rows.append(dict(B=b, ate=ate, col=c, tune_rmse=np.sqrt(mse[c]), picked=c == pick,
                         eval_rmse=np.sqrt(np.mean((e - ate) ** 2)), eval_bias_pct=100 * (e.mean() - ate) / abs(ate),
                         eval_sign=np.mean(np.sign(e) == np.sign(ate)), naive_rmse=np.sqrt(np.mean((ev.naive - ate) ** 2)) if 'naive' in ev else np.nan))
R = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print(R.round(3).to_string(index=False))
# single value across settings: minimize mean over settings of tuning MSE / ATE^2 (scale-free)
R['tune_rel'] = (R.tune_rmse / R.ate.abs()) ** 2
g = R.groupby('col').tune_rel.mean().sort_values()
print('\nacross-settings pick (mean relative tuning MSE):'); print(g.round(3).to_string())
