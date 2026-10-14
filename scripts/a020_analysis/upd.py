p='PROJECT_SUMMARY.md'; s=open(p).read()
notes=open('/tmp/claude-0/-root-or-gymnax/1597dc6d-eb4f-43c5-92f1-4327417a82f0/scratchpad/runlog_notes.md').read()
sec='''### Part 3: A=0.35 settings and fork-DQ (2026-10-09)
See `LSTD_DQ_REPORT.md` Part 3 and `f_curve.png`. In short:
- **Settings.** A=0.35 is the smallest A (0.05 grid) that allows +0.5%, −0.9% and −5% with B near A:
  - B=0.31: ATE +0.58 (+0.48%);
  - B=0.39: ATE −1.05 (−0.87%);
  - B=0.49: ATE −6.02 (−4.98%).

  Truth is in `s3://research/dq/ate3/`. ⚠️ Naive has the right sign in all three.
- **Value-function experiments (offline, dev).** Nothing fixed the long-horizon value gap. Time features, nonlinear counts,
  busy time, fixed effects, a local average reward, LSTD(λ_v) and γ<1 all made it worse or unstable. Spatial resolution
  trades finite-sample bias against feature bias.
- **New estimator.** `xp_gym/estimators/fork_dq_pool.py:ForkDQEstimator` (fork-DQ) replays the other arm's fleet on the
  observed requests and coin flips for H steps, then bootstraps with LSTD. It is verified exactly against brute-force forks
  in `tests/test_fork_dq_pool.py`.
- **Held-out results, fork-DQ H=1000** (pre-registered on dev):

  | Setting | fork-DQ H=1000 | Pure LSTD |
  |---|---|---|
  | B=0.49 | RMSE 2.40, right sign 100% | RMSE 4.88, 88% |
  | B=0.39 | RMSE 0.94, 98% | RMSE 1.21, 82% |
  | B=0.31 | RMSE 1.18, bias +159% (fails the bias goal) | RMSE 1.15, 58% |

  Pure LSTD meets all three goals here, but with a noisy sign. JK and DQ(λ) hurt in every A=0.35 setting.
- Config `scripts/config/fork_expts.yaml`; job specs `vast/fork-a035-*.yaml`, `vast/fork-long.yaml`.

'''
assert '## Run log (from 2026-10-08, LSTD-DQ work)' in s
s=s.replace('## Run log (from 2026-10-08, LSTD-DQ work)', sec+'## Run log (from 2026-10-08, LSTD-DQ work)')
s=s.rstrip('\n')+'\n'+notes
open(p,'w').write(s)
