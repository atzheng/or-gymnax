# LSTD-based DQ on the pooled rideshare env: what was tried and what worked

*2026-10-08. Code: xp_gym branch `ghosts` (commits 05154aa, f15a85e). The run-by-run config log
is in `PROJECT_SUMMARY.md`.*

## Goal

Build an LSTD version of the DQ estimator that does about as well as DN-ghost (RMSE ≈ 6 at
B=0.1) or better, and that holds up across the savings-threshold sweep. The paper needs three
things from it:
1. RMSE far below Naive.
2. |bias| below 100% of |ATE|.
3. The right sign where Naive gets it wrong.

Pure LSTD with engineered features was preferred. An interpolation such as n-step or LSTD(λ)
was the acceptable fallback.

## Summary

- **DQ(λ=0.9) beats DN-ghost at every threshold and meets all three goals.** It uses an LSTD
  value function plus a short eligibility trace of TD errors. At B=0.1 its RMSE is 1.8 (DN-ghost
  7.7, Naive 48.6), its bias is +1%, and it has the right sign in 100% of envs. At B=0.2 and 0.3
  its RMSE is about a third of DN-ghost's.
- **Pure LSTD-DQ only partly works.** On its own it is biased below the true effect and usually
  has the wrong sign. With a block-jackknife correction for finite-sample bias, it beats
  DN-ghost on RMSE at all three thresholds. It still recovers only about
  half the effect.
- **DQ's target is not the problem.** Average reward is almost linear in the treatment share, so
  DQ's estimand is close to the true ATE. The shortfalls in DN-ghost and pure LSTD come from
  estimation error, not from the estimand.

## Setup

- **Env:** pooled rideshare env, 300 cars, `max_active_trips=2`, 500k requests per run. The
  treatment is the pooling savings threshold B. Arm A is p=0 and arm B is p=1; the experiment
  randomizes per request with p=0.5.
- **Truth:** ATE per threshold from the 1000-run ATE table (B=0.1) or from the λ(p) sweep
  endpoints (B=0.2, 0.3).
- **Dev / test split:** all feature and hyperparameter choices were made on offline datasets
  from seed 0 (8 envs per threshold). The held-out evaluation is seed 42, with 50 envs
  (25 × 2 trials) per threshold. Ghost cars don't change trajectories for a fixed seed (I
  checked), so the LSTD runs are paired env-by-env with the DN-ghost runs.

## What I tried, in order

### 1. Checking that DQ's target is the ATE (λ(p) sweep)

DQ estimates λ′(0.5), the slope of average reward λ(p) with respect to the treatment share p.
If λ(p) were strongly curved, even a perfect DQ estimator would miss the ATE. I measured λ(p)
for p ∈ {0, .1, .3, .4, .5, .6, .7, .9, 1} with common random numbers, 64 envs each
(`scripts/lambda_p.py`).

| B | ATE | λ′(0.5) | Naive | P(arms differ) |
|---|---|---|---|---|
| 0.1 | 4.43 | 4.65 ± 0.14 | −44 | 9.9% |
| 0.2 | 7.23 | 6.73 ± 0.15 | −85 | 18.7% |
| 0.3 | 7.76 | 7.03 ± 0.17 | −121 | 26.3% |
| 0.5 | 0.37 | 1.67 ± 0.15 | −180 | 38.2% |

λ(p) is close to linear, so DQ's target is within about 10% of the ATE. DN-ghost recovering only
about a third of the effect at B=0.1 is therefore truncation error. I dropped B=0.5 from the
evaluation: its ATE is about 0, so it can't test sign recovery.

### 2. Understanding the mechanism (forked-rollout oracle)

To see what the value function has to capture, I forked trajectories at the decision points
where the arms differ (128 envs × 24 forks; `scripts/fork_oracle.py`).

- **The fleet is saturated:** about 0 idle cars, and 219 of 300 cars are full.
- **The arms only differ in one way.** In every differing step, A pools the request into a car
  that already has one trip, earning about 475. B either rejects it (about 54% of the time,
  reward 0) or sends a distant empty car (reward about 57).
- **The cost is paid back slowly.** The cumulative B−A reward gap is −302 after 10 requests,
  −133 after 100, −53 after 300, and +37 ± 36 after 1000. A trip lasts about 275 requests.

So the immediate gap is about −449 per differing step and the value gap about +490. Naive sees
only the first number, which is why its sign is always wrong. A value-based estimator needs V
accurate to within a few percent of a ~490 swing, mostly realized within a few hundred steps.

### 3. The estimator

I wrote `PoolLSTDDQEstimator` (`xp_gym/estimators/lstd_dq_pool.py`), an online estimator.

- **The value function.** It fits a post-decision fleet value U(y)=θᵀφ(y) by average-reward
  LSTD. Settings: γ=1, ridge 1e-6 on standardized features, solved in float64. It fits on the
  experiment's own trajectory.
- **Paired DQ** (`paired`). On steps where the arms differ, it rebuilds the other arm's
  post-decision fleet and reward from the known actions. It then estimates the advantage as the
  reward difference plus the value difference.
- **DQ(λ)** (`tr{λ}`). An IPW-weighted generalized-advantage estimate: Σ δ_t e_t / T, where δ_t
  is the TD error and e_t = λ e_{t−1} + IPW weight. λ=0 is roughly paired DQ, and λ→1 is
  truncated Monte Carlo DQ.
- **Jackknife** (`*_jk`). A delete-one-block correction over 5 blocks of 100k steps, aimed at
  LSTD's finite-sample bias.

`tests/test_lstd_dq_pool.py` checks every online accumulator against an independent numpy replay.

### 4. Feature search (pure LSTD, dev data)

Features are in `xp_gym/estimators/pool_features.py`. Lessons:

| Tried | Result |
|---|---|
| **Zone counts**: cars by number of active trips; idle cars by zone; 1-trip cars by final drop-off zone; full cars by first drop-off zone (63 taxi zones, 192 dims) | **Best.** Kept. |
| + busy-time histograms (time until a car frees up) | Worse |
| + drop-offs by zone × time bucket | Worse |
| + "match features": reward the greedy policy would earn for 128 fixed reference requests, now and 120 s / 300 s ahead (up to 1510 dims) | Worse |
| Coarser zones (k-means to 8/16/32 regions) | Smaller finite-sample bias, but a wash after the jackknife and an overshoot at B=0.3. Not kept. |
| LSTD(λ_v) with λ_v → 1, and Monte Carlo regression on truncated returns | Monotonically worse than plain TD |
| γ = 0.999 | Worse than γ=1 |
| Larger ridge | Worse |

Richer features making things worse is **projection bias, not noise**. Fitting θ on all 8 dev
envs pooled (4M transitions) gives the same ranking, and with pooled θ the spread of paired DQ
across envs is only about 0.15. Pure LSTD's bias at B=0.1 splits into about −2.5 finite-sample
bias (which the jackknife mostly removes) and about −2 projection bias that more data doesn't fix.

### 5. The fallback that worked: DQ(λ)

The trace adds the observed TD errors of roughly the next 1/(1−λ) steps to each one-step
advantage. That corrects the short-horizon part of the value gap, which is exactly where linear
features are weakest. Because LSTD keeps the TD errors small, the variance stays far below
truncated MC. Without a value function (U=0), the trace needs λ≈0.999 and has SD ≈ 16. λ=0.9
(about 10 steps) was picked on dev seed 0.

## Held-out results (seed 42)

The LSTD and Naive rows use 50 envs. DN-ghost at B=0.1 is the existing 2500-group, 50-env run.

| B (ATE) | Estimator | Mean ± SD | Bias % [95% CI] | RMSE | Sign OK |
|---|---|---|---|---|---|
| **0.1** (4.43) | Naive | −44.18 ± 0.41 | −1097 | 48.6 | 0% |
| | DN-ghost | 0.51 ± 6.63 | −89 [−130, −47] | 7.7 | 48% |
| | Pure LSTD-DQ | −1.33 ± 2.03 | −130 [−143, −117] | 6.1 | 32% |
| | Pure LSTD-DQ + jackknife | 1.84 ± 2.50 | −58 [−74, −43] | 3.6 | 82% |
| | DQ(λ=0.8) | 3.39 ± 1.74 | −24 [−34, −13] | 2.0 | 98% |
| | **DQ(λ=0.9)** | **4.50 ± 1.79** | **+1 [−10, 13]** | **1.8** | **100%** |
| **0.2** (7.23) | Naive | −84.62 ± 0.54 | −1271 | 91.8 | 0% |
| | DN-ghost† | 0.84 ± 10.59 | −88 [−146, −31] | 12.4 | 44% |
| | Pure LSTD-DQ | −2.60 ± 4.09 | −136 | 10.6 | 24% |
| | Pure LSTD-DQ + jackknife | 3.72 ± 4.75 | −48 [−67, −30] | 5.9 | 78% |
| | **DQ(λ=0.9)** | **7.81 ± 3.30** | **+8 [−5, 21]** | **3.4** | **98%** |
| **0.3** (7.76) | Naive | −120.81 ± 0.56 | −1657 | 128.6 | 0% |
| | DN-ghost† | 5.85 ± 12.71 | −25 [−89, 40] | 12.9 | 60% |
| | Pure LSTD-DQ | −5.38 ± 6.08 | −169 | 14.5 | 26% |
| | Pure LSTD-DQ + jackknife | 3.79 ± 6.90 | −51 [−76, −26] | 8.0 | 66% |
| | **DQ(λ=0.9)** | **9.62 ± 4.55** | **+24 [8, 40]** | **4.9** | **98%** |

† At B=0.2 and 0.3, DN-ghost used a cheaper config: 1000 ghost groups and 25 envs (trial 0 only).
The full config ran at about 7 steps/s, about 40 GPU-hours per run. On those same 25 envs,
DQ(λ=0.9) gets RMSE 3.3 (B=0.2) and 4.8 (B=0.3), with the right sign in 100% of envs both times.

**Sensitivity to λ** (25 trial-0 envs):

| B | λ=0.8 | λ=0.9 | λ=0.95 |
|---|---|---|---|
| 0.2 | 5.23, RMSE 3.9 | 7.67, RMSE 3.3 | 8.67, RMSE 3.8 |
| 0.3 | 5.69, RMSE 5.7 | 9.04, RMSE 4.8 | 10.57, RMSE 5.0 |

Higher λ moves the estimate up. Bias goes from negative at 0.8 to positive at 0.95, while RMSE
is fairly flat between 0.8 and 0.95.

**RMSE versus experiment length** at B=0.1:

| Steps | Naive | DN-ghost | Pure LSTD | Pure LSTD + JK | DQ(λ=0.9) |
|---|---|---|---|---|---|
| 50k | 48.9 | 21.5 | 26.8 | 26.8 | 12.5 |
| 100k | 48.7 | 16.5 | 15.3 | 15.3 | 7.5 |
| 300k | 48.6 | 9.9 | 7.8 | 5.0 | 3.1 |
| 500k | 48.6 | 7.7 | 6.1 | 3.6 | 1.8 |

DQ(λ=0.9) is ahead of DN-ghost at every horizon.

## Scorecard against the goals

| Goal | DQ(λ=0.9) | Pure LSTD + JK | DN-ghost |
|---|---|---|---|
| RMSE far below Naive | ✅ all B | ✅ all B | ✅ all B |
| \|bias\| < 100% of \|ATE\| | ✅ (+1%, +8%, +24%) | ✅ (−58%, −48%, −51%) | ✅ but borderline (−89%, −88%, −25%) |
| Right sign where Naive is wrong | ✅ 98–100% | ⚠️ 66–82% | ❌ 44–60% |
| Beats DN-ghost on RMSE | ✅ about 3–4× lower | ✅ all B (3.6 vs 7.7; 6.6 vs 12.4; 9.0 vs 12.9 on matched envs) | — |

## Open questions

- **Which method leads the paper:** DQ(λ), which is best, or pure LSTD + jackknife, which is
  closer to the original pitch but recovers only about half the effect.
- **Choosing λ in a principled way.** λ=0.9 was picked on dev seed 0 and held up on seed 42.
  At B=0.3, though, the bias is +24% and significant.
- **Pure LSTD's projection bias** needs better features. Untried ideas: per-car attribute bases
  (zone × trip count × busy bucket), and features for the route of a pooled car's second trip.
- **Side issue:** `xp_gym/designs/design.py:load_rideshare_clusters` builds its node→zone map in
  the wrong order. Env node i is row i of `taxi-zones.parquet`, but that function uses
  `manhattan-nodes` row order, so spatial cluster designs look scrambled. They aren't used here.

## Where things are

- Estimator: `xp_gym/estimators/lstd_dq_pool.py`; features: `xp_gym/estimators/pool_features.py`;
  test: `tests/test_lstd_dq_pool.py`.
- Config and job spec: `scripts/config/lstd_expts.yaml`, `vast/lstd.yaml`.
- Summary script: `scripts/summarize_lstd.py <B> <csv>...`.
- Diagnostics: `scripts/lambda_p.py`, `scripts/fork_oracle.py`, `scripts/collect_pool.py`.
- Results on R2:
  - `s3://research/dq/lstd/silky-stoic-woodlouse-of-wizardry/`
  - DN-ghost: `smooth-aquatic-mastiff` (B=0.1), `prophetic-victorious-bullfrog-from-pluto`
    (B=0.2), `innocent-cuddly-raven-of-enrichment` (B=0.3)
  - Diagnostics and dev data: `dq/lambda/`, `dq/fork/`, `dq/collect2/`
