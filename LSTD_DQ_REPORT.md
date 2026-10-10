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

---

# Part 2: small ATEs (paper levels −5%, −0.9%, +0.5% of average reward)

*Run configs are logged in `PROJECT_SUMMARY.md`. The truth for each setting comes from CRN λ(p) sweeps:
64 envs, plus 128–256 envs at the smallest effects. Every estimator run uses seed 42, 50 envs and 500k steps.*

## Finding settings with small ATEs

With arm A fixed at threshold 0, the ATE as a function of B (see `ate_vs_B.png`) rises to about +6.8% at
B=0.3, crosses zero at B≈0.50 and falls steeply after that. This gives three ways to reach the paper's levels:

1. **B near 0.5 (A=0).** B=0.497, 0.518 and 0.571 give +0.6%, −0.76% and −4.9%. Average reward
   λ(p) is **strongly convex** here: almost all the drop happens between p=0.7 and p=1. As a result, DQ's
   own estimand λ′(0.5) is far from the ATE (+1.75 vs +0.69, +0.85 vs −0.86, −1.82 vs −5.53). Even a
   perfect DQ estimator would be badly biased, and at −0.76% it would have the wrong sign.
2. **B near 0 (A=0).** B=0.005, 0.01 and 0.02 give +0.15%, +0.40% and +0.82%. λ(p) is linear, so DQ's
   estimand equals the ATE. Only about 1–2% of decisions differ between arms. This regime only gives
   positive ATEs.
3. **Threshold pairs, both nonzero.** A:B = 0.3:0.35, 0.4:0.45, 0.4:0.5 and 0.35:0.5 give −0.53%, −1.84%,
   −4.43% and −5.55%. λ(p) is linear for all of them (DQ estimand within ±0.2 of the ATE). This is the
   clean way to get negative small ATEs.

## Results

Each cell is mean ± SD / RMSE / % of envs with the right sign. "Pure" is λ=0 paired LSTD-DQ, JK is the
block jackknife, and tr is DQ(λ). The % column is relative to λ(0) of arm A.

| A:B | ATE (%) | DQ estimand | Naive | Pure LSTD | Pure + JK | DQ(λ=0.9) | DQ(λ=0.8) + JK |
|---|---|---|---|---|---|---|---|
| 0:0.005 | 0.17 (+0.15%) | 0.24 | −1.71±0.08 / 1.88 / 0% | −0.20±0.09 / 0.38 / 0% | −0.09±0.11 / 0.28 / 24% | 0.00±0.30 / 0.34 / 52% | 0.07±0.25 / 0.27 / 64% |
| 0:0.01 | 0.46 (+0.40%) | 0.36 | −3.96±0.12 / 4.41 / 0% | −0.29±0.22 / 0.78 / 10% | −0.04±0.27 / 0.56 / 48% | 0.27±0.54 / 0.57 / 66% | 0.35±0.45 / 0.46 / 74% |
| 0:0.02 | 0.93 (+0.82%) | 0.94 | −8.59±0.16 / 9.51 / 0% | −0.56±0.45 / 1.55 / 10% | 0.02±0.49 / 1.03 / 54% | 0.73±0.89 / 0.91 / 84% | 0.96±0.78 / 0.78 / 90% |
| 0:0.1 | 4.43 (+3.90%) | 4.65 | −44.18±0.41 / 48.61 / 0% | −1.33±2.03 / 6.11 / 32% | 1.84±2.50 / 3.60 / 82% | **4.50±1.79 / 1.79 / 100%** | 6.07±2.09 / 2.66 / 100% |
| 0.3:0.35 | −0.65 (−0.53%) | −0.65 | −22.49±0.28 / 21.84 / 100% | **−0.48±1.43 / 1.44 / 58%** | 1.46±1.60 / 2.65 / 20% | 1.86±1.47 / 2.91 / 12% | 2.97±1.62 / 3.96 / 4% |
| 0.4:0.45 | −2.20 (−1.84%) | −2.29 | −23.23±0.24 / 21.03 / 100% | **−1.75±1.47 / 1.54 / 86%** | 0.13±1.66 / 2.86 / 48% | 0.97±1.83 / 3.66 / 24% | 2.04±1.85 / 4.62 / 16% |
| 0.4:0.5 | −5.29 (−4.43%) | −5.18 | −46.87±0.36 / 41.59 / 100% | **−4.55±2.84 / 2.94 / 94%** | −0.62±3.22 / 5.67 / 66% | 1.31±2.96 / 7.23 / 36% | 3.43±3.24 / 9.30 / 12% |
| 0.35:0.5 | −6.70 (−5.55%) | −6.86 | −68.59±0.40 / 61.89 / 100% | **−5.48±4.28 / 4.45 / 90%** | 0.65±5.25 / 9.03 / 44% | 2.60±4.03 / 10.14 / 32% | 6.29±4.75 / 13.83 / 10% |
| 0:0.497 | 0.69 (+0.60%) | 1.75 | −179.32±0.77 / 180.0 / 0% | −17.75±9.85 / 20.90 / 6% | −5.44±11.63 / 13.14 / 30% | 5.30±8.74 / 9.88 / 72% | 10.51±10.80 / 14.60 / 82% |
| 0:0.518 | −0.86 (−0.76%) | 0.85 | −184.59±0.61 / 183.7 / 100% | −18.22±9.49 / 19.78 / 96% | −4.88±11.26 / 11.96 / 70% | 5.24±8.54 / 10.49 / 26% | 11.30±10.42 / 16.02 / 12% |
| 0:0.571 | −5.53 (−4.86%) | −1.82 | −197.03±0.76 / 191.5 / 100% | −20.11±11.04 / 18.29 / 98% | −5.58±13.48 / 13.48 / 66% | 4.98±9.69 / 14.29 / 30% | 11.75±12.08 / 21.08 / 18% |

## What it shows

- **No single LSTD variant works in every regime.**
  - With A=0 and small B, DQ(λ) works. It meets the bias goal everywhere except the very smallest
    effect, and it gets the right sign 66–100% of the time, limited by noise.
  - With both thresholds nonzero, **pure LSTD** works: bias +14% to +27%, RMSE 14–33× below Naive,
    right sign 58–94%. DQ(λ) and the jackknife push it the wrong way.
  - Near B=0.5, nothing works. Naive is about −190, i.e. 30–250× the ATE, and DQ's estimand is biased
    by the curvature.
- **Why DQ(λ) isn't a reliable fix.** The gap between DQ(λ=0.9) and pure LSTD is about −0.12×Naive in
  every regime (+5.8 at 0:0.1; +2.3, +2.7 and +8.1 for the pairs). As λ→1 the estimate levels off
  near the truth when A=0, but at **+2.5 to +3.7** for the pairs, where the truth is −5 to −7.
  - The trace replaces the LSTD value only over about 1/(1−λ) ≤ 30 requests. The fork oracle shows the
    payback from a pooling decision takes hundreds to ~1000 requests, so the long tail still comes from
    the LSTD value function.
  - That value function's error has a different sign in each regime. At A=0 the trace happened to
    cancel it; for the pairs it adds to it.
  - The bottleneck is still the **value function's long-horizon accuracy**. DQ(λ) gave the earlier
    B=0.1–0.3 results partly by cancelling errors.
- **TSRI** (Johari et al. 2020; job simple-expert-stallion-of-acceptance) fails at B=0.1, 0.2 and 0.3.
  This holds with a_C=a_L=0.5 and with the paper's market-balance design, for every β.
  - The closest variant (LR, market-balance design) is −10.5±13.7 at B=0.1, where the truth is +4.4.
  - Every TSRI-1/2 variant has the wrong sign in ≥80% of envs.
  - Two-sided randomization targets competition between customers and listings at one moment. Here
    the interference runs through fleet state over time, which TSR doesn't address.
- **OPE** (ATE = ρ̂_B − ρ̂_A, off-policy average reward with zone features; `xp_gym/estimators/ope_pool.py`;
  jobs sociable-dancing-cuttlefish-of-respect for B=0.1 and casual-warping-bullmastiff-of-admiration for
  B=0.2) **does no better than pure LSTD-DQ, even with oracle tuning.** Best grid value per family:

  | B (ATE) | OPE-LSTD, ridge 1e-8 (= GTD fixed point, η→0) | OPE-LSTD, CV-selected | differential TD, best α | Diff-GQ1 online, best α | pure LSTD-DQ | Naive |
  |---|---|---|---|---|---|---|
  | 0.1 (4.43) | −2.55±2.26 / RMSE 7.3 / 12% | −21.9 / 26.4 / 0% | −18.3 / 22.8 / 0% | −40.7 / 45.1 / 0% | −1.33 / 6.1 / 32% | −44.2 / 48.6 |
  | 0.2 (7.23) | −5.70±3.92 / 13.5 / 6% | −43.2 / 50.5 / 0% | −36.9 / 44.2 / 0% | −77.9 / 85.1 / 0% | −2.60 / 10.6 / 24% | −84.6 / 91.8 |

  - The tuned optimum is always the least regularization. Bigger ridge shrinks θ toward 0, which gives
    the immediate-reward difference, i.e. Naive.
  - Block-CV on the projected Bellman error picks ridge ≈ 0.01, which is far too much.
  - Online TD and GTD barely move away from θ=0 in 500k steps, and larger step sizes diverge.
  - The counterfactual and importance-sampling versions agree to within noise.
  - Unregularized OPE-LSTD is essentially pure LSTD-DQ with a separate θ per arm. It has the same
    projection bias and a bit more variance.
  - B=0.3 died with its machine and wasn't rerun.

---

# Part 3: one A, three B values near it; improving DQ

*2026-10-09. Code: xp_gym `ghosts` (`xp_gym/estimators/fork_dq_pool.py`, `tests/test_fork_dq_pool.py`,
`scripts/config/fork_expts.yaml`, `scripts/offline/`). Every job is in the run log in `PROJECT_SUMMARY.md`.*

## Choosing A

f(s) is the average reward when every request uses savings threshold s. We have ATE(A,B) = f(B) − f(A), with the
percentages relative to f(A). I scanned f(s) on a 0.025 grid with 128 envs (`f_curve.png`). f rises to a peak of about 121.7
near s≈0.275, then falls more and more steeply.

- **Negative effects (−0.9%, −5%)** need a B above A on the falling side.
- **The positive effect (+0.5%)** needs a B with f(B) about 0.6 higher than f(A). That is only possible if A is at least
  0.6 below the peak. The smallest A on the 0.05 grid that allows this is **A = 0.35**.

| A:B | ATE | % of f(A) | DQ estimand λ′(0.5) | Naive (dev) |
|---|---|---|---|---|
| 0.35:0.31 | +0.58 ± 0.04 | **+0.48%** | 0.52 ± 0.11 | +18.1 |
| 0.35:0.39 | −1.05 ± 0.05 | **−0.87%** | −1.05 ± 0.12 | −18.2 |
| 0.35:0.49 | −6.02 ± 0.05 | **−4.98%** | −5.75 ± 0.13 | −63.7 |

Truth comes from 128-env CRN λ(p) runs at p ∈ {0, .3, .7, 1}. λ(p) is close to linear in all three settings, so DQ's
estimand matches the ATE.

⚠️ **Naive has the right sign in all three settings.** Naive is the immediate reward difference. Near A=0.35, pooling
more (a lower threshold) helps both immediately and in the long run, so Naive's sign agrees with the truth. Naive is still
off by 30–17× in magnitude. These settings can therefore show the RMSE and bias goals, but **not** "DQ gets the sign right
where Naive doesn't". Only the A=0 settings show that.

## Why pure LSTD's value function is hard to improve (offline, dev data)

All tests use dev seed 0, 8 envs per setting. Most used the A=0, B ∈ {0.1, 0.2, 0.3} datasets; the A=0.35 dev sets were
collected later. To separate finite-sample bias from feature (projection) bias, I fitted θ on 1, 2, 4, 8 and 16 pooled
half-envs and evaluated on every env (`scripts/offline/fs*.py`).

- **Most of the A=0 bias is finite-sample, and it shrinks slowly**, roughly like 1/√n rather than 1/n. That is why the
  1/n block jackknife only half-fixes it. At B=0.1 with 63 zones, the estimate is −2.6 at half an env, −0.2 at one env,
  1.0 / 1.7 / 2.2 at 2 / 4 / 8 envs, against truth 4.65.
- **Spatial resolution is a real trade-off.** Coarser zones (4/8/16/32 k-means regions, or 16/32 smooth graph-Laplacian
  bases) have less finite-sample bias but level off lower. With only the 3 trip-count features, the estimate is −4.3 at any
  amount of data.
- **Every feature I added made things worse, even with pooled data:**
  - time of day (Fourier terms, demand rate, and their interactions with zone counts);
  - square-root or squared counts;
  - per-zone supply;
  - total busy time. This one sends the estimate to about 0.4 × Naive (−17 at B=0.1) at any amount of data.
- **Other fixes to the fit also failed:**
  - LSTD(λ_v) toward Monte Carlo: −6 to −90 at B=0.1, even pooled;
  - discounting with γ ∈ {0.998, 0.999, 0.9995};
  - averaging the next state over the coin flip;
  - time-block fixed effects in the value function: numerically unstable;
  - a time-block-varying average reward: shorter blocks push the estimate toward Naive.

  **Interpretation:** the long-horizon value gap is identified only from slow variation, on the scale of hours. That is
  exactly where fleet features are confounded with the 24× daily and 5× weekly swings in demand. Richer or
  more Monte-Carlo-like fits pick up that confounding instead of what a car is worth.
- **At A=0.35 the sign of LSTD's bias from features flips.** With pooled θ, the bias is about −8% of Naive, versus
  +5.5% at A=0. Per-env θ has a finite-sample bias in the opposite direction, which roughly cancels it. So pure LSTD
  looks good at A=0.35 partly by luck, just as DQ(λ) did at A=0.

## A new DQ variant: fork-DQ (n-step paired DQ with counterfactual replay)

Linear value features couldn't be fixed, so the next step was to stop relying on the value function for the first H steps
after a decision.

- **Forks.** At each step where the arms would dispatch differently, `ForkDQEstimator` opens a fork: the fleet in which
  the *other* arm acted. It is stored sparsely, as up to K ghost copies of the cars that diverge.
- **Replay.** The fork is replayed on the **observed** requests and the **observed** later coin flips, using the known
  greedy dispatch rule. This needs the dispatch rule and the request log, which a platform has, but no extra randomness.
  So the per-step reward difference between the two worlds is exact.
- **Estimate.** At age H, the remaining effect is bootstrapped with the same LSTD value:
  Δ_t(H) = Σ_{k<H}(r_{t+k} − r^fork_{t+k}) + U(y_{t+H−1}) − U(y^fork_{t+H−1}), and fork_H = Σ_t s_t Δ_t(H) / T.
  - H=1 is exactly pure (paired) LSTD-DQ, and the code checks this.
  - Forks that need more than K ghosts are closed early and bootstrapped at that age.
  - Ghosts that become identical to the real car are dropped.
- **Test.** `tests/test_fork_dq_pool.py` matches brute-force forked env rollouts exactly for H ∈ {1, 5, 20}: both the reward
  sums and the bootstrap features.
- **Cost.** With 192 fork slots and 64 ghosts, a run takes about 28 s per 10k steps for 25 envs, about 25 GPU-minutes per
  50-env setting. Forks average about 10 ghosts. About 9% overflow, at mean age about 840.

### Dev results (seed 0, 50 envs; truth = DQ estimand)

Each cell is mean ± SD / RMSE.

| A:B (truth) | Pure LSTD (= fork H1) | fork H10 | fork H100 | fork H300 | **fork H1000** | DQ(λ=0.9) | Naive |
|---|---|---|---|---|---|---|---|
| 0.35:0.31 (+0.52) | 0.70±0.98 / 1.00 | −2.02±0.76 / 2.65 | −1.47±0.42 / 2.03 | −0.24±0.36 / 0.85 | 1.46±0.65 / 1.14 | −1.44±1.43 / 2.42 | 18.10 / 17.6 |
| 0.35:0.39 (−1.05) | −0.80±1.22 / 1.24 | 1.90±0.91 / 3.10 | 1.47±0.54 / 2.58 | 0.25±0.38 / 1.36 | −1.72±0.72 / 0.98 | 1.30±1.39 / 2.74 | −18.23 / 17.2 |
| 0.35:0.49 (−5.75) | −4.77±4.41 / 4.51 | 4.95±3.61 / 11.3 | 4.56±1.88 / 10.5 | 0.82±1.02 / 6.65 | −7.89±1.67 / 2.72 | 2.80±4.49 / 9.65 | −63.71 / 58.0 |

- **Forks cut the SD a lot**: to about ⅓–½ of pure LSTD at large H.
- **The mean is not monotone in H.** At H=10–100 it is pushed *against* Naive's direction by about 12–15% of Naive. So the
  LSTD value gap between two fleets that differ in about 10 cars is badly calibrated. It comes back by H=300–1000.
- **I pre-registered H=1000** (no jackknife) as the lowest summed dev RMSE: 4.8, versus 6.8 for pure LSTD. Its remaining
  bias leans in Naive's direction by about 3–5% of Naive in all three settings.

### Held-out results (seed 42, 50 envs; truth = ATE)

Each cell is mean ± SD / bias % / RMSE / right-sign %.

| A:B (ATE) | Naive | Pure LSTD | Pure + JK | DQ(λ=0.9) | **fork H1000** | fork H300 |
|---|---|---|---|---|---|---|
| 0.35:0.31 (+0.58) | 18.16±0.22 / +3032% / 17.6 / 100% | 0.30±1.11 / −48% / 1.15 / 58% | −1.30±1.25 / −323% / 2.25 / 14% | −1.82±1.35 / −414% / 2.75 / 8% | 1.50±0.74 / **+159%** / 1.18 / 96% | −0.29±0.35 / −151% / 0.94 / 20% |
| 0.35:0.39 (−1.05) | −18.20±0.22 / −1638% / 17.2 / 100% | −0.95±1.20 / +9% / 1.21 / 82% | 0.54±1.36 / +152% / 2.09 / 24% | 1.01±1.43 / +196% / 2.50 / 20% | −1.69±0.68 / −61% / **0.94** / **98%** | 0.25±0.37 / +124% / 1.35 / 26% |
| 0.35:0.49 (−6.02) | −63.67±0.45 / −957% / 57.7 / 100% | −5.34±4.83 / +11% / 4.88 / 88% | 0.13±5.58 / +102% / 8.30 / 46% | 2.32±4.90 / +139% / 9.68 / 32% | −8.01±1.35 / −33% / **2.40** / **100%** | 0.54±1.15 / +109% / 6.67 / 28% |

- **Fork-DQ (H=1000) is the best estimator at −0.87% and −5%.**
  - At −5%, it halves pure LSTD's RMSE (2.40 vs 4.88), with the right sign in 100% of envs vs 88%.
  - At −0.87%, its RMSE is 0.94 vs 1.21, with the right sign 98% vs 82%.
  - It is 18–24× below Naive in RMSE.
- **At +0.48% it fails the bias goal (+159%).** Its RMSE ties pure LSTD (1.18 vs 1.15) and it has the right sign in 96% of
  envs (pure LSTD: 58%). But its mean is 1.50 against truth 0.58.
  - Its bias is about 5% of Naive in Naive's direction in every setting. The run below checks whether that is payback
    arriving after 1000 steps.
- **Pure LSTD meets all three goals here** (bias −48%, +9%, +11%; RMSE 15–17× below Naive). But its sign accuracy is
  noise-limited: 58–88%. The offline analysis shows its small bias comes from two cancelling errors.
- **Jackknife and DQ(λ) both hurt in every A=0.35 setting.** Their corrections assume the A=0 error pattern, which flips
  sign here.
- **The dev pick held up:** the held-out numbers are within noise of the dev numbers.

### Why fork-DQ at H=1000 is still biased (job legendary-curious-chicken-of-reward)

This dev run (seed 0, 0.35:0.31, truth +0.52) used 128 ghosts and H up to 3000. It also output the reward-only part of
each fork (`forkmc_H`, with U=0).

| H | 1 | 30 | 300 | 1000 | 2000 | 3000 |
|---|---|---|---|---|---|---|
| reward part only | 18.10 (= Naive) | 10.83 | 3.14 | 1.06 ± 0.63 | 0.72 ± 1.75 | 0.62 ± 2.13 |
| fork-DQ (reward + LSTD tail) | 0.70 ± 0.98 | −2.01 | −0.24 | 1.49 ± 0.69 | 0.79 ± 2.00 | 0.82 ± 2.26 |

- **Payback continues well past 1000 steps.** The exact reward difference keeps falling until about 3000, where it reaches
  the truth. The fork oracle at A=0 had suggested about 1000.
- **The LSTD tail term goes the wrong way.** At H=1000 it *adds* +0.43 when about −0.5 remains. At H=10–300 it
  overcorrects. So LSTD's value is miscalibrated at every horizon, not only at H=1.
- **Longer forks are close to unbiased but noisy.** H=2000–3000 have bias about +0.3 but SD about 2, and 83% of forks overflow
  128 ghosts by age about 2100. The SD grows because the two fleets diverge chaotically.
  - With this setting's tiny effect, H=1000 has the better RMSE (1.19 vs 2.0–2.3).
  - Fixing the bias would need either a value function that is right about the slow tail, or many more ghosts and slots,
    which costs a lot more compute.

## Bottom line for Part 3

- **At A=0.35, B ∈ {0.31, 0.39, 0.49}** (ATE +0.48%, −0.87%, −4.98%), the best DQ variants are:
  - **fork-DQ H=1000** for the two negative effects. RMSE 0.94 and 2.40 (Naive 17.2, 57.7); bias −61% and −33%; right sign
    in 98% and 100% of envs.
  - **pure LSTD** for the +0.48% effect. RMSE 1.15 (Naive 17.6); bias −48%; right sign 58%. Fork-DQ's RMSE is the same,
    1.18, and it has the right sign in 96% of envs, but its bias is +159%.
- **What no longer works here:** DQ(λ) and the jackknife, the A=0 winners, are both worse than pure LSTD in all three settings.
- **The main lesson:** the LSTD value function's error is the limiting factor, and the direction of that error depends on
  the regime. Fork-DQ shrinks its weight a lot, but payback lasting about 3000 requests keeps a residual bias of about 3–5%
  of Naive at an affordable horizon.
- **Possible next steps:**
  1. More ghosts and slots with H≈2000, run on a larger GPU budget.
  2. Tail value features fitted on the forks themselves: forks give exact counterfactual pairs, i.e. labelled value
     differences.
  3. Repeat fork-DQ at A=0 (B=0.1–0.3), where Naive has the wrong sign.

# Part 4: A=0.2, where Naive gets the sign wrong (2026-10-10)

Naive always reads a higher threshold as worse. To make it wrong, A must sit on the **rising** side of f(s), below the
peak, with a slightly higher B giving +0.5% and larger B values giving the two negative effects (`f_curve_a020.png`).

## Choosing A

A finer f(s) scan (128 CRN envs per point, jobs wandering-loose-pigeon / fearless-meek-rooster) puts the peak at
121.69 at s=0.275. f(0.2)=121.00, which leaves just enough room for +0.5%. A=0.21 or higher cannot reach +0.5%.

| A:B | ATE | % of f(A) | DQ estimand λ′(0.5) | Naive |
|---|---|---|---|---|
| 0.2:0.26 | +0.587 ± 0.048 | **+0.49%** | +0.52 ± 0.11 | **−26.5 (wrong sign)** |
| 0.2:0.385 | −1.057 ± 0.047 | **−0.87%** | −0.97 ± 0.11 | −80.2 |
| 0.2:0.49 | −6.108 ± 0.050 | **−5.05%** | −5.73 ± 0.12 | −121.8 |

Truth: job gainful-unnatural-goshawk, 128 envs, λ(p) at p ∈ {0, .3, .7, 1}.

Naive is off by a factor of 20–75. Every DQ variant needs its value term to cancel more than 98% of Naive, so a 2%
error in the tail value is already as large as the effect at +0.49%.

## Estimators tried (dev seed 0; held-out seed 42; 50 envs each)

1. **Baselines:** pure LSTD paired, JK, DQ(λ), and fork-DQ with the LSTD tail (`fork_H`).
2. **Fork-TD (new):** the tail value V(real) − V(fork) is fitted by LSTD on the forks' own one-step transitions:
   ψ_a → ψ_{a+1} with reward r_real − r_fork, where ψ = φ(fork) − φ(real). Both worlds see the same requests, so demand
   cycles cancel. Its accumulated sums match brute-force forked rollouts exactly (`tests/test_fork_dq_pool.py`).

   Variants explored on dev (jobs forktd/forktd2/forktd3/forktd4):

   | Variant | Result |
   |---|---|
   | 63 zones | Unbiased-ish but SD 15–100 (ridge 1e-6), or shrunk toward Naive (ridge ≥ 1e-3) |
   | Laplacian zone bases (32/16/8/4 modes) | 8–16 modes with ridge 1e-4 is the best trade-off |
   | Daily Fourier interactions (1–2 harmonics) | No change at all |
   | Busy-time extras (busy buckets, first-free buckets, busy sums) | Much worse: −1.3 at +0.49%, −11 at −5.05% |

3. **Fork slots.** 192 slots left 24% of differing steps at B=0.49 without a fork. Those fell back to the paired term,
   which biased every fork estimate there toward Naive. With 384 slots there are no drops.

## Held-out results (seed 42, 50 envs; truth = ATE; RMSE / bias / right sign)

| Estimator | +0.49% (0.2:0.26) | −0.87% (0.2:0.385) | −5.05% (0.2:0.49) |
|---|---|---|---|
| Naive | 27.1 / −4617% / **0%** | 79.2 / −7492% / 100% | 115.7 / −1895% / 100% |
| Pure LSTD paired | 1.71 / −85% / 62% | 5.35 / −159% / 64% | 6.26 / −21% / 92% |
| Fork-DQ, LSTD tail, H=300 | **0.42 / −17% / 86%** | 2.38 / +204% / 12% | 7.73 / +125% / 16% |
| Fork-DQ, LSTD tail, H=1000 | 1.90 / −274% / 18% | 4.40 / −387% / 100% | 4.82 / −73% / 100% |
| Fork-TD, 8 modes, ridge 1e-4, H=100 (dev pick) | 0.89 / −147% / 12% | **1.16 / −103% / 100%** | 1.25 / +19% / 100% |
| Fork-TD, 16 modes, ridge 1e-4, H=100 | 1.10 / −183% / 2% | 1.82 / −168% / 100% | **0.55 / +2% / 100%** |

Dev numbers agree with held-out to within about 0.1–0.2 for every row.

## Long-horizon diagnostic at 0.2:0.26 (job sassy-tench-of-lucky-correction; 20 envs, 192 ghosts, 768 slots)

| H | 300 | 1000 | 2000 | 3000 | 5000 |
|---|---|---|---|---|---|
| Reward-only fork sum (truth +0.59) | −3.34 ± 0.25 | −0.08 ± 0.83 | +0.27 ± 1.73 | +0.52 ± 2.17 | +0.91 ± 2.66 |

- **Payback lasts about 3000 requests (about an hour).** The mean reaches the truth at around 3000.
- **The SD grows with the horizon.** At H=3000 the sign is right only about 70% of the time.
- **Ghosts overflow.** 98% of forks overflow 192 ghosts, at a mean age of about 3000.

## Bottom line for Part 4

- **No variant meets all three goals at A=0.2.**
- **+0.49%, the setting where Naive is wrong:** only fork-DQ with the LSTD tail at H=300 gets the right sign reliably
  (86%, bias −17%, RMSE 0.42 vs Naive 27). The same estimator gets the sign wrong at −0.87% and has the right sign only
  16% of the time at −5.05%, so it is not a defensible pick.
- **Fork-TD fixes the large effect.** At −5.05% it gives bias +2% and RMSE 0.55, 10× better than pure LSTD. At −0.87% it
  has the right sign 100% of the time (bias −103 to −168%). At +0.49% it still has the wrong sign: it captures about 87%
  of the long-run payback when 100% is needed.
- **Every value model is biased toward Naive.** This holds on-trajectory and on forks. Adding features makes it worse,
  even in fork differences where demand confounding cancels. So the residual looks like a TD/projection effect of a
  payback that lasts about an hour, not confounding.
- **Exact Monte Carlo has the opposite problem.** It removes the bias, but the variance at the needed horizon is too
  large.

# Part 5: A=0.2 with B below A

Below the peak f rises with the threshold, so every B < A has a negative true effect. Naive is positive in every
setting, so it has the wrong sign in all three.

Truth from job phenomenal-archetypal-tarsier-of-hurricane, 128 envs:

| A:B | ATE | % of f(A) | DQ estimand | Naive |
|---|---|---|---|---|
| 0.2:0.175 | −0.537 | −0.44% | −0.48 | +11.2 (wrong sign) |
| 0.2:0.15 | −1.225 | −1.01% | −1.25 | +22.2 (wrong sign) |
| 0.2:0.1 | −2.799 | −2.31% | −2.81 | +44.1 (wrong sign) |
| 0.2:0.05 | −4.938 | −4.08% | −4.69 | (truth only) |

Reaching −5% would need B ≈ 0.04, far from A, so B=0.1 is the large setting.

Each estimator job (`vast/forkall-a020-*.yaml`) runs every variant in one pass: pure LSTD with traces, fork-DQ, and
fork-TD with L16/L8 modes. Seed 0 is dev and seed 42 is held out, 50 envs each.

## Held-out results (seed 42). Cells: RMSE / bias (% of |ATE|) / right sign

| Estimator | −0.44% (0.175) | −1.01% (0.15) | −2.31% (0.1) |
|---|---|---|---|
| Naive | 11.77 / +2192% / 0% | 23.47 / +1916% / 0% | 46.82 / +1673% / 0% |
| Pure LSTD, paired | 0.85 / +62% / 56% | 1.61 / +79% / 66% | 3.51 / +90% / 48% |
| **LSTD trace λ=0.5 (dev pick)** | 0.83 / −34% / 84% | 1.20 / +8% / 82% | 2.76 / +26% / 76% |
| LSTD trace λ=0.8 | 1.06 / −130% / 90% | 1.42 / −72% / 92% | 3.01 / −44% / 90% |
| Fork-DQ, LSTD tail, H=300 | 0.41 / +56% / 84% | 1.03 / +75% / 78% | 2.42 / +84% / 76% |
| Fork-DQ, LSTD tail, H=300, jackknife | **0.37 / +39% / 86%** | **0.95 / +63% / 80%** | 2.34 / +79% / 78% |
| Fork Monte Carlo, H=1000 | 0.49 / +46% / 70% | 0.90 / +55% / 80% | **1.43 / +38% / 96%** |
| Fork-TD, 8 modes, ridge 1e-4, H=100 | 0.63 / +113% / 36% | 1.36 / +110% / 32% | 3.10 / +110% / 16% |

The dev numbers agree with held-out to within about 15 points of bias.

The dev pick was the estimator with the smallest worst-case |bias| across the three settings. That is the LSTD trace
estimator with λ=0.5, at most 33% on dev and 34% on held-out.

## Bottom line for Part 5

- **B < A is much easier, and DQ meets every goal there.** With trace λ=0.5:
  - Naive has the wrong sign in every setting, and this estimator's mean has the right sign.
  - |bias| is at most 34% of the ATE.
  - Its RMSE is 14–20× below Naive's.
  - Per-env sign accuracy is 76–84%.
- **Fork-DQ and fork Monte Carlo also meet every goal**, with lower RMSE. They fall short of the true effect in Naive's
  direction by 40–85%, which is still under 100%.
- **Fork-TD fails here.** It overshoots in Naive's direction, so its mean has the wrong sign.
- **The ranking flips with the side of A, so no estimator is uniformly best.** On the B > A side (Part 4):
  - trace λ=0.5 has bias +142% / +158% / +62%;
  - fork-TD is the best there;
  - near the peak, the long payback hurts every value model.
- **B < A is easier because the effect is monotone between A and B.** The noise is also lower: per-env SD at a similar
  |ATE| is about half its Part 4 value.
