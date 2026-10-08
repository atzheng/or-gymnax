# Project summary: DQ / DN estimators for the UberPOOL savings-threshold experiment

Context for future agents. Last updated 2026-10-08 (LSTD-DQ results added). The results below come from the
R2 output CSVs. wandb has no project for these runs, and the vastlaunch server no longer
has the job configs.

## Goal

This work is for a **paper on the LSTD version of Differences-in-Q (DQ)**
([arXiv:2206.02371](https://arxiv.org/abs/2206.02371)), so the method to showcase is
**LSTD DQ**. The test case is a pooled-rideshare dispatch experiment that A/B tests the
**minimum savings threshold** a pooled match must clear. We want to show:

1. DQ has much lower **RMSE** than Naive across a range of treatment-effect sizes
   (that is, across several `savings_threshold_B` values). Lower *variance* than Naive
   is **not** a goal: Naive has tiny variance but huge bias.
2. DQ's bias is below 100% of |ATE|.
3. In at least some scenarios Naive gets the sign wrong and DQ gets it right. The
   experiment is built so that Naive gets the sign wrong.

**Status.** The **DN (Differences-in-Neighbors)** estimator
([arXiv:2503.02271](https://arxiv.org/abs/2503.02271)) works. DN is a truncated Monte
Carlo version of DQ. Its interference graph comes from a **ghost-car simulation** inside
the env, which tracks which past dispatch decisions still change current dispatches.
That cuts variance a lot compared with a fixed spatiotemporal neighborhood. See
"Significant results" below.

**Current objective: get some version of LSTD DQ to perform about as well as DN-ghost.** ✅ Done 2026-10-08; see "LSTD-DQ results".
- Preferred: fully LSTD, with well-engineered state features.
- Acceptable fallback: an estimator that interpolates between LSTD DQ and DN, such as
  n-step or LSTD(λ) returns that bootstrap from an LSTD value function, possibly with
  ghost-trigger masking.

See "Next goal: LSTD DQ" at the end for what already exists and what to try.

## Repositories

| Repo | Location | Branch | What lives there |
|---|---|---|---|
| `or-gymnax` (`git@github.com:atzheng/or-gymnax.git`) | `/root/or-gymnax` (this repo) | `pool` | JAX env `RidesharePoolDispatch` and all ghost-car logic |
| `xp_gym` (`git@github.com:atzheng/xp_gym.git`) | **`/root/xp_gym`** | `ghosts` | Experiment harness: designs, estimators (Naive, DN, DQ…), Hydra configs, vastlaunch job YAMLs, Snakefile, analysis scripts |

> ⚠️ **Use xp_gym commit `cfec47c` (branch `ghosts`, committed 2026-10-08) or later.**
> It checkpoints the work used for the June runs on top of `ghosts@4147713`:
> - the uv migration (`pyproject.toml`, `uv.lock`; poetry is gone);
> - `xp_gym/io.py` (S3-aware `to_csv`);
> - the `group_trigger_origin_steps` rename in `network.py` and
>   `environments/rideshare_pool.py`;
> - the current `dq_expts.yaml` and `vast/*.yaml`;
> - `scripts/summarize.py`, `scripts/plot_convergence.py` and `scripts/ghost_xp.py`;
> - the `Snakefile` and `implementation_plan.md`.
>
> Anything older, including `origin/ghosts` until this commit is pushed, does not run
> against current or-gymnax. Git in `/root/xp_gym` needs
> `git -c safe.directory=/root/xp_gym …`, because the files are owned by uid 501.
>
> ⚠️ **xp_gym pulls or-gymnax from GitHub, not from a local path.** `uv.lock` pins
> `pool#5c5d050`. To make env changes in `/root/or-gymnax` visible to xp_gym, push them
> and run `uv lock --upgrade-package or-gymnax`, or point `[tool.uv.sources]` at a
> local path while you develop.

## Environment (or-gymnax, `or_gymnax/rideshare_pool.py`)

- `RidesharePoolDispatch` / `ManhattanRidesharePoolDispatch` (`rideshare_pool.py:655`, `:973`):
  Manhattan taxi events, 4333 nodes, 300 cars by default. Cars hold waypoint plans and
  can't be diverted mid-leg (see README). `max_active_trips` is the pool size; runs used 2.
- **Action** is `Float[2]` = `[threshold_canonical, threshold_counterfactual]`. The env
  picks cars itself with `greedy_select_car` (`:620`): the cheapest eligible car, where a
  car is eligible if it is solo or its marginal cost is below
  `direct_cost * (1 - threshold)`.
- **Reward** (`:898`) is `direct_cost * (1 + profit_margin) - marginal_cost` on dispatch
  and 0 on unfulfill.
- **Ghost cars**: each step forks a counterfactual "ghost group" (what if the other
  arm's threshold had been used here?). Each later step, every active group rebuilds a
  counterfactual fleet with its ghosts swapped in, dispatches on it, and compares with
  the canonical dispatch. A mismatch is a **trigger**, meaning the decision at the
  group's `origin_step` still matters now. When a trigger fires, the group grows by
  adding ghosts for newly divergent cars, up to `max_ghosts_per_group`. The full design
  is in `GHOST_CARS.md`.
  - Key functions: `check_group_triggers` (`:302`), `update_triggered_ghost_groups` (`:369`),
    `grow_group_on_trigger` (`:404`), `create_ghost_group` (`:535`), `expire_groups` (`:603`),
    `_ghost_step` (`:664`).
  - `EnvParams` knobs (`:125`): `max_groups` (ring buffer size), `max_ghosts_per_group`,
    `ghost_max_lifespan`.
  - Info keys used downstream: `action_A`, `action_B` (chosen car index, -1 if
    unfulfilled), `step`, `group_triggered`, `group_trigger_origin_steps`, `n_group_triggers`.
- Tests: `tests/test_ghost_cars.py`, which includes an oracle forked-simulation test.
  Run it with `python tests/test_ghost_cars.py`.
- `scripts/ghost_sim.py` is a tiny 1-car, 3-node simulation that dumps ghost state to
  JSONL for debugging.
- Legacy code: `scripts/dq_pool_estimators.py`, `scripts/lstd.py`, `scripts/sparse_lstd.py`
  and `scripts/mc_v2.py` hold an older sacred-based DQ-LSTD / DQ-MC pipeline for the pool
  env, from before xp_gym. `scripts/birth_death/` is a toy birth–death chain used to
  validate LSTD/DQ against an analytical ATE.


## Estimators (xp_gym)

| Name in CSV | Class | Notes |
|---|---|---|
| `naive` | `estimators/naive.py:NaiveEstimator`, `use_known_actions: true` | IPW difference of rewards. Steps where `action_A == action_B` contribute 0 but still count |
| `naive_ipw` | `NaiveEstimator`, `use_known_actions: false` | Plain IPW difference in means. Numerically about the same as `naive` here |
| `dn_ghost` | `estimators/dn.py:LimitedMemoryDNEstimator` + `network.py:GhostInterferenceNetwork` | DN with edges from ghost triggers. Past step x is a neighbor of current step y iff x's origin step triggered at y, or x == y. Uses `use_known_actions=true` (only counts past steps where `action_A != action_B`), `baseline=48`, `window_size=2000` |
| `dn_network_*` | `LimitedMemoryDNEstimator` + `RideshareNetwork` | DN with a fixed spatiotemporal neighborhood (`lookahead_steps`, `max_spatial_distance` km). `dn_network_1000` = 1000-step lookahead |
| (DQ-LSTD) | `estimators/dq.py:AvgRewardLSTDDQEstimator` | **Doesn't run on the pool env.** See "Next goal" below |

The DN update (`dn.py`) is
`est += Σ_{neighbors j in window} (z_j/p − (1−z_j)/(1−p)) · (r_t − baseline)`,
and the estimate is `est / t`. Neighbors are deduplicated by cluster through
`interference_mask` in `network.py`.

## Running experiments (all from `/root/xp_gym`)

- Install: `uv sync` (add `--group cuda` on GPU). A `.venv` already exists.
- Estimators: `uv run python scripts/run.py --config-name=dq_expts [overrides]`, using
  `scripts/config/dq_expts.yaml`. Current defaults are 25 envs × 2 trials, 500k steps, an
  estimate every 10k steps, seed 42, and a unit-randomized design with p=0.5. Ghost
  params are `+env_params.env_params.{max_groups,max_ghosts_per_group,ghost_max_lifespan}=…`.
- True ATE: `scripts/compute-ate.py` (same Hydra config; `ate.k=1000` runs) runs each
  arm alone. The output has columns `treatment,metric,value`, and ATE = mean(B) − mean(A)
  on `metric=="reward"`. Ghosts aren't needed here, so `vast/ate.yaml` sets
  `max_groups=1` to make it cheap.
- Analysis: `scripts/summarize.py` computes bias/SD/RMSE as % of |ATE| with bootstrap
  CIs and a plot. `scripts/plot_convergence.py <run_csv> <ate_csv> <out.png>` plots
  convergence.
- Sweep: `Snakefile` runs `max_active_trips ∈ {2,3}` × `savings_threshold_B ∈
  {0.1,0.2,0.3,0.5}` locally and writes to `outputs/pool/mat*_stb*/`. The `.hydra/` dirs
  for this sweep date from May 9, before the ghost work, and there are no local outputs.
- **GPU runs use vastlaunch** (`/root/vastlaunch`, CLI `vastlaunch`, server set by
  `VASTLAUNCH_SERVER_URL`). Job specs: `vast/dq.yaml` (estimators), `vast/ate.yaml`
  (ATE), plus `vast/smoke-*.yaml` and `vast/dq-short.yaml`. Each job runs on an A100,
  installs with `uv sync --group cuda`, and writes to `s3://research/dq/$VASTLAUNCH_JOB_ID`.
  The job ID is the coolname-style run name, such as `kickass-radiant-gibbon-of-symmetry`.
  Per xp_gym's `AGENTS.md`, **test locally first** with
  `vastlaunch launch --local <small-yaml>`. Use `vastlaunch launch|submit|status|logs|list`.
  Job records expire on the server, so **record each run's config overrides in its
  bead or in this file when you launch it.**
- Results storage: Cloudflare R2 bucket `research`, prefix `dq/`. `AWS_*` env vars
  are already set, including `AWS_ENDPOINT_URL`. The ATE file is `dq/ate/ate_stb.csv`.
  It came from `vast/ate.yaml`, where `ate_stb${threshold}.csv` didn't expand, and it
  corresponds to **B=0.1, `max_active_trips=2`**. A sweep needs one ATE file per
  threshold, each with a real name.

### Reproducing the DN-ghost results

You need exactly two code trees:

1. **xp_gym:** commit `cfec47c` on `ghosts`. vastlaunch rsyncs the working directory,
   so whatever is on disk is what runs, uncommitted edits included.
2. **or-gymnax:** commit **`5c5d050`** (group growth, = `origin/pool` HEAD). You don't
   need to check it out; `uv sync` installs it from GitHub through the `uv.lock` pin.
   The local `/root/or-gymnax` checkout is **not** used by these runs, and its
   uncommitted `scripts/lstd.py` edits don't matter here.

The Manhattan trip and distance data download automatically through pooch from
`atzheng/nyc-taxi-simulator-data`.

```bash
cd /root/xp_gym
# Latest dn_ghost run (smooth-aquatic-mastiff, 06-05). vast/dq.yaml already holds this config:
#   max_active_trips=2, savings_threshold_B=0.1, max_groups=2500,
#   max_ghosts_per_group=16, ghost_max_lifespan=2500, 25 envs x 2 trials, 500k steps
vastlaunch launch vast/dq.yaml
# ATE for B=0.1 (fix the unexpanded ${threshold} in vast/ate.yaml's output path first)
vastlaunch launch vast/ate.yaml
```

The other June runs are the same command with different ghost overrides. **Only
`max_ghosts_per_group` (and `max_groups` for memorable) is known.** The lifespan and
`max_groups` for kickass and formidable, and everything for `accelerated`, weren't
recorded.

| Run | Known overrides |
|---|---|
| kickass-radiant-gibbon | `max_ghosts_per_group=4` |
| formidable-chubby-seal | `max_ghosts_per_group=10` |
| memorable-puzzling-falcon | `max_groups=1000`, `max_ghosts_per_group=16` |
| smooth-aquatic-mastiff | full config above |

Local check (2026-10-08): a scratch copy of `/root/xp_gym` plus `uv sync --frozen`,
running `run.n_steps=2000 run.n_envs=2` with 200 groups × 16 ghosts, finishes on CPU in
about 2 minutes and outputs all four estimator columns. To smoke-test before launching
on GPU, use `vastlaunch launch --local` with small `run.*` overrides. (Don't use
`vast/smoke-dq.yaml`: it points at the synthetic env and writes to a shared S3 path.)

Note: the `steps` column marks the **start** of each estimate window. The row labelled
490000 is the estimate after all 500k steps.

### Reproducing the bias numbers

The user's "bias range" numbers are a 95% normal CI on percent bias at the final
estimate step (490k), taken across envs. With `ATE = 4.4333` (reward/step, from
`ate_stb.csv`):

```python
last = df[df.steps == df.steps.max()]
x = last["dn_ghost"]; m, se = x.mean(), x.std() / len(x)**0.5
bias_pct = lambda v: (v - ATE) / abs(ATE) * 100
ci = (bias_pct(m - 1.96*se), bias_pct(m + 1.96*se))
```

This reproduces the logged numbers exactly (e.g. kickass gives [-129.04, -95.66]).

## Significant results so far

Setting: the pool env with 300 cars, `max_active_trips=2`, and savings thresholds
A=0.0 and B=0.1.
- **True ATE (B − A) = +4.433 reward/step.** Arm B has a lower unfulfill rate
  (0.404 vs 0.424) and a lower marginal cost (510 vs 528).
- **Naive = −44.17 (SD 0.38): sign wrong, bias ≈ −1096%, RMSE ≈ 48.6.** `naive_ipw` gives the same.
- **DN with a spatiotemporal network (`dn_network_1000`) = −35.9 (SD 13.4): sign wrong, bias ≈ −909%.**

`dn_ghost` by run. These used 50 envs × 1 trial and 500k steps unless noted. All runs
share seed 42, so naive and `dn_network_1000` are identical across runs and the
comparisons are paired. "Sign OK" is the fraction of envs where the estimate is
positive. RMSE ≈ sqrt((mean − 4.433)² + SD²).

| Date | Run (R2 key) | Ghost config | DN mean | DN SD | RMSE | Bias % 95% CI | Sign OK |
|---|---|---|---|---|---|---|---|
| 05-28 | fantastic-tremendous-cormorant-of-art | first ghost version, 2 ghosts/group | −4.24 | 0.75 | 8.7 | [−200, −191] | 0% |
| 05-29 | flawless-steadfast-pheasant-of-pride | ? | −3.52 | 1.16 | 8.0 | [−187, −172] | 0% |
| 05-30 | strange-vociferous-asp-of-wonder | ? | −4.55 | 1.26 | 9.1 | [−211, −195] | 0% |
| 06-02 | kickass-radiant-gibbon-of-symmetry | group growth (or-gymnax `5c5d050`), 4 ghosts/group | −0.55 | 2.67 | 5.7 | [−129.0, −95.7] | 44% |
| 06-02 | formidable-chubby-seal-of-examination | 10 ghosts/group | **+1.43** | 5.37 | 6.2 | **[−101.3, −34.2]** | 68% |
| 06-03 | memorable-puzzling-falcon-of-tact | 1000 groups, 16/group | +0.18 | 3.91 | 5.8 | [−120.4, −71.5] | 40% |
| 06-03 | accelerated-competent-caracal-of-justice | unknown (not in notes) | +1.28 | 4.98 | 5.9 | [−102.3, −40.0] | 58% |
| 06-05 | smooth-aquatic-mastiff-of-psychology | 2500 groups, 16/group, lifespan 2500 (the current `vast/dq.yaml`; 25 envs × 2 trials) | +0.51 | 6.63 | 7.7 | [−130.0, −47.1] | 48% |

Takeaways:
- **Goal 1 (RMSE) is met by DN-ghost at B=0.1.** RMSE is about 6 vs Naive's 48.6, roughly
  8× lower. DN's per-env SD (2.7–6.6) is larger than Naive's, but that isn't a goal.
  The effect-size sweep hasn't been done yet.
- **Group growth was the breakthrough.** Before it, DN-ghost was consistently the wrong
  sign (bias about −190%). With 4+ ghosts per group the mean is near 0 or positive, and
  with 10 per group the point estimate has the right sign and bias around −68%.
- **Goal 2 (bias < 100%) is met by the point estimate** in the 10/group and
  `accelerated` runs. The CI lower bound still touches −100%. DN shrinks toward zero and
  recovers about a third of the true effect.
- **Goal 3 (sign) is met only weakly.** The mean DN estimate is positive in the best
  configs, but each env gets the sign right only 50–68% of the time. Naive is wrong in
  100% of envs.
- Raising ghosts per group raises SD; more groups (1000 → 2500) didn't clearly help.

### Earlier, superseded runs (May 20–26)

These used other settings (the `dn_network_*` sweeps, plus runs where naive ≈ −10.6 or
≈ −0.07). No ATE is stored for them, so don't compare them against 4.433. Notable:
`prehistoric-stimulating-caterpillar-of-attack` (05-25) has `dn_ghost` = +36.6, which
predates the "Bug fix for ghost" commit and is probably buggy. `dq/$VASTLAUNCH_JOB_ID`
is a failed run where the variable wasn't expanded.

## Next goal: LSTD DQ

Target: an LSTD-based DQ estimator with RMSE about equal to DN-ghost (about 6 at B=0.1)
or better, that also holds up across the threshold sweep. Ideally it's pure LSTD with
engineered features.

### What already exists

1. **`xp_gym/estimators/dq.py:AvgRewardLSTDDQEstimator`** (marked "not fully tested").
   It solves average-reward LSTD for a state value V, `θ = (Σ φ(φ−φ')ᵀ + λI)⁻¹ Σ φ(r − r̄)`,
   and returns `θᵀ(φ̄_treated − φ̄_control)`. Here φ is the post-step state paired with
   the action that led to it, so it's a post-decision-state value. **It can't run on the
   pool env:** its features (`available_cars_per_zone`) read `state.locations`,
   `design_info.env_state` and `design_info.src_2_zone`. Those exist only for the
   non-pool `rideshare` env and its cluster design. It has no known-actions masking, and
   the reward/state alignment hasn't been checked. `estimators/lstd_lambda.py` has the
   same feature dependency.
2. **Legacy or-gymnax pipeline** (`scripts/dq_pool_estimators.py` + `scripts/lstd.py`,
   sacred + Snakemake, 2025). This one *did* run on the pool env.
   - `dqlstd` fits a separate IPS-weighted LSTD per arm, using known-action
     probabilities when A and B choose the same car, and supports γ and ridge λ.
   - Features (`obs_to_repr`): a 4-dim histogram of cars by number of active trips. A
     richer version is commented out in the same file: zone × zone counts of remaining
     seats by (next waypoint zone, final zone), plus solo cars per zone, square-rooted.
   - `scripts/mc_v2.py:dqmc` is truncated-MC DQ.
   - `scripts/lstd.py` has **uncommitted** edits (renaming to `action_co`/`action_tr`)
     that may not match `dq_pool_estimators.py`; check before running.
   - Results are in `scripts/results/<config>/{dq,ate,tsr}-*.csv`, from an older env
     version with 500 cars, 1M events and `max_active_trips=3`, so they aren't directly
     comparable:

   | Config (B) | ATE | Naive RMSE (sign OK) | best DQ-LSTD RMSE (sign OK) | untuned `dqlstd` SD | `dqmc-100` RMSE |
   |---|---|---|---|---|---|
   | dapper-desk (0.1) | 5.37 | 6.4 (20%) | 5.0 (`opelstd` λ=1e5, γ=.995; 74%) | ~15,600 | 15.3 |
   | succinct-measurement (0.2) | 10.94 | 15.6 (8%) | 9.7 (`opelstd` λ=1e3, γ=.995) / 10.5 (`dqlstd` λ=10^4.5, γ=.99; 74%) | ~8,500 | 17.6 |
   | superficial-tennis (0.5) | 31.76 | 74.3 (0%) | 24.4 (`dqlstd` λ≤10, γ=.99; 80%) | ~9,900 | 27.6 |

   Lessons: unregularized average-reward LSTD (γ=1) is numerically unstable here.
   Discounting (γ≈0.99) and heavy ridge make DQ-LSTD beat Naive on RMSE. These "best"
   rows were picked out of 48 hyperparameter combinations on the evaluation runs
   themselves, which is optimistic. The paper needs a principled way to choose them.

### Directions to try (roughly in priority order)

- **Port a pool-env DQ-LSTD into xp_gym** as a proper `Estimator`, following the
  `dqlstd` design. It needs its own pool-env feature map built from
  `obs_to_state(n_cars, max_waypoints, obs)` (waypoints and times per car). Add
  `use_known_actions` the same way DN and Naive do: only steps where `action_A != action_B`
  carry treatment signal. `implementation_plan.md` in xp_gym lists this as route 1.
- **Feature engineering**, which is the main lever for a pure-LSTD paper:
  - spatial supply by zone (taxi zones in `xp_gym/data/taxi-zones.parquet`): idle cars,
    and cars with one passenger by next and final zone;
  - time until each car frees up (histogram buckets);
  - remaining seats;
  - interactions with the current request's origin zone;
  - square-root or log scaling, plus an intercept.
  Since the effect works through "pooled-capable cars near future demand", features
  that measure match opportunities for upcoming requests should matter most.
- **Discounted or regularized LSTD**: γ ∈ [0.99, 0.999] and ridge λ, chosen on a
  separate seed or setting rather than the evaluation runs.
- **Interpolating to DN** (the fallback): n-step or LSTD(λ) targets
  `Σ_{k<n} r_{t+k} + V̂(s_{t+n})`. As n→∞ (or λ→1) this approaches truncated-MC DQ and
  DN; with n=1 it's pure LSTD. A ghost-aware variant would sum only rewards from steps
  where the ghost group of t triggered (as DN does) and bootstrap V̂ once the group
  expires. That keeps DN's variance reduction while LSTD covers the tail beyond the
  ghost lifespan.
- Evaluate on the B ∈ {0.1, 0.2, 0.3, 0.5} sweep (one ATE per threshold) against
  Naive and DN-ghost.

## Other open threads

- Push xp_gym `cfec47c`, which is committed locally but not pushed yet. Record the
  exact config of every future run.
- Bring down DN-ghost variance at high ghosts-per-group, e.g. more envs, tuning
  `baseline`, or `ghost_max_lifespan`/`window_size`.
- Beads (`bd list --all`) record the ghost-car design history: `toc`, `v7s`, `43t`,
  `89g`, `ohf`, `lq0`, `d3i`, all closed. `xp_gym` has its own beads DB (`xp_gym-64g`:
  GhostInterferenceNetwork).

## LSTD-DQ results (2026-10-08)

**Bottom line.** An LSTD-based DQ estimator now beats DN-ghost at B=0.1 (RMSE 1.8 vs 7.7,
bias +1% vs −89%, right sign in 100% vs 48% of envs). It also meets all three goals at
B=0.2 and B=0.3, where its RMSE is about 1/3 of DN-ghost's (3.3 vs 12.4, 4.8 vs 12.9). The winning variant is **DQ(λ)**: IPW-weighted generalized advantage
estimation with an LSTD value function, which interpolates between pure LSTD-DQ (λ=0)
and truncated-MC DQ (λ→1). λ=0.9 means a TD-error trace of about 10 requests. Pure LSTD-DQ
(λ=0) with a block-jackknife bias correction also beats DN-ghost at B=0.1, but it
under-recovers the effect by about 50%.

### Code (xp_gym `ghosts`, commit 05154aa+)
- `xp_gym/estimators/lstd_dq_pool.py:PoolLSTDDQEstimator`: online estimator; its docstring
  has the formulas. U(y) is the post-decision fleet value, U(y)=θᵀφ(y), fitted by average-reward
  LSTD (γ=1, standardized ridge) on the experiment's own trajectory. It outputs a vector, one
  column per variant:
  - `paired`: Σ s_t d_t[(r_t − r^cf_t) − (U(y^cf_t) − U(y_t))]/T. It uses known actions to
    rebuild the other arm's post-decision fleet and reward.
  - `tr{λ}`: Σ_t δ_t e_t/T, with δ_t = r_t − r̄ + U(y_t) − U(y_{t−1}) and
    e_t = λe_{t−1} + (z_t/p − (1−z_t)/(1−p))·1[arms differ]. `tr0.0` ≈ `paired`.
  - `*_jk`: delete-one-block jackknife (5 blocks of 100k steps) applied to θ.
  - The O(D²) stats update once per chunk through a new `end_chunk` hook in `simulator.py`;
    `run.py` expands vector estimates using `labels`. The solve runs in float64 through
    `jax.pure_callback`. `tests/test_lstd_dq_pool.py` checks every accumulator against a
    numpy replay.
- `xp_gym/estimators/pool_features.py`: features. The **chosen set is "zones"**: counts of cars
  by number of active trips (3), idle cars by zone, 1-trip cars by final-dropoff zone, and full
  cars by first-dropoff zone, using the 63 taxi zones. That's 192 dims plus an intercept. Also
  available, but they hurt (see below): busy-time histograms, dropoffs by zone × time bucket,
  and "match features" (the reward the greedy policy would earn for K fixed reference requests,
  optionally at future offsets).
- Config `scripts/config/lstd_expts.yaml`; job spec `vast/lstd.yaml`. Summary script:
  `scripts/summarize_lstd.py <stb> <csv>...`, which pairs runs by seed and checks the naive column.

### Held-out evaluation
Seed 42, 25 envs × 2 trials, 500k steps (job silky-stoic-woodlouse-of-wizardry). Ghosts don't
change trajectories for a fixed seed (checked), so B=0.1 is paired with the
smooth-aquatic-mastiff DN-ghost run. Hyperparameters were chosen on the seed-0 dev data below.

| B | ATE | estimator | mean ± SD | bias % [95% CI] | RMSE | sign OK |
|---|---|---|---|---|---|---|
| 0.1 | 4.43 | naive | −44.18 ± 0.41 | −1097 | 48.6 | 0% |
| | | dn_ghost | 0.51 ± 6.63 | −89 [−130, −47] | 7.7 | 48% |
| | | **lstd_tr0.9** | **4.50 ± 1.79** | **+1 [−10, 13]** | **1.8** | **100%** |
| | | lstd_tr0.8 | 3.39 ± 1.74 | −24 [−34, −13] | 2.0 | 98% |
| | | lstd_paired (pure LSTD) | −1.33 ± 2.03 | −130 [−143, −117] | 6.1 | 32% |
| | | lstd_paired_jk (pure LSTD + JK) | 1.84 ± 2.50 | −58 [−74, −43] | 3.6 | 82% |
| 0.2 | 7.23 | naive | −84.62 ± 0.54 | −1271 | 91.8 | 0% |
| | | dn_ghost† | 0.84 ± 10.59 | −88 [−146, −31] | 12.4 | 44% |
| | | **lstd_tr0.9** | **7.81 ± 3.30** | **+8 [−5, 21]** | **3.4** | **98%** |
| | | lstd_paired | −2.60 ± 4.09 | −136 | 10.6 | 24% |
| | | lstd_paired_jk | 3.72 ± 4.75 | −48 [−67, −30] | 5.9 | 78% |
| 0.3 | 7.76 | naive | −120.81 ± 0.56 | −1657 | 128.6 | 0% |
| | | dn_ghost† | 5.85 ± 12.71 | −25 [−89, 40] | 12.9 | 60% |
| | | **lstd_tr0.9** | **9.62 ± 4.55** | **+24 [8, 40]** | **4.9** | **98%** |
| | | lstd_paired | −5.38 ± 6.08 | −169 | 14.5 | 26% |
| | | lstd_paired_jk | 3.79 ± 6.90 | −51 [−76, −26] | 8.0 | 66% |

RMSE versus experiment length at B=0.1, with steps counted as the window start + 10k:

| steps | naive | dn_ghost | lstd_paired | lstd_paired_jk | lstd_tr0.9 |
|---|---|---|---|---|---|
| 50k | 48.9 | 21.5 | 26.8 | 26.8 | 12.5 |
| 100k | 48.7 | 16.5 | 15.3 | 15.3 | 7.5 |
| 300k | 48.6 | 9.9 | 7.8 | 5.0 | 3.1 |
| 500k | 48.6 | 7.7 | 6.1 | 3.6 | 1.8 |

† DN-ghost at B=0.2/0.3 used a cheaper config (1000 groups, 25 envs = trial 0 only) because the
B=0.1 config costs about 40 GPU-hours per run. On the same 25 trial-0 envs, lstd_tr0.9 gives
7.67 ± 3.30, RMSE 3.3, 100% sign (B=0.2) and 9.04 ± 4.67, RMSE 4.8, 100% sign (B=0.3), versus
DN-ghost RMSE 12.4 and 12.9. So DQ(λ=0.9) has roughly 1/3 the RMSE of DN-ghost at every threshold.
DN-ghost's B=0.3 mean is less biased but its SD (12.7) is larger than |ATE|.

### What we learned
- **The DQ estimand is nearly unbiased here.** The λ(p) sweep (`scripts/lambda_p.py`, CRN,
  64 envs) shows average reward is almost linear in the treatment probability p, so
  DQ's target λ′(0.5) ≈ ATE:

  | B | ATE (λ(1)−λ(0)) | λ′(0.5) (finite difference over [0.3, 0.7]) | naive | P(arms differ) at p=.5 |
  |---|---|---|---|---|
  | 0.1 | 4.38 (4.433 from 1000-run ATE) | 4.65 ± 0.14 | −44 | 9.9% |
  | 0.2 | 7.23 | 6.73 ± 0.15 | −85 | 18.7% |
  | 0.3 | 7.76 | 7.03 ± 0.17 | −121 | 26.3% |
  | 0.5 | 0.37 (λ non-monotone) | 1.67 ± 0.15 | −180 | 38.2% |

  DN-ghost's shrinkage (it recovers about a third) is therefore truncation error, not DQ
  bias. B=0.5 is a poor showcase because its ATE is about 0.
- **Mechanism.** The fleet is saturated: about 0 idle cars and 219 of 300 full. Where the
  arms differ, A always pools the request into a 1-trip car for an immediate reward of about
  475. B either rejects it (about 54% of differing steps, reward 0) or sends a distant solo
  car (reward about 57). Per differing step the immediate gap is about −449 and the value
  gap about +490, so V must be accurate to within a few percent. The forked-rollout oracle
  (`scripts/fork_oracle.py`, 3072 forks; data `dq/fork/fork_stb0.1_s1.npz`) shows the
  cumulative B−A gap is −302 after 10 requests, −133 after 100, −53 after 300, and
  +37 ± 36 after 1000. One request takes 1.27 s on average and a trip about 350 s
  (about 275 steps). Most of the payback arrives within a few hundred requests.
- **Feature lessons** (dev data, seed 0, 8 envs × 3 thresholds; `dq/collect2/`):
  - Zone counts work best for pure LSTD.
  - Adding busy-time histograms, dropoff-by-zone×time features or match features made
    LSTD-DQ **worse**, more negative. That's not a finite-sample effect: fitting θ on 8 pooled
    envs (4M transitions) shows the same ordering. With pooled θ, paired-DQ SD across envs
    is only about 0.15, so per-env variance comes almost entirely from estimating θ.
  - The TD fixed point beats Monte Carlo regression. LSTD(λ_v) with λ_v → 1 and MC
    regression on truncated returns were both monotonically worse.
  - γ<1 (0.999) was worse than γ=1. Ridge regularization hurts; use 1e-6 on standardized
    features.
  - Pure LSTD's bias has two parts: finite-sample (about −2.5 at B=0.1; the jackknife
    removes most of it) and projection bias (about −2 even with pooled θ).
- **Why DQ(λ) works.** The trace adds the observed TD errors of the next ~1/(1−λ) steps to
  the one-step LSTD advantage. That fixes the short-horizon part of the value gap where
  linear features are weakest, while LSTD keeps the TD errors small, so variance stays far
  below truncated MC. With no value function (U=0) the trace needs λ≈0.999 and has
  SD ≈ 16.

- Coarser spatial aggregation (zones merged into 8/16/32 regions by k-means on centroids,
  `scratch/coarse.py`) cuts pure LSTD's finite-sample bias (B=0.1 paired: −0.3 → +1.2).
  After the jackknife it's a wash, and at B=0.3 it makes DQ(0.9) overshoot (11–13 vs 8.9).
  Kept the 63 zones.
- Node→zone mapping: env node i = row i of `xp_gym/data/taxi-zones.parquet`. Checked: the
  distance matrix correlates 0.993 with haversine under this order and 0.03 under
  `manhattan-nodes.idx`. ⚠️ `designs/design.py:load_rideshare_clusters` builds its map in
  `manhattan-nodes` row order, so the spatial cluster designs' zone lookup looks scrambled.
  Unused in the current experiments, but check before using cluster designs.

### Next steps
- For the paper, decide whether to present DQ(λ) (best) or pure LSTD+JK as the main
  method. Choosing λ in a principled way is open; λ=0.9 was picked on dev seed 0 and
  held up on seed 42.
- The pure-LSTD projection bias still needs better features. Untried: per-car attribute
  bases (zone × trip count × busy bucket) and features for the second
  trip's route.

### Small-ATE follow-up (2026-10-08, part 2)
See `LSTD_DQ_REPORT.md` Part 2 for the full table. In short:
- ATE(B) with A=0 peaks around B=0.3 and crosses zero at B≈0.50. Past 0.5, λ(p) is strongly convex and
  DQ's estimand no longer tracks the ATE.
- Paper-level ATEs in the linear regime: B=0.01 (+0.40%) with A=0; threshold pairs 0.3:0.35 (−0.53%),
  0.4:0.45 (−1.84%), 0.4:0.5 (−4.43%), 0.35:0.5 (−5.55%).
- DQ(λ) wins when A=0. Pure LSTD wins for the threshold pairs. Nothing works near B=0.5. The gap between
  DQ(λ) and pure LSTD is about −0.12×Naive in every regime, so DQ(λ)'s earlier wins were partly error
  cancellation. The long-horizon value function is the real bottleneck.
- TSRI fails (Naive-like). OPE is pending.
- Tools: `scripts/lambda_p.py` accepts `A:B`; `scripts/summarize_lstd.py` accepts `A:B` and prints
  DQ's estimand.

## Run log (from 2026-10-08, LSTD-DQ work)

| Job ID | Spec | Purpose / overrides | Output |
|---|---|---|---|
| stirring-tasteful-lemur-of-examination | xp_gym `vast/lambda.yaml` (STBS=0.1) | λ(p) sweep, 64 envs × 500k, p∈{0,.1,.3,.4,.5,.6,.7,.9,1}, CRN, mat=2, no ghosts | `s3://research/dq/lambda/lambda_stb0.1.csv` |
| tested-bright-seal-from-lemuria | xp_gym `vast/lambda-sweep.yaml` (STBS="0.2 0.3 0.5") | same as above for other thresholds | `s3://research/dq/lambda/lambda_stb{0.2,0.3,0.5}.csv` |
| curvy-jasper-trout-of-authority | xp_gym `vast/fork.yaml` (STB=0.1, SEED=1) | forked-rollout oracle: 128 envs × 24 forks, warmup 100k, gap 3k, horizon 40k, p=0.5 | `s3://research/dq/fork/fork_stb0.1_s1.npz` |
| angelic-honored-muskrat-of-upgrade | xp_gym `vast/collect.yaml` (then writing to prefix `collect/`; STB=0.1, SEED=0, K=128, E=8) | offline dataset for LSTD dev: per-step features incl. match features over 128 reference requests | `s3://research/dq/collect/stb0.1_s0_K128/env*.npz` |
| winged-nifty-firefly-of-superiority / naughty-porcelain-mongrel-of-elegance / loud-ancient-rat-of-tempest | xp_gym `vast/collect.yaml` with STB∈{0.1,0.2,0.3}, SEED=0, K=128, E=8 | dev datasets v2: + drop-zone×time features, match features at offsets 0/120/300 s (1510 dims) | `s3://research/dq/collect2/stb{0.1,0.2,0.3}_s0_K128/` |
| ~~unbeatable-stalwart-okapi~~ | destroyed before running; superseded by silky-stoic-woodlouse-of-wizardry | | |
| ~~omniscient-watchful-lionfish-of-dew~~ | DN-ghost B=0.2 with 2500 groups × 2 trials | destroyed after ~2 h: ~7 steps/s ⇒ ~40 GPU-h | |
| ~~meek-orange-rattlesnake-from-pluto~~ | DN-ghost B=0.3 with 2500 groups × 2 trials | destroyed (same reason) | |
| silky-stoic-woodlouse-of-wizardry | xp_gym `vast/lstd.yaml` (STBS="0.1 0.2 0.3") | held-out eval of `PoolLSTDDQEstimator` (zones, γ=1, ridge 1e-6, traces 0/.5/.8/.9/.95, jackknife 5 blocks) + naive; config `lstd_expts`, seed 42, 25 envs × 2 trials, 500k, no ghosts | `s3://research/dq/lstd/silky-stoic-woodlouse-of-wizardry/lstd_stb{s}.csv` |
| prophetic-victorious-bullfrog-from-pluto | xp_gym `vast/dq-stb0.2.yaml` | DN-ghost (+naive, dn_network_1000) at B=0.2: max_groups=1000, 16/group, lifespan 2500, **num_trials=1** (25 envs), seed 42 | `s3://research/dq/prophetic-victorious-bullfrog-from-pluto` |
| innocent-cuddly-raven-of-enrichment | xp_gym `vast/dq-stb0.3.yaml` | same at B=0.3 | `s3://research/dq/innocent-cuddly-raven-of-enrichment` |
| inscrutable-topaz-guppy-from-venus / loud-origami-cuscus-of-artistry | xp_gym `vast/lambda-scan.yaml` / `lambda-scan2.yaml` | λ(p) scan for small-ATE targets: 64 envs × 500k, p∈{.3,.7,1} (p=0 reused from lambda_stb0.5, same keys), B∈{.4,.45,.55,.6} / {.65,.7,.75,.8} | `s3://research/dq/lambda/lambda_stb{B}.csv` |
| tiny-jellyfish-of-great-certainty | xp_gym `vast/ate-precise.yaml` | precise truth at small-ATE thresholds B∈{.497,.518,.571}: 256 envs × 500k, p∈{0,.3,.7,1} | `s3://research/dq/ate2/ate_stb{B}.csv` |
| demonic-neat-axolotl-of-unity / bizarre-talented-pudu-of-virtuosity | xp_gym `vast/lstd-small-a.yaml` / `lstd-small-b.yaml` | LSTD eval (config `lstd_expts`, seed 42, 50 envs, traces 0/.5/.8/.9/.95/.97) at B∈{.497,.518} / {.571}, targeting ATE ≈ +0.5%, −0.9%, −5% of avg reward | `s3://research/dq/lstd/<job>/lstd_stb{B}.csv` |
| unnatural-shellfish-of-necessary-economy | xp_gym `vast/lambda-small-b.yaml` | λ(p) scan near B=0: 128 envs × 500k, p=0 once (`ate_p0_E128.csv`) then p∈{.3,.7,1} for B∈{.005,.01,.02,.03,.05} | `s3://research/dq/ate2/ate_stb{B}.csv`, `ate_p0_E128.csv` |
| casual-warping-bullmastiff-of-admiration | xp_gym `vast/ope.yaml` (STBS="0.1 0.2 0.3") | OPE (LSTD/Diff-GQ1/differential TD, cf & IS, reg grids + CV selection), config `ope_tsr_expts`, seed 42, 50 envs, paired with silky-stoic | `s3://research/dq/ope/<job>/ope_stb{B}.csv` |
| simple-expert-stallion-of-acceptance | xp_gym `vast/tsri.yaml` (STBS="0.1 0.2 0.3") | TSR design + TSRN/CR/LR/TSRI-1/2 (β grid), a_C=a_L=0.5 and market-balance eq. (26) | `s3://research/dq/tsri/<job>/tsri_stb{B}_{ac0.5_al0.5,mb1}.csv` |
| miraculous-deer-of-sudden-storm | xp_gym `vast/lstd-small-c.yaml` | LSTD eval (config `lstd_expts`, seed 42, 50 envs, traces 0/.5/.8/.9/.95/.97) at B∈{.01,.02,.005} (near-zero ATE) | `s3://research/dq/lstd/miraculous-deer-of-sudden-storm/lstd_stb{B}.csv` |
| spectral-invincible-tench-of-protection | xp_gym `vast/lambda-pairs.yaml` | λ(p) for threshold pairs A:B ∈ {.4:.45, .4:.5, .3:.35, .3:.4, .45:.5, .35:.5}, 64 envs × 500k, p∈{0,.3,.7,1} (negative small ATEs in the near-linear regime) | `s3://research/dq/lambda/lambda_pair{A:B}.csv` |
| provocative-humble-stork-of-protection / bright-manipulative-dogfish-of-music | xp_gym `vast/lstd-pairs-a.yaml` / `lstd-pairs-b.yaml` | LSTD eval (config `lstd_expts`, seed 42, 50 envs, traces 0/.5/.8/.9/.95/.97) at A:B ∈ {.3:.35, .4:.5} / {.4:.45, .35:.5} (negative ATEs −0.56%…−5.55%) | `s3://research/dq/lstd/<job>/lstd_pair{A:B}.csv` |
| ~~perky-magnificent-lion-of-wind~~ (failed to provision) → brainy-discerning-peccary-of-education | xp_gym `vast/ate-pairs.yaml` | precise truth, 256 envs × 500k, p∈{0,1}, A:B ∈ {.3:.35, .4:.45} | `s3://research/dq/ate2/ate_pair{A:B}.csv` |
| sociable-dancing-cuttlefish-of-respect | xp_gym `vast/ope-0.1.yaml` | rerun of OPE at B=0.1 (casual-warping's B=0.1 failed at startup: GitHub download timeout) | `s3://research/dq/ope/sociable-dancing-cuttlefish-of-respect/ope_stb0.1.csv` |
