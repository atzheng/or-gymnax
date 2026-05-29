# Ghost Car Counterfactual Tracking

## Goal

Track long-range causal effects of dispatch decisions in the pooled rideshare environment. At each timestep the policy submits two car indices: a **canonical** dispatch (actually executed) and a **counterfactual** dispatch (the alternative). Two "ghost" cars are spawned to track what would have happened under each scenario, and at every future timestep the system checks whether the dispatch decision still matters — i.e., whether either the ghost or the canonical car would have been chosen for a new ride request.

This produces a per-timestep **trigger signal** that records which past decisions are still affecting the present, enabling downstream causal inference (e.g., ATE estimation with proper credit assignment).

## Design

### Ghost cars

Each dispatch creates two ghosts stored in a fixed-size ring buffer:

| Ghost | Type | Contents | Meaning |
|-------|------|----------|---------|
| A | 0 | Canonical car's state *before* dispatch | "What if we hadn't dispatched this car?" |
| B | 1 | Counterfactual car's state *with* the trip added | "What if we'd dispatched this car instead?" |

### Trigger condition

At each future timestep, for each active ghost, a **trigger** fires if either:

1. **Ghost wins**: The ghost car would beat every real car (excluding cars in its exclusion set) for the current ride request. The trip is dispatched to the ghost, updating its state.
2. **Canonical wins**: One of the excluded (displaced) real cars is the best real car for the current request. The ghost state is *not* updated, but the trigger is recorded.

Both cases represent a divergence between the real and counterfactual worlds — a signal that the original decision still matters.

### Exclusion sets

Each ghost tracks which real cars have been displaced in its counterfactual world via `ghost_excluded_cars` (a fixed-size array of car indices, padded with -1). For depth-0 (no branching), each ghost excludes exactly one car: the one it forked from. The max exclusion size is `ghost_max_branch_depth + 1`.

### Ghost creation

Ghost creation is symmetric and independent: Ghost A is written iff A fulfills; Ghost B is written iff B fulfills.

| A | B | Ghosts written | `ghost_write_idx` advance |
|---|---|---|---|
| fulfills | fulfills | Ghost A + Ghost B | +2 |
| fulfills | unfulfills | Ghost A only | +1 |
| unfulfills | fulfills | Ghost B only | +1 |
| unfulfills | unfulfills | none | 0 |

Both policies select independently from the full fleet and may choose the same car (action_A == action_B). Ghost A and Ghost B are written independently based solely on whether each policy dispatches. At future timesteps, each ghost excludes its corresponding real car (via `ghost_excluded_cars`) when making dispatch decisions.

Ghost A = canonical car's state **before** dispatch ("what if A hadn't dispatched?").
Ghost B = counterfactual car's state **with** the trip inserted ("what if B had dispatched?").

### Ring buffer

Ghosts are stored in a fixed-size buffer of `max_ghosts` slots. `ghost_write_idx` advances as shown in the table above, wrapping via modulo. Old ghosts are also deactivated when they exceed `ghost_max_lifespan`.

### Expiry ordering

Within each step, trigger detection runs *before* expiry. A ghost that expires at time `t` can still trigger at `t`, but will be inactive at `t+1`. This matters for the oracle test.

## State additions

`EnvState` ghost fields (per-group + per-slot):

```python
# Group-level (ring buffer of groups)
group_origin_step: Integer[Array, "max_groups"]
group_birth_time: Integer[Array, "max_groups"]
group_active: Bool[Array, "max_groups"]
group_threshold: Float[Array, "max_groups"]  # creation-time threshold (logging only)
group_n_ghosts: Integer[Array, "max_groups"]  # 0, 1, or 2
group_excluded_cars: Integer[Array, "max_groups max_ghosts_per_group"]  # -1 padded
group_ghost_car_ids: Integer[Array, "max_groups max_ghosts_per_group"]  # -1 padded
group_write_idx: Integer[Array, ""]

# Ghost slot state (contiguous blocks: group g owns slots [g*mpg, (g+1)*mpg))
ghost_waypoints: Integer[Array, "max_ghost_slots max_waypoints"]
ghost_times: Integer[Array, "max_ghost_slots max_waypoints"]
```

`EnvParams` ghost fields:

```python
max_groups: int = 64                 # ring buffer size for groups
max_ghosts_per_group: int = 2        # slots per group (default 2: Ghost A + B)
ghost_max_lifespan: int = 50         # steps before expiry
# max_ghost_slots = max_groups * max_ghosts_per_group (derived property)
```

## Action interface

The action is `Float[Array, "2"]` = `[savings_threshold_A, savings_threshold_B]`.

- `threshold_A`: the canonical (treatment A) policy's savings threshold
- `threshold_B`: the counterfactual (treatment B) policy's savings threshold

The environment internally selects cars using a deterministic greedy policy (`greedy_select_car`):
1. Canonical car = cheapest eligible car under `threshold_A`
2. Counterfactual car = cheapest eligible car under `threshold_B`, excluding canonical

A car is **eligible** if it is solo (no active trips) or its marginal cost is below `direct_cost × (1 − threshold)`, where `direct_cost = distances[src, dest]` (the direct pickup-to-dropoff distance). This is the sole baseline for the savings threshold — independent of which solo cars happen to be available.

Ghost creation and logging are symmetric: `action_A` is the canonical car index or -1 if A unfulfills; `action_B` is the counterfactual car index or -1 if B unfulfills. Each is independent.

`GreedyPolicy.apply` returns `[savings_threshold, savings_threshold]` — the policy's threshold for both arms. Car selection happens inside the environment.

The chosen car indices are logged as `action_A` and `action_B` in the info dict.

## Info dict additions

Each step returns these keys in `info`:

- `action_A`: scalar int32 — canonical car index chosen (-1 on unfulfill)
- `action_B`: scalar int32 — counterfactual car index chosen (-1 on unfulfill)
- `ghost_triggered`: `Bool[max_ghosts]` — which ghosts triggered (ghost wins OR canonical wins)
- `ghost_trigger_origin_steps`: `Integer[max_ghosts]` — origin step of each triggered ghost (-1 if not triggered)
- `n_ghost_triggers`: scalar — total number of triggers this step
- `n_active_ghosts`: scalar — number of active ghosts after expiry

## Key functions

All in `or_gymnax/rideshare_pool.py`:

| Function | Purpose |
|----------|---------|
| `check_ghost_triggers` | Flatten groups to per-slot, vmap trigger check; returns `triggered`, `ghost_wins`, costs, feasibility |
| `update_triggered_ghosts` | vmap dispatch trip to ghost-wins slots |
| `create_ghost_group` | Create a ghost group with 1 or 2 ghosts (replaces `apply_ghost_a`/`apply_ghost_b`) |
| `expire_groups` | Deactivate groups past lifespan |
| `_flatten_groups_to_slots` | Expand group-level fields to per-slot arrays for vmap |
| `compute_real_car_costs` | vmap marginal cost over real fleet |
| `_ghost_step` | Orchestrates trigger check, state update, and expiry |

## Tests

`tests/test_ghost_cars.py` — 19 tests sharing a single env instance (to avoid a gymnax JIT caching bug):

1. **Smoke test** — env compiles and runs with ghost tracking
2. **Ghost creation** — 1 active group with 2 ghosts created per dispatch
3. **Ghost A snapshot** — slot 0 matches canonical car's pre-dispatch state
4. **Ghost B dispatch** — slot 1 matches counterfactual car with trip inserted
5. **Ring buffer wrapping** — oldest group overwritten when buffer fills
6. **Exclusion sets** — correct car indices in group exclusion set
7. **Both fulfill, two ghosts** — 1 group with 2 ghosts, group_write_idx +1
8. **Same-car case** — no group when both policies select same car
9. **A fulfills, B unfulfills** — 1 group with 1 ghost (Ghost A)
10. **A unfulfills, B fulfills** — 1 group with 1 ghost (Ghost B)
11. **Both unfulfill, no ghosts** — no group
12. **Ghost A content when B unfulfills** — content verification
13. **Ghost B content when A unfulfills** — content verification
14. **Buffer advances correctly** — group_write_idx advances by 1/1/1/0
15. **Shared exclusion set** — both ghosts in group share same exclusion set
16. **Threshold uses current action** — triggers use current threshold, not stored
17. **Oracle forked simulation** — oracle vs tracker trigger verification over 8 steps
18. **Policy action shape** — `GreedyPolicy` returns `(2,)` action
19. **JIT + scan compatibility** — works under `jax.jit` and `jax.lax.scan`

Run with: `python tests/test_ghost_cars.py`

## Files modified

- `or_gymnax/rideshare_pool.py` — all ghost car logic, action interface change, policy change
- `xp_gym/environments/environment.py` — `XPEnvironment.step_env` composes `[canonical, cf]` action pair
- `xp_gym/environments/rideshare_pool.py` — simplified to import from `or_gymnax`
- `tests/test_ghost_cars.py` — new test file

## Revised model: ghost groups

### Core concept: one counterfactual world per group

Each dispatch decision at timestep `t` creates a **ghost group** identified by `origin_step = t`. The group collectively represents a single counterfactual world — "what if we had made a different dispatch decision at time t?" All ghosts in a group share one exclusion set (the set of real cars whose states differ between the canonical and counterfactual worlds).

At creation, the group contains two ghosts:
- **Ghost A**: car A's pre-dispatch state (canonical car, undispatched in the counterfactual)
- **Ghost B**: car B's with-trip state (cf car, dispatched in the counterfactual)

The group's exclusion set is `{A, B}` — both cars have different states in the counterfactual world.

### Trigger rule (single, unified)

At each subsequent timestep `t'`, for each active ghost group `g`:

1. **Build the counterfactual fleet**: take the real fleet, swap out all cars in `g`'s exclusion set, swap in `g`'s ghost cars.
2. **Run dispatch** on the counterfactual fleet (using the group's savings threshold).
3. **Compare** to the canonical dispatch result.

**If the canonical dispatches car X and the counterfactual dispatches car Y, and X ≠ Y, a trigger fires.** This includes fulfillment asymmetry: canonical fulfills but cf doesn't (or vice versa) is also X ≠ Y (with the unfulfilling side as "no car").

Both dispatches use the **current action's threshold** — not a stored group threshold. The cf fleet has different car states but the same policy parameters, so triggers isolate the causal effect of car-state divergence.

That's the only rule. No special cases needed:
- If X is in the group's exclusion set, the cf fleet has a ghost in X's position with different state, so cf will necessarily pick Y ≠ X. The trigger fires naturally.
- If X is not in the exclusion set but the cf fleet picks differently due to swapped-in ghosts changing the cost landscape, the trigger fires.
- If both worlds agree (X = Y), no trigger.

### Group growth on trigger

When a trigger fires (canonical dispatches X, cf dispatches Y, X ≠ Y), the group grows to track the new divergence:

For **car X** (dispatched canonically, not in cf world):
- If X is NOT already in the group's exclusion set: add Ghost X = X's pre-dispatch state. Add X to exclusion set.
- If X is already excluded: X's ghost already exists. No new ghost needed (the existing ghost already tracks X's counterfactual state; the divergence just deepened).

For **car Y** (dispatched in cf world, not canonically):
- If Y is already a ghost in the group: update Ghost Y's state (insert the trip).
- If Y is a real car NOT in the group: add Ghost Y = Y's with-trip state. Add Y to exclusion set.

Each trigger can add 0, 1, or 2 new ghosts depending on how many of {X, Y} are already in the group.

**Example**: Group from time `t` has ghosts `{A, B}` with exclusion set `{A, B}`. At time `t'`:
- Canonical dispatches car **C**, cf fleet dispatches ghost **B** → trigger
- Ghost B is updated (insert trip)
- Car C is new → Ghost C = C's pre-dispatch state added, exclusion set grows to `{A, B, C}`

The group now has ghosts `{A, B (updated), C}` and exclusion set `{A, B, C}`.

### Group size limit

`max_ghosts_per_group` caps how many ghosts a group can hold. When a trigger would add a new ghost but the group is at capacity, the addition is skipped (the trigger is still recorded). The exclusion set size equals the number of ghosts in the group.

### Memory layout: contiguous per-group blocks

Ghost slots (waypoints/times) are allocated in contiguous blocks per group. Group `g` owns slots `[g * max_ghosts_per_group, (g+1) * max_ghosts_per_group)`. Total buffer size = `max_groups × max_ghosts_per_group`.

This eliminates stale reference bugs: when the group ring buffer wraps and group `g` is overwritten, all its slots are overwritten too. No separate slot write pointer needed — slot index is always `group_idx * max_ghosts_per_group + local_offset`.

### What changes from the current implementation

| Aspect | Current (per-ghost) | Revised (per-group) |
|--------|---------------------|---------------------|
| Trigger unit | Individual ghost slot | Ghost group (by `origin_step`) |
| Trigger rule | Ghost beats fleet, or excluded car is best | Single rule: canonical ≠ counterfactual dispatch |
| Exclusion set | Per ghost: `{A}` or `{B}` | Per group: `{A, B, ...}`, grows on trigger |
| Counterfactual dispatch | Each ghost independently vs real fleet minus its own exclusion | Build full counterfactual fleet (all group ghosts swapped in), dispatch once |
| Group growth | Not implemented | Automatic on trigger: add ghosts for newly-divergent cars |
| Trigger info | Per ghost slot | Per `origin_step` |
| Size limit | `ghost_max_branch_depth` (confusing name) | `max_ghosts_per_group` (direct) |

### Not yet implemented

- Per-group trigger checking logic (currently per-ghost)
- Group growth on trigger
- Shared exclusion sets within groups
- `max_ghosts_per_group` param (replacing `ghost_max_branch_depth`)
