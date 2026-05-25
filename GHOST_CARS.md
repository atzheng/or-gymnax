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

### Same-car skip

If `action[0] == action[1]` (canonical and counterfactual are the same car), the two counterfactual worlds are identical and there is nothing to track. No ghost pair is created and `ghost_write_idx` is **not** advanced. All existing ghosts and their state (including `ghost_origin_step`) are left intact, so trigger detection and origin-step reporting for pre-existing ghosts are unaffected. The next genuine dispatch writes to the same buffer positions that would have been used had the skip not occurred.

### Ring buffer

Ghosts are stored in a fixed-size buffer of `max_ghosts` slots. A `ghost_write_idx` pointer advances by 2 each dispatch step where `canonical != counterfactual`, wrapping via modulo. Old ghosts are also deactivated when they exceed `ghost_max_lifespan`.

### Expiry ordering

Within each step, trigger detection runs *before* expiry. A ghost that expires at time `t` can still trigger at `t`, but will be inactive at `t+1`. This matters for the oracle test.

## State additions

`EnvState` gained 9 fields:

```python
ghost_waypoints: Integer[Array, "max_ghosts max_waypoints"]
ghost_times: Integer[Array, "max_ghosts max_waypoints"]
ghost_birth_time: Integer[Array, "max_ghosts"]
ghost_origin_step: Integer[Array, "max_ghosts"]
ghost_type: Integer[Array, "max_ghosts"]        # 0=A, 1=B
ghost_active: Bool[Array, "max_ghosts"]
ghost_excluded_cars: Integer[Array, "max_ghosts max_exclusions"]
ghost_n_excluded: Integer[Array, "max_ghosts"]
ghost_write_idx: Integer[Array, ""]              # ring buffer pointer
```

`EnvParams` gained 3 fields:

```python
max_ghosts: int = 128                # ring buffer size
ghost_max_lifespan: int = 50         # steps before expiry
ghost_max_branch_depth: int = 0      # 0 = no branching (not yet implemented)
```

## Action interface change

The action changed from a scalar (car index) to `Integer[Array, "2"]` = `[canonical_car, counterfactual_car]`. Use `action[0] < 0` to signal unfulfill.

`GreedyPolicy.apply` now returns a `(2,)` array: the best and second-best car.

The `xp_gym` `XPEnvironment` wrapper composes the action pair from the two policies' outputs based on the treatment assignment.

## Info dict additions

Each step returns these ghost-related keys in `info`:

- `ghost_triggered`: `Bool[max_ghosts]` — which ghosts triggered (ghost wins OR canonical wins)
- `ghost_trigger_origin_steps`: `Integer[max_ghosts]` — origin step of each triggered ghost (-1 if not triggered)
- `n_ghost_triggers`: scalar — total number of triggers this step
- `n_active_ghosts`: scalar — number of active ghosts after expiry

## Key functions

All in `or_gymnax/rideshare_pool.py`:

| Function | Purpose |
|----------|---------|
| `check_ghost_triggers` | vmap over ghosts; returns `triggered`, `ghost_wins`, costs, feasibility |
| `update_triggered_ghosts` | vmap dispatch trip to ghost-wins ghosts |
| `create_ghost_pair` | Build Ghost A + Ghost B from canonical/cf car indices |
| `write_ghosts_to_buffer` | Write 2 ghosts into ring buffer at write_idx |
| `expire_ghosts` | Deactivate ghosts past lifespan |
| `compute_real_car_costs` | vmap marginal cost over real fleet |
| `_ghost_step` | Orchestrates trigger check, state update, and expiry |

## Tests

`tests/test_ghost_cars.py` — 12 tests sharing a single env instance (to avoid a gymnax JIT caching bug):

1. **Smoke test** — env compiles and runs with ghost tracking
2. **Ghost creation** — exactly 2 ghosts created per dispatch, types 0 and 1
3. **Ghost A snapshot** — matches canonical car's pre-dispatch state
4. **Ghost B dispatch** — matches counterfactual car with trip inserted
5. **Ring buffer wrapping** — oldest slots overwritten when buffer fills
6. **Oracle forked simulation** — forks simulation at step t, independently tracks counterfactual car state, verifies trigger pattern matches ghost tracker over 8 future steps (including canonical-wins triggers)
7. **Exclusion sets** — correct car indices in exclusion lists
8. **JIT + scan compatibility** — works under `jax.jit` and `jax.lax.scan`
9. **Policy action shape** — `GreedyPolicy` returns `(2,)` action
10. **No ghost when same car** — no ghosts created and `ghost_write_idx` unchanged when canonical == counterfactual
11. **Buffer position after skip** — `ghost_write_idx` and `ghost_origin_step` are correct for ghosts created immediately after a same-car skip
12. **Oracle after same-car skip** — oracle test verifying that ghost pairs created after a skip produce correct trigger patterns and origin-step reporting over 6 future steps

Run with: `python -m pytest tests/test_ghost_cars.py -p no:logfire`

## Files modified

- `or_gymnax/rideshare_pool.py` — all ghost car logic, action interface change, policy change
- `xp_gym/environments/environment.py` — `XPEnvironment.step_env` composes `[canonical, cf]` action pair
- `xp_gym/environments/rideshare_pool.py` — simplified to import from `or_gymnax`
- `tests/test_ghost_cars.py` — new test file

## Not yet implemented

- **Branching** (`ghost_max_branch_depth > 0`): When a ghost triggers, it could spawn child ghosts to track second-order effects. The buffer structure and exclusion lists support this, but `_ghost_step` doesn't yet spawn children. The `ghost_max_branch_depth` param defaults to 0 (disabled).
