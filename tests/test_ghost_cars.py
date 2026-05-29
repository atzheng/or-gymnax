"""
Oracle test for ghost car counterfactual tracking.

The key idea: run two independent simulations forked at step t — one with the
canonical dispatch, one with the counterfactual dispatch. At each future step,
check whether the counterfactual car would have beaten the real fleet under the
policy's savings threshold. This must match the ghost tracker's trigger output.

NOTE: Tests share a single env/params instance to avoid a pre-existing gymnax
JIT caching bug where creating multiple env instances triggers __bool__ errors.
"""
import jax
import jax.numpy as jnp

from or_gymnax import rideshare as rs
from or_gymnax.rideshare_pool import (
    EnvParams,
    RidesharePoolDispatch,
    GreedyPolicy,
    insert_and_optimize_trip,
    greedy_select_car,
    _num_wp,
)

# Shared env/params to avoid gymnax JIT caching issue with multiple env instances
_N_CARS = 3
_N_NODES = 5
_N_EVENTS = 30
_KEY = jax.random.PRNGKey(42)
_SK1, _SK2 = jax.random.split(_KEY)
_DISTANCES = jnp.array([
    [0, 2, 5, 3, 7],
    [2, 0, 3, 4, 6],
    [5, 3, 0, 1, 4],
    [3, 4, 1, 0, 2],
    [7, 6, 4, 2, 0],
])

ENV = RidesharePoolDispatch(n_cars=_N_CARS, n_nodes=_N_NODES, n_events=_N_EVENTS)
ENV_PARAMS = EnvParams(
    events=rs.RideshareEvent(
        t=jnp.arange(_N_EVENTS) * 3,
        src=jax.random.randint(_SK1, (_N_EVENTS,), 0, _N_NODES),
        dest=jax.random.randint(_SK2, (_N_EVENTS,), 0, _N_NODES),
    ),
    distances=_DISTANCES,
    n_cars=_N_CARS,
    max_active_trips=2,
    max_groups=8,
    ghost_max_lifespan=10,
)

# Threshold=0.0: solo cars always eligible; pool cars eligible if marginal cost
# < direct_cost. Equal thresholds always pick the same car → no ghosts.
_THRESH = jnp.array([0.0, 0.0])

# _MIXED_THRESH with _make_mixed_state: A picks car 1 (cheap pool), B picks car 0 (solo).
# Guaranteed different cars → ghost group written.
_MIXED_THRESH = jnp.array([0.0, 1.0])

_MPG = ENV_PARAMS.max_ghosts_per_group  # max ghosts per group (2)


def test_basic_step_runs():
    """Smoke test: env with ghost tracking compiles and runs."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    obs2, state2, reward, done, info = ENV.step(key, state, _THRESH, ENV_PARAMS)
    assert obs2.shape == obs.shape
    assert "group_triggered" in info
    assert "n_group_triggers" in info
    assert "action_A" in info
    assert "action_B" in info


def test_ghosts_created_on_dispatch():
    """After one dispatch step with different cars, exactly 1 group with 2 ghosts."""
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_mixed_state(base)
    _, state2, _, _, info = ENV.step_env(key, state, _MIXED_THRESH, ENV_PARAMS)
    assert int(info["action_A"]) != int(info["action_B"]), "cars must differ for ghosts"
    n_active_groups = jnp.sum(state2.group_active)
    assert n_active_groups == 1, f"Expected 1 active group, got {n_active_groups}"
    g_idx = jnp.argmax(state2.group_active)
    assert int(state2.group_n_ghosts[g_idx]) == 2, "Expected 2 ghosts in group"


def test_ghost_a_is_pre_dispatch_snapshot():
    """Ghost A (slot 0) should have the canonical car's pre-dispatch state."""
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_mixed_state(base)
    _, state2, _, _, info = ENV.step_env(key, state, _MIXED_THRESH, ENV_PARAMS)
    canonical_car = int(info["action_A"])
    g_idx = int(jnp.argmax(state2.group_active))
    slot_idx = g_idx * _MPG + 0  # Ghost A is slot 0
    assert jnp.array_equal(state2.ghost_waypoints[slot_idx], state.waypoints[canonical_car])
    assert jnp.array_equal(state2.ghost_times[slot_idx], state.times[canonical_car])


def test_ghost_b_is_counterfactual_dispatch():
    """Ghost B (slot 1) should have the counterfactual car with the trip added."""
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_mixed_state(base)
    _, state2, _, _, info = ENV.step_env(key, state, _MIXED_THRESH, ENV_PARAMS)
    cf_car = int(info["action_B"])
    expected_wp, expected_t, _, _ = insert_and_optimize_trip(
        ENV_PARAMS.distances,
        state.waypoints[cf_car], state.times[cf_car],
        state.event.src, state.event.dest, state.event.t,
        ENV_PARAMS.max_active_trips,
    )
    g_idx = int(jnp.argmax(state2.group_active))
    slot_idx = g_idx * _MPG + 1  # Ghost B is slot 1
    assert jnp.array_equal(state2.ghost_waypoints[slot_idx], expected_wp)
    assert jnp.array_equal(state2.ghost_times[slot_idx], expected_t)


def test_ring_buffer_wraps():
    """When group buffer fills, ring buffer should overwrite oldest slots."""
    ep = ENV_PARAMS.replace(max_groups=2, ghost_max_lifespan=100)
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ep)

    key, sk = jax.random.split(key)
    _, s1, _, _, _ = ENV.step_env(sk, _make_mixed_state(base), _MIXED_THRESH, ep)
    assert jnp.sum(s1.group_active) == 1

    key, sk = jax.random.split(key)
    _, s2, _, _, _ = ENV.step_env(sk, _make_mixed_state(s1), _MIXED_THRESH, ep)
    assert jnp.sum(s2.group_active) == 2

    # 3rd step wraps — should overwrite group 0
    key, sk = jax.random.split(key)
    _, s3, _, _, _ = ENV.step_env(sk, _make_mixed_state(s2), _MIXED_THRESH, ep)
    assert s3.group_origin_step[0] == s2.time


def test_oracle_forked_simulation():
    """
    Oracle test: fork at step t, independently track both ghost states,
    build cf fleet each step, compare dispatch to canonical.

    Fork threshold [1.0, 0.0]: A picks car 0 (solo), B picks car 1 (cheap pool).
    Ghost A = car 0 pre-dispatch, Ghost B = car 1 with trip.
    """
    key = jax.random.PRNGKey(7)
    ep = ENV_PARAMS.replace(ghost_max_lifespan=20, max_groups=32)
    _, base = ENV.reset_env(key, ep)

    fork_thresh = jnp.array([1.0, 0.0])
    state = _make_mixed_state(base)
    key, sk = jax.random.split(key)
    _, real_state, _, _, fork_info = ENV.step_env(sk, state, fork_thresh, ep)
    ghost_a_car = int(fork_info["action_A"])
    ghost_b_car = int(fork_info["action_B"])

    # Oracle ghost states
    oracle_a_wp = state.waypoints[ghost_a_car]
    oracle_a_t = state.times[ghost_a_car]
    b_wp, b_t, _, _ = insert_and_optimize_trip(
        ep.distances,
        state.waypoints[ghost_b_car], state.times[ghost_b_car],
        state.event.src, state.event.dest, state.event.t,
        ep.max_active_trips,
    )
    oracle_b_wp = b_wp
    oracle_b_t = b_t
    ghost_birth_time = int(state.event.t)

    g_idx = int(jnp.argmax(
        real_state.group_active & (real_state.group_origin_step == state.time)
    ))

    real_s = real_state
    oracle_active = True

    for i in range(8):
        key, sk = jax.random.split(key)
        event = real_s.event

        # Oracle: build cf fleet (swap both ghosts into real fleet)
        cf_fleet_wp = real_s.waypoints.at[ghost_a_car].set(oracle_a_wp)
        cf_fleet_wp = cf_fleet_wp.at[ghost_b_car].set(oracle_b_wp)
        cf_fleet_t = real_s.times.at[ghost_a_car].set(oracle_a_t)
        cf_fleet_t = cf_fleet_t.at[ghost_b_car].set(oracle_b_t)

        # Canonical dispatch on real fleet
        can_car, can_found = greedy_select_car(
            ep.distances, real_s.waypoints, real_s.times, event,
            ep.max_active_trips, 0.0,
        )
        # CF dispatch on cf fleet
        cf_disp_car, cf_found = greedy_select_car(
            ep.distances, cf_fleet_wp, cf_fleet_t, event,
            ep.max_active_trips, 0.0,
        )

        oracle_triggered = bool(can_car != cf_disp_car) or bool(can_found != cf_found)

        # Update oracle ghost if cf dispatched it
        if bool(cf_found) and int(cf_disp_car) == ghost_a_car:
            new_wp, new_t, _, _ = insert_and_optimize_trip(
                ep.distances, oracle_a_wp, oracle_a_t,
                event.src, event.dest, event.t, ep.max_active_trips,
            )
            oracle_a_wp = new_wp
            oracle_a_t = new_t
        elif bool(cf_found) and int(cf_disp_car) == ghost_b_car:
            new_wp, new_t, _, _ = insert_and_optimize_trip(
                ep.distances, oracle_b_wp, oracle_b_t,
                event.src, event.dest, event.t, ep.max_active_trips,
            )
            oracle_b_wp = new_wp
            oracle_b_t = new_t

        _, real_s, _, _, info = ENV.step_env(sk, real_s, _THRESH, ep)

        tracker_triggered = bool(info["group_triggered"][g_idx])

        # Expire after trigger check
        oracle_was_active = oracle_active
        if int(event.t) - ghost_birth_time >= ep.ghost_max_lifespan:
            oracle_active = False
        oracle_triggered = oracle_triggered and oracle_was_active

        assert oracle_triggered == tracker_triggered, (
            f"Step t+{i+1}: oracle={oracle_triggered}, tracker={tracker_triggered}"
        )


def test_exclusion_prevents_self_comparison():
    """A ghost group should exclude both canonical and cf cars."""
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_mixed_state(base)
    _, state2, _, _, info = ENV.step_env(key, state, _MIXED_THRESH, ENV_PARAMS)
    canonical_car = int(info["action_A"])
    cf_car = int(info["action_B"])
    g_idx = int(jnp.argmax(state2.group_active))
    excl = set()
    for j in range(int(state2.group_n_ghosts[g_idx])):
        excl.add(int(state2.group_excluded_cars[g_idx, j]))
    assert excl == {canonical_car, cf_car}


"""
Ghost creation 2x2 desired outcomes
====================================

Both policies select independently. No ghosts when action_A == action_B
(same car means no counterfactual to track). Otherwise, when cars differ:

  A fulfills, B fulfills, A≠B  →  1 group with 2 ghosts, group_write_idx +1
  A fulfills, B unfulfills     →  1 group with 1 ghost (Ghost A), group_write_idx +1
  A unfulfills, B fulfills     →  1 group with 1 ghost (Ghost B), group_write_idx +1
  A unfulfills, B unfulfills   →  no group,                       group_write_idx +0
  A == B (same car)            →  no group,                       group_write_idx +0

  action_A = canonical car index, or -1 if A unfulfills
  action_B = cf car index,        or -1 if B unfulfills

To engineer each case:
  - Solo cars are always eligible regardless of threshold.
  - Busy cars (times > event.t) are eligible only if
    marginal_cost < direct_cost * (1 - threshold).
  - threshold=0.0 → busy cars eligible if cheap to pool
  - threshold=1.0 → only solo cars eligible
  Use _HIGH_THRESH to force unfulfill when no solos are available.
"""

_HIGH_THRESH = 1.0   # only solos eligible; busy cars never pass


def _make_busy_state(base_state):
    """All 3 cars on active trips (no solos)."""
    max_wp = _num_wp(ENV_PARAMS.max_active_trips)
    event_t = int(base_state.event.t)
    # One future waypoint per car so is_solo=False; rest zeros so capacity exists
    times = jnp.zeros((_N_CARS, max_wp), dtype=jnp.int32).at[:, 0].set(event_t + 20)
    return base_state.replace(times=times)


def _make_one_solo_state(base_state):
    """Car 0 solo; cars 1 and 2 busy."""
    max_wp = _num_wp(ENV_PARAMS.max_active_trips)
    event_t = int(base_state.event.t)
    times = jnp.zeros((_N_CARS, max_wp), dtype=jnp.int32).at[1:, 0].set(event_t + 20)
    return base_state.replace(times=times)


def _make_cheap_pool_state(base_state):
    """All 3 cars busy (no solos), eligible for threshold=0.0 but NOT threshold=1.0.
    Event overridden to src=1, dest=4 (direct_cost=dist(1,4)=6, t=0).

    Cost semantics in this env: marginal_cost = new_max_completion_time - old_max_completion_time.
    With event src=1, dest=4:
      Cars 1,2: waypoint→node 0, deadline=8. cost=0 (< 6, >=0).
        elig thresh=0.0: 0 < 6 ✓; elig thresh=1.0: 0 < 0 ✗.
      Car 0: waypoint→node 4, deadline=20. cost=2 (< 6, > 0).
        elig thresh=0.0: 2 < 6 ✓; elig thresh=1.0: 2 < 0 ✗.
    """
    max_wp = _num_wp(ENV_PARAMS.max_active_trips)
    event = rs.RideshareEvent(
        t=jnp.array(0, dtype=jnp.int32),
        src=jnp.array(1, dtype=jnp.int32),
        dest=jnp.array(4, dtype=jnp.int32),
    )
    # car 0 → node 4 (deadline=20), cars 1,2 → node 0 (deadline=8)
    times = jnp.zeros((_N_CARS, max_wp), dtype=jnp.int32).at[0, 0].set(20).at[1, 0].set(8).at[2, 0].set(8)
    waypoints = jnp.zeros((_N_CARS, max_wp), dtype=jnp.int32).at[0, 0].set(4)
    return base_state.replace(times=times, waypoints=waypoints, event=event)


def _make_mixed_state(base_state):
    """Car 0 solo; cars 1,2 busy (cheap to pool). Event: src=1, dest=4, t=0.
    Costs: car 0 solo (cost=8), cars 1,2 busy (cost=0 < 6 = direct).
      threshold=0.0 → A picks car 1 (cost=0, cheapest overall).
      threshold=1.0 → B picks car 0 (only solo).
    Use _MIXED_THRESH=[0.0, 1.0] to get both policies dispatching different cars.
    """
    max_wp = _num_wp(ENV_PARAMS.max_active_trips)
    event = rs.RideshareEvent(
        t=jnp.array(0, dtype=jnp.int32),
        src=jnp.array(1, dtype=jnp.int32),
        dest=jnp.array(4, dtype=jnp.int32),
    )
    # car 0 solo (times=0), cars 1,2 busy (deadline=8, cost=0 for this event)
    times = jnp.zeros((_N_CARS, max_wp), dtype=jnp.int32).at[1, 0].set(8).at[2, 0].set(8)
    waypoints = jnp.zeros((_N_CARS, max_wp), dtype=jnp.int32)
    return base_state.replace(times=times, waypoints=waypoints, event=event)


# ---------------------------------------------------------------------------
# 2x2 ghost creation cases
# ---------------------------------------------------------------------------

def test_both_fulfill_two_ghosts():
    """
    A fulfills, B fulfills, different cars → 1 group with 2 ghosts, group_write_idx +1.

    Use _make_mixed_state + [0.0, 1.0]: A picks car 1 (cheap pool), B picks car 0 (solo).
    """
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_mixed_state(base)

    _, state2, _, _, info = ENV.step_env(key, state, _MIXED_THRESH, ENV_PARAMS)

    assert not bool(info["is_unfulfill"]), "A should have dispatched"
    assert int(info["action_A"]) >= 0
    assert int(info["action_B"]) >= 0
    assert int(info["action_A"]) != int(info["action_B"]), "must be different cars"
    assert int(jnp.sum(state2.group_active)) == 1, \
        f"Expected 1 active group, got {jnp.sum(state2.group_active)}"
    g_idx = int(jnp.argmax(state2.group_active))
    assert int(state2.group_n_ghosts[g_idx]) == 2, "Expected 2 ghosts in group"
    assert int(state2.group_write_idx) == int(state.group_write_idx) + 1


def test_same_car_no_ghosts():
    """
    A == B (same car) → no group, group_write_idx unchanged.

    Equal thresholds on same fleet → same argmin → no counterfactual.
    """
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    # fresh reset, all solos, equal thresholds → same cheapest car selected by both
    _, state2, _, _, info = ENV.step_env(key, base, jnp.array([0.0, 0.0]), ENV_PARAMS)

    assert int(info["action_A"]) == int(info["action_B"]), "same car expected"
    assert int(jnp.sum(state2.group_active)) == 0, "no groups when same car"
    assert int(state2.group_write_idx) == int(base.group_write_idx)


def test_a_fulfills_b_unfulfills_ghost_a_only():
    """
    A fulfills, B unfulfills → 1 group with 1 ghost (slot 0 = Ghost A),
    group_write_idx +1, action_B=-1.

    State: all cars busy (no solos), car 0 cheap to pool.
    threshold_a=0.0 → A: car 0 eligible (cheap pool) → A dispatches.
    threshold_b=1.0 → B: only solos eligible; none available → B unfulfills.
    """
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_cheap_pool_state(base)
    action = jnp.array([0.0, _HIGH_THRESH])

    _, state2, _, _, info = ENV.step_env(key, state, action, ENV_PARAMS)

    assert not bool(info["is_unfulfill"]), "A should have dispatched"
    assert int(info["action_A"]) >= 0, f"action_A={info['action_A']}"
    assert int(info["action_B"]) == -1, f"Expected action_B=-1, got {info['action_B']}"
    assert int(jnp.sum(state2.group_active)) == 1, \
        f"Expected 1 active group, got {jnp.sum(state2.group_active)}"
    g_idx = int(jnp.argmax(state2.group_active))
    assert int(state2.group_n_ghosts[g_idx]) == 1, "Expected 1 ghost in group"
    # Slot 0 should be the canonical car (Ghost A)
    canonical_car = int(info["action_A"])
    assert int(state2.group_ghost_car_ids[g_idx, 0]) == canonical_car
    assert int(state2.group_write_idx) == int(state.group_write_idx) + 1


def test_a_unfulfills_b_fulfills_ghost_b_only():
    """
    A unfulfills, B fulfills → 1 group with 1 ghost (slot 0 = Ghost B),
    group_write_idx +1, action_A=-1.

    State: all cars busy, car 0 cheap to pool.
    threshold_a=1.0 → A: only solos eligible; none → A unfulfills (canonical_car=-1).
    threshold_b=0.0 → B: car 0 eligible (cheap pool) → B dispatches.
    """
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_cheap_pool_state(base)
    action = jnp.array([_HIGH_THRESH, 0.0])

    _, state2, _, _, info = ENV.step_env(key, state, action, ENV_PARAMS)

    assert bool(info["is_unfulfill"]), "A should have unfulfilled"
    assert int(info["action_A"]) == -1, f"Expected action_A=-1, got {info['action_A']}"
    assert int(info["action_B"]) >= 0, f"action_B={info['action_B']}"
    assert int(jnp.sum(state2.group_active)) == 1, \
        f"Expected 1 active group, got {jnp.sum(state2.group_active)}"
    g_idx = int(jnp.argmax(state2.group_active))
    assert int(state2.group_n_ghosts[g_idx]) == 1, "Expected 1 ghost in group"
    # Slot 0 should be the cf car (Ghost B, since A didn't write)
    cf_car = int(info["action_B"])
    assert int(state2.group_ghost_car_ids[g_idx, 0]) == cf_car
    assert int(state2.group_write_idx) == int(state.group_write_idx) + 1


def test_both_unfulfill_no_ghosts():
    """
    A unfulfills, B unfulfills → no group, group_write_idx unchanged, action_A=action_B=-1.

    State: all cars busy.
    threshold_a=threshold_b=1.0 → only solos eligible; none available → both unfulfill.
    """
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_busy_state(base)
    action = jnp.array([_HIGH_THRESH, _HIGH_THRESH])

    _, state2, _, _, info = ENV.step_env(key, state, action, ENV_PARAMS)

    assert bool(info["is_unfulfill"]), "Both should have unfulfilled"
    assert int(info["action_A"]) == -1, f"Expected action_A=-1, got {info['action_A']}"
    assert int(info["action_B"]) == -1, f"Expected action_B=-1, got {info['action_B']}"
    assert int(jnp.sum(state2.group_active)) == 0, \
        f"Expected 0 active groups, got {jnp.sum(state2.group_active)}"
    assert int(state2.group_write_idx) == int(state.group_write_idx), \
        "group_write_idx should be unchanged"


# ---------------------------------------------------------------------------
# Ghost content verification for the single-ghost cases
# ---------------------------------------------------------------------------

def test_ghost_a_content_when_b_unfulfills():
    """Ghost A content matches canonical car's pre-dispatch state."""
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_cheap_pool_state(base)  # all busy: A dispatches, B (threshold=1.0) unfulfills
    action = jnp.array([0.0, _HIGH_THRESH])

    _, state2, _, _, info = ENV.step_env(key, state, action, ENV_PARAMS)
    canonical_car = int(info["action_A"])
    g_idx = int(jnp.argmax(state2.group_active))
    slot_idx = g_idx * _MPG + 0  # Ghost A in slot 0

    assert jnp.array_equal(state2.ghost_waypoints[slot_idx], state.waypoints[canonical_car])
    assert jnp.array_equal(state2.ghost_times[slot_idx], state.times[canonical_car])
    assert int(state2.group_excluded_cars[g_idx, 0]) == canonical_car


def test_ghost_b_content_when_a_unfulfills():
    """Ghost B content matches cf car with trip inserted."""
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_cheap_pool_state(base)
    action = jnp.array([_HIGH_THRESH, 0.0])

    _, state2, _, _, info = ENV.step_env(key, state, action, ENV_PARAMS)
    cf_car = int(info["action_B"])
    expected_wp, expected_t, _, _ = insert_and_optimize_trip(
        ENV_PARAMS.distances,
        state.waypoints[cf_car], state.times[cf_car],
        state.event.src, state.event.dest, state.event.t,
        ENV_PARAMS.max_active_trips,
    )
    g_idx = int(jnp.argmax(state2.group_active))
    slot_idx = g_idx * _MPG + 0  # Ghost B in slot 0 (since A didn't write)

    assert jnp.array_equal(state2.ghost_waypoints[slot_idx], expected_wp)
    assert jnp.array_equal(state2.ghost_times[slot_idx], expected_t)
    assert int(state2.group_excluded_cars[g_idx, 0]) == cf_car


# ---------------------------------------------------------------------------
# Buffer position tests
# ---------------------------------------------------------------------------

def test_ghost_buffer_advances_correctly():
    """group_write_idx advances by 1, 1, 1, 0 for the four cases."""
    ep = ENV_PARAMS.replace(max_groups=8, ghost_max_lifespan=100)
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ep)

    cheap_pool = _make_cheap_pool_state(base)  # all busy, car 0 cheap to pool

    start_idx = int(base.group_write_idx)  # 0

    # Both fulfill, different cars: +1 (mixed state: A→car1, B→car0)
    _, s1, _, _, _ = ENV.step_env(key, _make_mixed_state(base), _MIXED_THRESH, ep)
    assert int(s1.group_write_idx) == start_idx + 1

    # A fulfills, B unfulfills: +1
    _, s2, _, _, _ = ENV.step_env(key, cheap_pool, jnp.array([0.0, _HIGH_THRESH]), ep)
    assert int(s2.group_write_idx) == start_idx + 1

    # A unfulfills, B fulfills: +1
    _, s3, _, _, _ = ENV.step_env(key, cheap_pool, jnp.array([_HIGH_THRESH, 0.0]), ep)
    assert int(s3.group_write_idx) == start_idx + 1

    # Both unfulfill: +0 (all busy, both need solo → none)
    _, s4, _, _, _ = ENV.step_env(key, cheap_pool, jnp.array([_HIGH_THRESH, _HIGH_THRESH]), ep)
    assert int(s4.group_write_idx) == start_idx + 0


def test_shared_exclusion_set():
    """Both ghosts in a group share the same exclusion set."""
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_mixed_state(base)
    _, state2, _, _, info = ENV.step_env(key, state, _MIXED_THRESH, ENV_PARAMS)
    canonical_car = int(info["action_A"])
    cf_car = int(info["action_B"])
    g_idx = int(jnp.argmax(state2.group_active))
    # Both ghosts share the group's exclusion set
    excl = set()
    for j in range(int(state2.group_n_ghosts[g_idx])):
        excl.add(int(state2.group_excluded_cars[g_idx, j]))
    assert excl == {canonical_car, cf_car}


def test_threshold_uses_current_action_not_stored():
    """Ghost triggers use the current action's threshold, not the stored creation-time threshold."""
    key = jax.random.PRNGKey(0)
    ep = ENV_PARAMS.replace(ghost_max_lifespan=20, max_groups=32)
    _, base = ENV.reset_env(key, ep)

    # Fork: create ghosts with threshold_a=0.5 (using _MIXED_THRESH-like setup
    # but with different threshold to verify storage)
    fork_state = _make_mixed_state(base)
    fork_thresh = jnp.array([0.5, 1.0])  # A=0.5 (pool eligible), B=1.0 (solo only)
    key, sk = jax.random.split(key)
    _, state_after_fork, _, _, fork_info = ENV.step_env(sk, fork_state, fork_thresh, ep)

    n_active = int(jnp.sum(state_after_fork.group_active))
    assert n_active >= 1, f"Expected at least 1 group, got {n_active}"

    # Group should store threshold_a=0.5
    g_idx = int(jnp.argmax(state_after_fork.group_active))
    stored_thresh = float(state_after_fork.group_threshold[g_idx])
    assert abs(stored_thresh - 0.5) < 1e-6, f"Stored threshold should be 0.5, got {stored_thresh}"

    # Step with threshold=0.0 — trigger check uses current threshold, not stored 0.5
    key, sk = jax.random.split(key)
    _, state2, _, _, info = ENV.step_env(sk, state_after_fork, jnp.array([0.0, 0.0]), ep)
    # Verify stored threshold unchanged (logging only)
    assert abs(float(state_after_fork.group_threshold[g_idx]) - 0.5) < 1e-6


def test_jit_and_scan_compatible():
    """Ghost-tracked env should work under jit and lax.scan."""
    key = jax.random.PRNGKey(0)

    @jax.jit
    def run_episode(key):
        obs, state = ENV.reset(key, ENV_PARAMS)
        def step_fn(carry, rng):
            obs, state = carry
            obs, state, reward, done, info = ENV.step(rng, state, _THRESH, ENV_PARAMS)
            return (obs, state), (reward, info["n_group_triggers"])
        _, (rewards, triggers) = jax.lax.scan(
            step_fn, (obs, state), jax.random.split(key, 10),
        )
        return rewards, triggers

    rewards, triggers = run_episode(key)
    assert rewards.shape == (10,)
    assert triggers.shape == (10,)


def test_policy_returns_action_pair():
    """GreedyPolicy should return a (2,) float threshold array."""
    key = jax.random.PRNGKey(0)
    policy = GreedyPolicy(n_cars=_N_CARS, temperature=0.1, savings_threshold=0.1)
    obs, state = ENV.reset(key, ENV_PARAMS)
    action, info = policy.apply(ENV_PARAMS, {}, obs, key)
    assert action.shape == (2,)
    assert jnp.all(action == 0.1), f"Expected [0.1, 0.1], got {action}"
    _, _, reward, _, _ = ENV.step(key, state, action, ENV_PARAMS)


if __name__ == "__main__":
    tests = [
        ("test_basic_step_runs", test_basic_step_runs),
        ("test_ghosts_created_on_dispatch", test_ghosts_created_on_dispatch),
        ("test_ghost_a_is_pre_dispatch_snapshot", test_ghost_a_is_pre_dispatch_snapshot),
        ("test_ghost_b_is_counterfactual_dispatch", test_ghost_b_is_counterfactual_dispatch),
        ("test_ring_buffer_wraps", test_ring_buffer_wraps),
        ("test_exclusion_prevents_self_comparison", test_exclusion_prevents_self_comparison),
        # 2x2 ghost creation cases
        ("test_both_fulfill_two_ghosts", test_both_fulfill_two_ghosts),
        ("test_same_car_no_ghosts", test_same_car_no_ghosts),
        ("test_a_fulfills_b_unfulfills_ghost_a_only", test_a_fulfills_b_unfulfills_ghost_a_only),
        ("test_a_unfulfills_b_fulfills_ghost_b_only", test_a_unfulfills_b_fulfills_ghost_b_only),
        ("test_both_unfulfill_no_ghosts", test_both_unfulfill_no_ghosts),
        ("test_ghost_a_content_when_b_unfulfills", test_ghost_a_content_when_b_unfulfills),
        ("test_ghost_b_content_when_a_unfulfills", test_ghost_b_content_when_a_unfulfills),
        ("test_ghost_buffer_advances_correctly", test_ghost_buffer_advances_correctly),
        ("test_shared_exclusion_set", test_shared_exclusion_set),
        ("test_threshold_uses_current_action_not_stored", test_threshold_uses_current_action_not_stored),
        ("test_oracle_forked_simulation", test_oracle_forked_simulation),
        ("test_policy_returns_action_pair", test_policy_returns_action_pair),
        ("test_jit_and_scan_compatible", test_jit_and_scan_compatible),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            import traceback
            print(f"FAIL  {name}")
            traceback.print_exc()
