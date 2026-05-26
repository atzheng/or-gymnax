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
    compute_real_car_costs,
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
    max_ghosts=16,
    ghost_max_lifespan=10,
)

# Threshold=0.0: solo cars always eligible; pool cars eligible if marginal cost
# < direct_cost (which is almost always true for pooling). Deterministic argmin.
_THRESH = jnp.array([0.0, 0.0])


def test_basic_step_runs():
    """Smoke test: env with ghost tracking compiles and runs."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    obs2, state2, reward, done, info = ENV.step(key, state, _THRESH, ENV_PARAMS)
    assert obs2.shape == obs.shape
    assert "ghost_triggered" in info
    assert "n_ghost_triggers" in info
    assert "action_A" in info
    assert "action_B" in info


def test_ghosts_created_on_dispatch():
    """After one dispatch step, exactly 2 ghosts should be active."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    _, state2, _, _, info = ENV.step(key, state, _THRESH, ENV_PARAMS)
    n_active = jnp.sum(state2.ghost_active)
    assert n_active == 2, f"Expected 2 active ghosts, got {n_active}"
    active_types = state2.ghost_type[state2.ghost_active]
    assert jnp.sum(active_types == 0) == 1
    assert jnp.sum(active_types == 1) == 1


def test_ghost_a_is_pre_dispatch_snapshot():
    """Ghost A (type 0) should have the canonical car's pre-dispatch state."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    _, state2, _, _, info = ENV.step(key, state, _THRESH, ENV_PARAMS)
    canonical_car = int(info["action_A"])
    ghost_idx = jnp.argmax(state2.ghost_active & (state2.ghost_type == 0))
    assert jnp.array_equal(state2.ghost_waypoints[ghost_idx], state.waypoints[canonical_car])
    assert jnp.array_equal(state2.ghost_times[ghost_idx], state.times[canonical_car])


def test_ghost_b_is_counterfactual_dispatch():
    """Ghost B (type 1) should have the counterfactual car with the trip added."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    _, state2, _, _, info = ENV.step(key, state, _THRESH, ENV_PARAMS)
    cf_car = int(info["action_B"])
    expected_wp, expected_t, _, _ = insert_and_optimize_trip(
        ENV_PARAMS.distances,
        state.waypoints[cf_car], state.times[cf_car],
        state.event.src, state.event.dest, state.event.t,
        ENV_PARAMS.max_active_trips,
    )
    ghost_idx = jnp.argmax(state2.ghost_active & (state2.ghost_type == 1))
    assert jnp.array_equal(state2.ghost_waypoints[ghost_idx], expected_wp)
    assert jnp.array_equal(state2.ghost_times[ghost_idx], expected_t)


def test_ring_buffer_wraps():
    """When buffer fills, ring buffer should overwrite oldest slots."""
    ep = ENV_PARAMS.replace(max_ghosts=4, ghost_max_lifespan=100)
    key = jax.random.PRNGKey(0)
    _, state = ENV.reset_env(key, ep)
    key, sk = jax.random.split(key)
    _, s1, _, _, _ = ENV.step_env(sk, state, _THRESH, ep)
    assert jnp.sum(s1.ghost_active) == 2
    key, sk = jax.random.split(key)
    _, s2, _, _, _ = ENV.step_env(sk, s1, _THRESH, ep)
    assert jnp.sum(s2.ghost_active) == 4
    # 3rd step wraps — should overwrite slots 0,1
    key, sk = jax.random.split(key)
    _, s3, _, _, _ = ENV.step_env(sk, s2, _THRESH, ep)
    assert s3.ghost_origin_step[0] == s2.time
    assert s3.ghost_origin_step[1] == s2.time


def _oracle_trigger_check(oracle_cf_wp, oracle_cf_t, real_s, cf_car, threshold, ep):
    """Compute oracle ghost-B trigger for one step under the given threshold."""
    maxint = jnp.iinfo(jnp.int32).max
    event = real_s.event
    direct_cost = int(ep.distances[event.src, event.dest])

    cf_new_wp, cf_new_t, cf_cost, cf_feas = insert_and_optimize_trip(
        ep.distances, oracle_cf_wp, oracle_cf_t,
        event.src, event.dest, event.t, ep.max_active_trips,
    )
    real_costs, real_feasible = compute_real_car_costs(
        ep.distances, real_s.waypoints, real_s.times, event, ep.max_active_trips,
    )
    # Ghost eligibility: solo or passes threshold vs direct cost
    cf_is_solo = bool(jnp.all(oracle_cf_t <= event.t))
    cf_eligible = cf_is_solo or (int(cf_cost) < direct_cost * (1 - threshold))

    real_is_solo = jnp.all(real_s.times <= event.t, axis=1)
    real_eligible = real_feasible & (real_is_solo | (real_costs < direct_cost * (1 - threshold)))

    # Ghost wins: ghost beats eligible real fleet (excluding cf_car)
    masked_real = jnp.where(
        real_eligible & (jnp.arange(_N_CARS) != cf_car), real_costs, maxint,
    )
    oracle_ghost_wins = bool(cf_feas) and cf_eligible and bool(cf_cost < jnp.min(masked_real))

    # Canonical wins: cf_car is best eligible real car
    eligible_real_costs = jnp.where(real_eligible, real_costs, maxint)
    best_real_car = int(jnp.argmin(eligible_real_costs))
    oracle_canonical_wins = (
        bool(best_real_car == cf_car) and bool(jnp.min(eligible_real_costs) < maxint)
    )

    oracle_triggered = oracle_ghost_wins or oracle_canonical_wins
    return oracle_triggered, oracle_ghost_wins, cf_new_wp, cf_new_t


def test_oracle_forked_simulation():
    """
    Oracle test: fork at step t, independently track the counterfactual car's
    evolving state, and verify ghost trigger pattern matches.
    """
    key = jax.random.PRNGKey(7)
    ep = ENV_PARAMS.replace(ghost_max_lifespan=20, max_ghosts=64)
    obs, state = ENV.reset_env(key, ep)

    # Setup step
    key, sk = jax.random.split(key)
    _, state, _, _, _ = ENV.step_env(sk, state, _THRESH, ep)

    # Fork point: let the env choose cars, then read which were chosen
    key, sk = jax.random.split(key)
    _, real_state, _, _, fork_info = ENV.step_env(sk, state, _THRESH, ep)
    canonical_car = int(fork_info["action_A"])
    cf_car = int(fork_info["action_B"])

    # Oracle: cf car gets the trip at fork, track its evolving state
    cf_wp, cf_t, _, _ = insert_and_optimize_trip(
        ep.distances,
        state.waypoints[cf_car], state.times[cf_car],
        state.event.src, state.event.dest, state.event.t,
        ep.max_active_trips,
    )
    oracle_cf_wp = cf_wp
    oracle_cf_t = cf_t
    ghost_birth_time = int(state.event.t)

    ghost_type1_idx = jnp.argmax(
        real_state.ghost_active & (real_state.ghost_type == 1)
        & (real_state.ghost_origin_step == state.time)
    )

    real_s = real_state
    oracle_active = True

    for i in range(8):
        key, sk = jax.random.split(key)
        event = real_s.event

        oracle_triggered, oracle_ghost_wins, cf_new_wp, cf_new_t = _oracle_trigger_check(
            oracle_cf_wp, oracle_cf_t, real_s, cf_car, 0.0, ep,
        )

        if oracle_ghost_wins:
            oracle_cf_wp = cf_new_wp
            oracle_cf_t = cf_new_t

        _, real_s, _, _, info = ENV.step_env(sk, real_s, _THRESH, ep)
        tracker_triggered = bool(info["ghost_triggered"][ghost_type1_idx])

        # Expire after trigger check (matches tracker behaviour)
        if int(event.t) - ghost_birth_time >= ep.ghost_max_lifespan:
            oracle_active = False
        oracle_triggered = oracle_triggered and oracle_active

        assert oracle_triggered == tracker_triggered, (
            f"Step t+{i+1}: oracle={oracle_triggered}, tracker={tracker_triggered}"
        )


def test_exclusion_prevents_self_comparison():
    """A ghost forked from car C should exclude car C from comparison."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    _, state2, _, _, info = ENV.step(key, state, _THRESH, ENV_PARAMS)
    canonical_car = int(info["action_A"])
    cf_car = int(info["action_B"])
    excl = set()
    for i in range(ENV_PARAMS.max_ghosts):
        if state2.ghost_active[i]:
            excl.add(int(state2.ghost_excluded_cars[i][0]))
    assert excl == {canonical_car, cf_car}


"""
Ghost creation 2x2 desired outcomes
====================================

Both policies select independently from the full fleet. They may pick the
same car (action_A == action_B), which is a valid case meaning both policies
agree.

  A fulfills, B fulfills  →  Ghost A (type 0) + Ghost B (type 1), write_idx +2
  A fulfills, B unfulfills →  Ghost A only (type 0),               write_idx +1
  A unfulfills, B fulfills →  Ghost B only (type 1),               write_idx +1
  A unfulfills, B unfulfills → no ghosts,                          write_idx +0

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


# ---------------------------------------------------------------------------
# 2x2 ghost creation cases
# ---------------------------------------------------------------------------

def test_both_fulfill_two_ghosts():
    """
    A fulfills, B fulfills → Ghost A + Ghost B, write_idx +2.

    With threshold=[0.0, 0.0] and all-solo cars, both policies pick the same
    cheapest car. Ghost A = that car pre-dispatch; Ghost B = that car with trip.
    """
    key = jax.random.PRNGKey(0)
    _, state = ENV.reset_env(key, ENV_PARAMS)  # all cars solo

    _, state2, _, _, info = ENV.step_env(key, state, jnp.array([0.0, 0.0]), ENV_PARAMS)

    assert not bool(info["is_unfulfill"]), "A should have dispatched"
    assert int(info["action_A"]) >= 0
    assert int(info["action_B"]) >= 0
    assert int(jnp.sum(state2.ghost_active)) == 2, \
        f"Expected 2 active ghosts, got {jnp.sum(state2.ghost_active)}"
    active_types = state2.ghost_type[state2.ghost_active]
    assert int(jnp.sum(active_types == 0)) == 1, "Expected 1 Ghost A"
    assert int(jnp.sum(active_types == 1)) == 1, "Expected 1 Ghost B"
    assert int(state2.ghost_write_idx) == int(state.ghost_write_idx) + 2


def test_a_fulfills_b_unfulfills_ghost_a_only():
    """
    A fulfills, B unfulfills → Ghost A only (type 0), write_idx +1, action_B=-1.

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
    assert int(jnp.sum(state2.ghost_active)) == 1, \
        f"Expected 1 active ghost, got {jnp.sum(state2.ghost_active)}"
    ghost_idx = int(jnp.argmax(state2.ghost_active))
    assert int(state2.ghost_type[ghost_idx]) == 0, "Expected Ghost A (type 0)"
    assert int(state2.ghost_write_idx) == int(state.ghost_write_idx) + 1


def test_a_unfulfills_b_fulfills_ghost_b_only():
    """
    A unfulfills, B fulfills → Ghost B only (type 1), write_idx +1, action_A=-1.

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
    assert int(jnp.sum(state2.ghost_active)) == 1, \
        f"Expected 1 active ghost, got {jnp.sum(state2.ghost_active)}"
    ghost_idx = int(jnp.argmax(state2.ghost_active))
    assert int(state2.ghost_type[ghost_idx]) == 1, "Expected Ghost B (type 1)"
    assert int(state2.ghost_write_idx) == int(state.ghost_write_idx) + 1


def test_both_unfulfill_no_ghosts():
    """
    A unfulfills, B unfulfills → no ghosts, write_idx unchanged, action_A=action_B=-1.

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
    assert int(jnp.sum(state2.ghost_active)) == 0, \
        f"Expected 0 active ghosts, got {jnp.sum(state2.ghost_active)}"
    assert int(state2.ghost_write_idx) == int(state.ghost_write_idx), \
        "write_idx should be unchanged"


# ---------------------------------------------------------------------------
# Ghost content verification for the single-ghost cases
# ---------------------------------------------------------------------------

def test_ghost_a_content_when_b_unfulfills():
    """Ghost A content matches canonical car's pre-dispatch state."""
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ENV_PARAMS)
    state = _make_one_solo_state(base)
    action = jnp.array([0.0, _HIGH_THRESH])

    _, state2, _, _, info = ENV.step_env(key, state, action, ENV_PARAMS)
    canonical_car = int(info["action_A"])
    ghost_idx = int(jnp.argmax(state2.ghost_active))

    assert jnp.array_equal(state2.ghost_waypoints[ghost_idx], state.waypoints[canonical_car])
    assert jnp.array_equal(state2.ghost_times[ghost_idx], state.times[canonical_car])
    assert int(state2.ghost_excluded_cars[ghost_idx][0]) == canonical_car


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
    ghost_idx = int(jnp.argmax(state2.ghost_active))

    assert jnp.array_equal(state2.ghost_waypoints[ghost_idx], expected_wp)
    assert jnp.array_equal(state2.ghost_times[ghost_idx], expected_t)
    assert int(state2.ghost_excluded_cars[ghost_idx][0]) == cf_car


# ---------------------------------------------------------------------------
# Buffer position tests (replacing same-car skip tests)
# ---------------------------------------------------------------------------

def test_ghost_buffer_advances_correctly():
    """write_idx advances by 2, 1, 1, 0 for the four cases."""
    ep = ENV_PARAMS.replace(max_ghosts=16, ghost_max_lifespan=100)
    key = jax.random.PRNGKey(0)
    _, base = ENV.reset_env(key, ep)

    cheap_pool = _make_cheap_pool_state(base)  # all busy, car 0 cheap to pool
    all_solo = base  # fresh reset: all solo

    start_idx = int(base.ghost_write_idx)  # 0

    # Both fulfill: +2 (all solo, both pick same cheapest car)
    _, s1, _, _, _ = ENV.step_env(key, all_solo, jnp.array([0.0, 0.0]), ep)
    assert int(s1.ghost_write_idx) == start_idx + 2

    # A fulfills, B unfulfills: +1 (car 0 poolable; threshold_b=1.0 needs solo → none)
    _, s2, _, _, _ = ENV.step_env(key, cheap_pool, jnp.array([0.0, _HIGH_THRESH]), ep)
    assert int(s2.ghost_write_idx) == start_idx + 1

    # A unfulfills, B fulfills: +1 (threshold_a=1.0 needs solo → none; car 0 poolable for B)
    _, s3, _, _, _ = ENV.step_env(key, cheap_pool, jnp.array([_HIGH_THRESH, 0.0]), ep)
    assert int(s3.ghost_write_idx) == start_idx + 1

    # Both unfulfill: +0 (all busy, both need solo → none)
    _, s4, _, _, _ = ENV.step_env(key, cheap_pool, jnp.array([_HIGH_THRESH, _HIGH_THRESH]), ep)
    assert int(s4.ghost_write_idx) == start_idx + 0


def test_jit_and_scan_compatible():
    """Ghost-tracked env should work under jit and lax.scan."""
    key = jax.random.PRNGKey(0)

    @jax.jit
    def run_episode(key):
        obs, state = ENV.reset(key, ENV_PARAMS)
        def step_fn(carry, rng):
            obs, state = carry
            obs, state, reward, done, info = ENV.step(rng, state, _THRESH, ENV_PARAMS)
            return (obs, state), (reward, info["n_ghost_triggers"])
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
        ("test_a_fulfills_b_unfulfills_ghost_a_only", test_a_fulfills_b_unfulfills_ghost_a_only),
        ("test_a_unfulfills_b_fulfills_ghost_b_only", test_a_unfulfills_b_fulfills_ghost_b_only),
        ("test_both_unfulfill_no_ghosts", test_both_unfulfill_no_ghosts),
        ("test_ghost_a_content_when_b_unfulfills", test_ghost_a_content_when_b_unfulfills),
        ("test_ghost_b_content_when_a_unfulfills", test_ghost_b_content_when_a_unfulfills),
        ("test_ghost_buffer_advances_correctly", test_ghost_buffer_advances_correctly),
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
