"""
Oracle test for ghost car counterfactual tracking.

The key idea: run two independent simulations forked at step t — one with the
canonical dispatch, one with the counterfactual dispatch. At each future step,
check whether the counterfactual car would have beaten the real fleet. This must
match the ghost tracker's trigger output.

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


def test_basic_step_runs():
    """Smoke test: env with ghost tracking compiles and runs."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    action = jnp.array([0, 1])
    obs2, state2, reward, done, info = ENV.step(key, state, action, ENV_PARAMS)
    assert obs2.shape == obs.shape
    assert "ghost_triggered" in info
    assert "n_ghost_triggers" in info


def test_ghosts_created_on_dispatch():
    """After one dispatch step, exactly 2 ghosts should be active."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    action = jnp.array([0, 1])
    _, state2, _, _, _ = ENV.step(key, state, action, ENV_PARAMS)
    n_active = jnp.sum(state2.ghost_active)
    assert n_active == 2, f"Expected 2 active ghosts, got {n_active}"
    active_types = state2.ghost_type[state2.ghost_active]
    assert jnp.sum(active_types == 0) == 1
    assert jnp.sum(active_types == 1) == 1


def test_ghost_a_is_pre_dispatch_snapshot():
    """Ghost A (type 0) should have the canonical car's pre-dispatch state."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    canonical_car = 0
    action = jnp.array([canonical_car, 1])
    _, state2, _, _, _ = ENV.step(key, state, action, ENV_PARAMS)
    ghost_idx = jnp.argmax(state2.ghost_active & (state2.ghost_type == 0))
    assert jnp.array_equal(state2.ghost_waypoints[ghost_idx], state.waypoints[canonical_car])
    assert jnp.array_equal(state2.ghost_times[ghost_idx], state.times[canonical_car])


def test_ghost_b_is_counterfactual_dispatch():
    """Ghost B (type 1) should have the counterfactual car with the trip added."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    cf_car = 1
    action = jnp.array([0, cf_car])
    _, state2, _, _, _ = ENV.step(key, state, action, ENV_PARAMS)
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
    # Use different cars each step to avoid infeasibility
    key, sk = jax.random.split(key)
    _, s1, _, _, _ = ENV.step_env(sk, state, jnp.array([0, 1]), ep)
    assert jnp.sum(s1.ghost_active) == 2
    key, sk = jax.random.split(key)
    _, s2, _, _, _ = ENV.step_env(sk, s1, jnp.array([1, 2]), ep)
    assert jnp.sum(s2.ghost_active) == 4
    # 3rd step wraps — should overwrite slots 0,1
    key, sk = jax.random.split(key)
    _, s3, _, _, _ = ENV.step_env(sk, s2, jnp.array([2, 0]), ep)
    assert s3.ghost_origin_step[0] == s2.time
    assert s3.ghost_origin_step[1] == s2.time


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
    _, state, _, _, _ = ENV.step_env(sk, state, jnp.array([0, 1]), ep)

    # Fork point
    canonical_car, cf_car = 0, 2
    fork_action = jnp.array([canonical_car, cf_car])
    key, sk = jax.random.split(key)
    _, real_state, _, _, _ = ENV.step_env(sk, state, fork_action, ep)

    # Oracle: cf car gets the trip, track its evolving state
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
    maxint = jnp.iinfo(jnp.int32).max
    oracle_active = True

    for i in range(8):
        key, sk = jax.random.split(key)
        event = real_s.event

        cf_new_wp, cf_new_t, cf_cost, cf_feas = insert_and_optimize_trip(
            ep.distances, oracle_cf_wp, oracle_cf_t,
            event.src, event.dest, event.t, ep.max_active_trips,
        )

        real_costs, real_feasible = compute_real_car_costs(
            ep.distances, real_s.waypoints, real_s.times,
            event, ep.max_active_trips,
        )
        # Ghost wins: ghost beats real fleet excluding cf_car
        masked_real = jnp.where(
            real_feasible & (jnp.arange(_N_CARS) != cf_car),
            real_costs, maxint,
        )
        best_real = jnp.min(masked_real)
        oracle_ghost_wins = oracle_active & bool(cf_feas & (cf_cost < best_real))
        # Canonical wins: cf_car is the best real car overall
        all_real = jnp.where(real_feasible, real_costs, maxint)
        best_real_car = int(jnp.argmin(all_real))
        oracle_canonical_wins = oracle_active & bool(best_real_car == cf_car) & bool(jnp.min(all_real) < maxint)
        oracle_triggered = oracle_ghost_wins | oracle_canonical_wins

        if oracle_ghost_wins:
            oracle_cf_wp = cf_new_wp
            oracle_cf_t = cf_new_t

        _, real_s, _, _, info = ENV.step_env(sk, real_s, jnp.array([0, 1]), ep)
        tracker_triggered = bool(info["ghost_triggered"][ghost_type1_idx])

        # Expire after trigger check (matches tracker: expire_ghosts runs after
        # check_ghost_triggers, setting ghost_active for the NEXT step)
        if int(event.t) - ghost_birth_time >= ep.ghost_max_lifespan:
            oracle_active = False

        assert oracle_triggered == tracker_triggered, (
            f"Step t+{i+1}: oracle={oracle_triggered}, tracker={tracker_triggered}"
        )


def test_exclusion_prevents_self_comparison():
    """A ghost forked from car C should exclude car C from comparison."""
    key = jax.random.PRNGKey(0)
    obs, state = ENV.reset(key, ENV_PARAMS)
    action = jnp.array([0, 1])
    _, state2, _, _, _ = ENV.step(key, state, action, ENV_PARAMS)
    excl = set()
    for i in range(ENV_PARAMS.max_ghosts):
        if state2.ghost_active[i]:
            excl.add(int(state2.ghost_excluded_cars[i][0]))
    assert excl == {0, 1}


def test_jit_and_scan_compatible():
    """Ghost-tracked env should work under jit and lax.scan."""
    key = jax.random.PRNGKey(0)

    @jax.jit
    def run_episode(key):
        obs, state = ENV.reset(key, ENV_PARAMS)
        def step_fn(carry, rng):
            obs, state = carry
            action = jnp.array([0, 1])
            obs, state, reward, done, info = ENV.step(rng, state, action, ENV_PARAMS)
            return (obs, state), (reward, info["n_ghost_triggers"])
        _, (rewards, triggers) = jax.lax.scan(
            step_fn, (obs, state), jax.random.split(key, 10),
        )
        return rewards, triggers

    rewards, triggers = run_episode(key)
    assert rewards.shape == (10,)
    assert triggers.shape == (10,)


def test_policy_returns_action_pair():
    """GreedyPolicy should return a (2,) action array."""
    key = jax.random.PRNGKey(0)
    policy = GreedyPolicy(n_cars=_N_CARS, temperature=0.1, savings_threshold=0.1)
    obs, state = ENV.reset(key, ENV_PARAMS)
    action, info = policy.apply(ENV_PARAMS, {}, obs, key)
    assert action.shape == (2,)
    _, _, reward, _, _ = ENV.step(key, state, action, ENV_PARAMS)


if __name__ == "__main__":
    tests = [
        ("test_basic_step_runs", test_basic_step_runs),
        ("test_ghosts_created_on_dispatch", test_ghosts_created_on_dispatch),
        ("test_ghost_a_is_pre_dispatch_snapshot", test_ghost_a_is_pre_dispatch_snapshot),
        ("test_ghost_b_is_counterfactual_dispatch", test_ghost_b_is_counterfactual_dispatch),
        ("test_ring_buffer_wraps", test_ring_buffer_wraps),
        ("test_exclusion_prevents_self_comparison", test_exclusion_prevents_self_comparison),
        ("test_policy_returns_action_pair", test_policy_returns_action_pair),
        ("test_jit_and_scan_compatible", test_jit_and_scan_compatible),
        ("test_oracle_forked_simulation", test_oracle_forked_simulation),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            import traceback
            print(f"FAIL  {name}")
            traceback.print_exc()
