import jax
import jax.numpy as jnp
import pytest

import or_gymnax.rideshare as rs
import or_gymnax.rideshare_pool as rsp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_state(n_cars, max_active_trips, key, event):
    """Create a state with idle cars at node 0."""
    wp = rsp._num_wp(max_active_trips)
    return rsp.EnvState(
        time=0,
        waypoints=jnp.zeros((n_cars, wp), dtype=jnp.int32),
        times=jnp.zeros((n_cars, wp), dtype=jnp.int32),
        key=key,
        event=event,
    )


def _uniform_distances(n_nodes, dist=10):
    d = jnp.ones((n_nodes, n_nodes), dtype=jnp.int32) * dist
    return d.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0)


def _uniform_env_params(n_cars=2, n_nodes=4, n_events=3, max_active_trips=2, dist=10):
    events = rs.RideshareEvent(
        t=jnp.arange(n_events, dtype=jnp.int32) * 100,
        src=jnp.zeros(n_events, dtype=jnp.int32),
        dest=jnp.ones(n_events, dtype=jnp.int32),
    )
    return rsp.EnvParams(
        events=events,
        distances=_uniform_distances(n_nodes, dist),
        n_cars=n_cars,
        max_active_trips=max_active_trips,
    )


def _make_event(t=0, src=0, dest=1):
    return rs.RideshareEvent(
        t=jnp.array(t, dtype=jnp.int32),
        src=jnp.array(src, dtype=jnp.int32),
        dest=jnp.array(dest, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# admissible_sequences
# ---------------------------------------------------------------------------

class TestAdmissibleSequences:
    def test_m2_count(self):
        seqs = rsp.admissible_sequences(2)
        # 4! / (2*2) = 6 permutations with P_i < D_i
        assert seqs.shape == (6, 4)

    def test_m2_all_respect_precedence(self):
        seqs = rsp.admissible_sequences(2)
        for s in seqs:
            s = list(s)
            assert s.index(0) < s.index(1), "P0 must precede D0"
            assert s.index(2) < s.index(3), "P1 must precede D1"

    def test_m3_all_respect_precedence(self):
        seqs = rsp.admissible_sequences(3)
        for s in seqs:
            s = list(s)
            for i in range(3):
                assert s.index(2 * i) < s.index(2 * i + 1), f"P{i} must precede D{i}"


# ---------------------------------------------------------------------------
# optimize_waypoints
# ---------------------------------------------------------------------------

class TestOptimizeWaypoints:
    def test_single_active_trip(self):
        """With one active trip, optimizer finds direct route."""
        distances = jnp.array([
            [0, 5, 3], [5, 0, 2], [3, 2, 0],
        ], dtype=jnp.int32)
        waypoints = jnp.array([1, 2, 0, 0], dtype=jnp.int32)
        max_t = jnp.iinfo(jnp.int32).max
        times = jnp.array([max_t, max_t, 0, 0], dtype=jnp.int32)

        new_times, cost = rsp.optimize_waypoints(
            distances, waypoints, times, start_waypoint=0, start_time=10,
            max_active_trips=2,
        )
        # Route: 0→1 (dist=5) → 2 (dist=2), total=7
        assert new_times[0] == 15
        assert new_times[1] == 17
        assert cost == 7

    def test_two_active_trips_finds_optimal(self):
        distances = jnp.array([
            [0, 1, 10, 1], [1, 0, 1, 10],
            [10, 1, 0, 1], [1, 10, 1, 0],
        ], dtype=jnp.int32)
        waypoints = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        max_t = jnp.iinfo(jnp.int32).max
        times = jnp.full(4, max_t, dtype=jnp.int32)

        new_times, cost = rsp.optimize_waypoints(
            distances, waypoints, times, start_waypoint=0, start_time=0,
            max_active_trips=2,
        )
        # Best: 0→P0(0)→D0(1)→P1(2)→D1(3) = 0+1+1+1 = 3
        assert cost == 3

    def test_all_inactive_returns_original_times(self):
        distances = jnp.ones((4, 4), dtype=jnp.int32) - jnp.eye(4, dtype=jnp.int32)
        waypoints = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        times = jnp.array([1, 2, 3, 4], dtype=jnp.int32)

        new_times, _ = rsp.optimize_waypoints(
            distances, waypoints, times, start_waypoint=0, start_time=10,
            max_active_trips=2,
        )
        assert jnp.array_equal(new_times, times)

    def test_pickup_before_dropoff_constraint(self):
        distances = jnp.array([
            [0, 2, 3, 1], [2, 0, 1, 3],
            [3, 1, 0, 2], [1, 3, 2, 0],
        ], dtype=jnp.int32)
        waypoints = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        max_t = jnp.iinfo(jnp.int32).max
        times = jnp.full(4, max_t, dtype=jnp.int32)

        new_times, _ = rsp.optimize_waypoints(
            distances, waypoints, times, start_waypoint=0, start_time=0,
            max_active_trips=2,
        )
        assert new_times[0] < new_times[1], "P0 before D0"
        assert new_times[2] < new_times[3], "P1 before D1"

    def test_mixed_active_inactive(self):
        """Inactive trip times unchanged; active trip times updated."""
        distances = jnp.array([
            [0, 5, 3], [5, 0, 2], [3, 2, 0],
        ], dtype=jnp.int32)
        waypoints = jnp.array([1, 2, 0, 0], dtype=jnp.int32)
        max_t = jnp.iinfo(jnp.int32).max
        times = jnp.array([max_t, max_t, 3, 5], dtype=jnp.int32)

        new_times, _ = rsp.optimize_waypoints(
            distances, waypoints, times, start_waypoint=0, start_time=10,
            max_active_trips=2,
        )
        assert new_times[2] == 3
        assert new_times[3] == 5
        assert new_times[0] > 10
        assert new_times[1] > new_times[0]


# ---------------------------------------------------------------------------
# insert_and_optimize_trip
# ---------------------------------------------------------------------------

class TestInsertAndOptimizeTrip:
    def test_insert_into_empty_car(self):
        distances = jnp.array([
            [0, 5, 3], [5, 0, 2], [3, 2, 0],
        ], dtype=jnp.int32)
        waypoints = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        times = jnp.zeros(4, dtype=jnp.int32)

        new_wps, new_times, cost, feasible = rsp.insert_and_optimize_trip(
            distances, waypoints, times, pickup_id=1, dropoff_id=2,
            time=10, max_active_trips=2,
        )
        assert feasible
        assert new_wps[0] == 1
        assert new_wps[1] == 2
        assert new_times[0] == 15
        assert new_times[1] == 17

    def test_insert_when_full_returns_infeasible(self):
        distances = jnp.ones((4, 4), dtype=jnp.int32) - jnp.eye(4, dtype=jnp.int32)
        waypoints = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        times = jnp.array([100, 200, 150, 250], dtype=jnp.int32)

        _, _, cost, feasible = rsp.insert_and_optimize_trip(
            distances, waypoints, times, pickup_id=0, dropoff_id=1,
            time=50, max_active_trips=2,
        )
        assert not feasible
        assert cost == jnp.iinfo(jnp.int32).max

    def test_identical_trip_zero_marginal_cost(self):
        """Same-route pool should cost nothing extra."""
        n = 4
        distances = _uniform_distances(n, dist=10)
        waypoints = jnp.array([0, 1, 0, 0], dtype=jnp.int32)
        times = jnp.array([15, 25, 0, 0], dtype=jnp.int32)

        _, _, cost, feasible = rsp.insert_and_optimize_trip(
            distances, waypoints, times, pickup_id=0, dropoff_id=1,
            time=5, max_active_trips=2,
        )
        assert feasible
        assert cost == 0

    def test_detour_has_positive_marginal_cost(self):
        distances = jnp.array([
            [0, 1, 100], [1, 0, 100], [100, 100, 0],
        ], dtype=jnp.int32)
        waypoints = jnp.array([0, 1, 0, 0], dtype=jnp.int32)
        times = jnp.array([11, 12, 0, 0], dtype=jnp.int32)

        _, _, cost, feasible = rsp.insert_and_optimize_trip(
            distances, waypoints, times, pickup_id=2, dropoff_id=0,
            time=10, max_active_trips=2,
        )
        assert feasible
        assert cost > 0


# ---------------------------------------------------------------------------
# step_env state transitions
# NOTE: We call step_env directly to avoid gymnax's JIT-wrapped step()
# which has cache issues with pytree_node=False distances arrays.
# ---------------------------------------------------------------------------

class TestStepEnv:
    def test_dispatch_updates_waypoints_and_times(self):
        env = rsp.RidesharePoolDispatch(n_cars=2, n_nodes=4, n_events=3)
        params = _uniform_env_params()
        key = jax.random.PRNGKey(42)
        state = _empty_state(2, params.max_active_trips, key, _make_event())

        obs, new_state, reward, done, info = env.step_env(key, state, 0, params)

        assert new_state.waypoints[0, 0] == 0  # Pickup
        assert new_state.waypoints[0, 1] == 1  # Dropoff
        assert jnp.array_equal(new_state.waypoints[1], state.waypoints[1])
        assert not info["is_unfulfill"]

    def test_unfulfill_leaves_state_unchanged(self):
        env = rsp.RidesharePoolDispatch(n_cars=2, n_nodes=4, n_events=3)
        params = _uniform_env_params()
        key = jax.random.PRNGKey(42)
        state = _empty_state(2, params.max_active_trips, key, _make_event())

        obs, new_state, reward, done, info = env.step_env(key, state, -1, params)

        assert jnp.array_equal(new_state.waypoints, state.waypoints)
        assert jnp.array_equal(new_state.times, state.times)
        assert reward == 0.0
        assert info["is_unfulfill"] == True

    def test_dispatch_reward_formula(self):
        """Reward = direct_cost * (1 + profit_margin) - marginal_cost."""
        env = rsp.RidesharePoolDispatch(n_cars=2, n_nodes=4, n_events=3)
        params = _uniform_env_params()
        key = jax.random.PRNGKey(42)
        state = _empty_state(2, params.max_active_trips, key, _make_event())

        _, _, reward, _, info = env.step_env(key, state, 0, params)

        direct_cost = params.distances[0, 1]  # 10
        expected = direct_cost * (1 + params.profit_margin) - info["marginal_cost"]
        assert jnp.isclose(reward, expected)

    def test_infeasible_dispatch_falls_back_to_unfulfill(self):
        env = rsp.RidesharePoolDispatch(n_cars=2, n_nodes=4, n_events=3)
        params = _uniform_env_params()
        key = jax.random.PRNGKey(42)
        state = rsp.EnvState(
            time=0,
            waypoints=jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=jnp.int32),
            times=jnp.array([[100, 200, 150, 250], [100, 200, 150, 250]], dtype=jnp.int32),
            key=key,
            event=_make_event(t=50),
        )

        _, new_state, reward, _, info = env.step_env(key, state, 0, params)

        assert info["is_unfulfill"] == True
        assert reward == 0.0

    def test_time_advances_each_step(self):
        env = rsp.RidesharePoolDispatch(n_cars=2, n_nodes=4, n_events=3)
        params = _uniform_env_params()
        key = jax.random.PRNGKey(42)
        state = _empty_state(2, params.max_active_trips, key, _make_event())

        _, state1, _, _, _ = env.step_env(key, state, 0, params)
        assert state1.time == 1

        _, state2, _, _, _ = env.step_env(key, state1, -1, params)
        assert state2.time == 2


# ---------------------------------------------------------------------------
# Key advancement bug
# ---------------------------------------------------------------------------

class TestKeyAdvancement:
    def test_key_advances_on_unfulfill(self):
        env = rsp.RidesharePoolDispatch(n_cars=2, n_nodes=4, n_events=3)
        params = _uniform_env_params()
        key = jax.random.PRNGKey(42)
        state = _empty_state(2, params.max_active_trips, key, _make_event())

        _, state1, _, _, _ = env.step_env(key, state, -1, params)
        _, state2, _, _, _ = env.step_env(key, state1, -1, params)

        assert not jnp.array_equal(state1.key, state.key)
        assert not jnp.array_equal(state2.key, state1.key)

    def test_key_advances_on_dispatch(self):
        env = rsp.RidesharePoolDispatch(n_cars=2, n_nodes=4, n_events=3)
        params = _uniform_env_params()
        key = jax.random.PRNGKey(42)
        state = _empty_state(2, params.max_active_trips, key, _make_event())

        _, state1, _, _, _ = env.step_env(key, state, 0, params)

        assert not jnp.array_equal(state1.key, state.key)


# ---------------------------------------------------------------------------
# GreedyPolicy
# ---------------------------------------------------------------------------

class TestGreedyPolicy:
    def test_selects_closest_empty_car(self):
        n_cars = 3
        env = rsp.RidesharePoolDispatch(n_cars=n_cars, n_nodes=4, n_events=3)
        distances = jnp.array([
            [0, 1, 5, 10], [1, 0, 4, 9],
            [5, 4, 0, 5], [10, 9, 5, 0],
        ], dtype=jnp.int32)
        env_params = rsp.EnvParams(
            events=rs.RideshareEvent(
                t=jnp.array([0, 100, 200], dtype=jnp.int32),
                src=jnp.array([0, 0, 0], dtype=jnp.int32),
                dest=jnp.array([1, 1, 1], dtype=jnp.int32),
            ),
            distances=distances, n_cars=n_cars, max_active_trips=2,
        )
        state = rsp.EnvState(
            time=0,
            waypoints=jnp.array([
                [0, 0, 0, 0], [2, 2, 2, 2], [3, 3, 3, 3],
            ], dtype=jnp.int32),
            times=jnp.zeros((3, 4), dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
            event=_make_event(),
        )
        obs = env.get_obs(state)
        policy = rsp.GreedyPolicy(n_cars=n_cars, temperature=0.01)
        action, _ = policy.apply(env_params, {}, obs, jax.random.PRNGKey(1))
        assert action == 0  # Car 0 at source node 0

    def test_savings_threshold_prevents_pooling(self):
        n_cars = 2
        env = rsp.RidesharePoolDispatch(n_cars=n_cars, n_nodes=5, n_events=3)
        distances = jnp.array([
            [0, 1, 100, 100, 100], [1, 0, 100, 100, 100],
            [100, 100, 0, 1, 100], [100, 100, 1, 0, 100],
            [100, 100, 100, 100, 0],
        ], dtype=jnp.int32)
        env_params = rsp.EnvParams(
            events=rs.RideshareEvent(
                t=jnp.array([0, 50, 100], dtype=jnp.int32),
                src=jnp.array([0, 0, 0], dtype=jnp.int32),
                dest=jnp.array([1, 1, 1], dtype=jnp.int32),
            ),
            distances=distances, n_cars=n_cars, max_active_trips=2,
        )
        state = rsp.EnvState(
            time=0,
            waypoints=jnp.array([[2, 3, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32),
            times=jnp.array([[100, 200, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32),
            key=jax.random.PRNGKey(0),
            event=_make_event(t=50),
        )
        obs = env.get_obs(state)
        policy = rsp.GreedyPolicy(
            n_cars=n_cars, temperature=0.01, savings_threshold=0.99
        )
        action, _ = policy.apply(env_params, {}, obs, jax.random.PRNGKey(1))
        assert action == 1  # Idle car preferred over expensive pool


# ---------------------------------------------------------------------------
# num_active_trips
# ---------------------------------------------------------------------------

class TestNumActiveTrips:
    def test_no_active_trips(self):
        result = rsp.num_active_trips(
            jnp.zeros((2, 4), dtype=jnp.int32),
            jnp.zeros((2, 4), dtype=jnp.int32),
            current_time=10,
        )
        assert jnp.array_equal(result, jnp.array([0, 0]))

    def test_all_active(self):
        result = rsp.num_active_trips(
            jnp.ones((2, 4), dtype=jnp.int32),
            jnp.array([[100, 200, 150, 250], [110, 210, 160, 260]], dtype=jnp.int32),
            current_time=10,
        )
        assert jnp.array_equal(result, jnp.array([2, 2]))

    def test_mixed_active(self):
        result = rsp.num_active_trips(
            jnp.ones((1, 4), dtype=jnp.int32),
            jnp.array([[5, 8, 15, 20]], dtype=jnp.int32),
            current_time=10,
        )
        assert result[0] == 1


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_multiple_dispatches(self):
        env = rsp.RidesharePoolDispatch(n_cars=2, n_nodes=4, n_events=3)
        params = _uniform_env_params()
        key = jax.random.PRNGKey(42)
        state = _empty_state(2, params.max_active_trips, key, _make_event())

        obs1, state1, _, _, info1 = env.step_env(key, state, 0, params)
        assert not info1["is_unfulfill"]
        assert jnp.any(state1.times[0] > 0)

    def test_obs_roundtrip_after_step(self):
        env = rsp.RidesharePoolDispatch(n_cars=2, n_nodes=4, n_events=3)
        params = _uniform_env_params()
        key = jax.random.PRNGKey(42)
        state = _empty_state(2, params.max_active_trips, key, _make_event())

        obs1, state1, _, _, _ = env.step_env(key, state, 0, params)

        event_r, wps_r, times_r = rsp.obs_to_state(
            2, rsp._num_wp(params.max_active_trips), obs1
        )
        assert jnp.array_equal(wps_r, state1.waypoints)
        assert jnp.array_equal(times_r, state1.times)
        assert event_r.t == state1.event.t
        assert event_r.src == state1.event.src
        assert event_r.dest == state1.event.dest
