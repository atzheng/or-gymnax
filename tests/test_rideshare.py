import jax
import jax.numpy as jnp
import pytest

import or_gymnax.rideshare as rs
import or_gymnax.rideshare_pool as rsp


def test_obs_state_inverse_rideshare():
    """Test that get_obs and obs_to_state are inverses for RideshareDispatch"""
    n_cars = 3
    env = rs.RideshareDispatch(n_cars=n_cars, n_nodes=4, n_events=1)

    # Create a test state
    state = rs.EnvState(
        time=0,
        locations=jnp.array([1, 2, 3]),
        times=jnp.array([10, 20, 30]),
        key=jax.random.PRNGKey(0),
        event=rs.RideshareEvent(
            t=jnp.array(5), src=jnp.array(2), dest=jnp.array(4)
        ),
    )

    # Convert state -> obs -> state components
    obs = env.get_obs(state)
    event, locations, times = rs.obs_to_state(n_cars, obs)

    # Check that each component matches
    assert jnp.array_equal(event.t, state.event.t)
    assert jnp.array_equal(event.src, state.event.src)
    assert jnp.array_equal(event.dest, state.event.dest)
    assert jnp.array_equal(locations, state.locations)
    assert jnp.array_equal(times, state.times)


def test_obs_state_inverse_pool():
    """Test that get_obs and obs_to_state are inverses for RidesharePoolDispatch"""
    n_cars = 2
    max_waypoints = 4
    env = rsp.RidesharePoolDispatch(
        n_cars=n_cars,
        n_nodes=9,
        n_events=1,
    )

    # Create a test state with waypoints
    state = rsp.EnvState(
        time=0,
        waypoints=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        times=jnp.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        key=jax.random.PRNGKey(0),
        event=rs.RideshareEvent(
            t=jnp.array(5), src=jnp.array(2), dest=jnp.array(4)
        ),
    )

    # Convert state -> obs -> state components
    obs = env.get_obs(state)
    event, waypoints, times = rsp.obs_to_state(n_cars, max_waypoints, obs)

    # Check that each component matches
    assert jnp.array_equal(event.t, state.event.t)
    assert jnp.array_equal(event.src, state.event.src)
    assert jnp.array_equal(event.dest, state.event.dest)
    assert jnp.array_equal(waypoints, state.waypoints)
    assert jnp.array_equal(times, state.times)


# def test_optimize_waypoints():
#     # Test 1: Basic optimal ordering with asymmetric distances
#     distances = jnp.array(
#         [
#             [0, 1, 5, 2],  # From node 0
#             [1, 0, 3, 4],  # From node 1
#             [5, 3, 0, 1],  # From node 2
#             [2, 4, 1, 0],  # From node 3
#         ]
#     )
#     waypoints = jnp.array([0, 1, 2, 3])  # P1, D1, P2, D2
#     times = jnp.array([20, 25, 22, 27])  # All in future
#     current_time = 15

#     new_times, cost = rsp.optimize_waypoints(
#         distances,
#         waypoints,
#         times,
#         0,
#         current_time,
#         max_active_trips=2,
#     )
#     # Should choose optimal path respecting P≺D constraints
#     assert new_times[0] < new_times[1]  # P1 before D1
#     assert new_times[2] < new_times[3]  # P2 before D2
#     # Verify optimal cost found
#     assert cost <= 5

#     # Test 2: Cannot change imminent waypoint
#     times = jnp.array([10, 16, 22, 27])  # First waypoint very soon
#     current_time = 15

#     new_times, cost = rsp.optimize_waypoints(
#         distances, waypoints, times, current_time, 0, max_active_trips=2
#     )
#     # Must keep immediate next waypoint
#     assert new_times[0] == 10
#     assert new_times[1] == 16
#     # Rest should be optimally ordered
#     assert new_times[1] > new_times[0]  # D1 after P1
#     assert new_times[2] < new_times[3]  # P2 before D2

#     # Test 3: Completed waypoints remain unchanged
#     times = jnp.array([10, 12, 22, 27])  # First two waypoints in past
#     current_time = 15

#     new_times, cost = rsp.optimize_waypoints(
#         distances, waypoints, times, current_time
#     )
#     # Past waypoints unchanged
#     assert jnp.array_equal(new_times[:2], times[:2])
#     # Future waypoints optimized
#     assert new_times[2] < new_times[3]

#     # Test 4: Triangle inequality test case
#     distances = jnp.array(
#         [
#             [0, 10, 1, 1],  # Going 0->1 directly is worse than 0->2->3->1
#             [10, 0, 1, 1],
#             [1, 1, 0, 1],
#             [1, 1, 1, 0],
#         ]
#     )
#     waypoints = jnp.array([0, 1, 2, 3])
#     times = jnp.array([20, 25, 22, 27])
#     current_time = 15

#     new_times, cost = rsp.optimize_waypoints(
#         distances, waypoints, times, current_time, 0, max_active_trips=2
#     )
#     # Should find path that violates triangle inequality if optimal
#     assert cost < 10  # Direct 0->1 cost

#     # Test 5: All waypoints completed
#     times = jnp.array([5, 8, 10, 12])  # All in past
#     current_time = 15

#     new_times, cost = rsp.optimize_waypoints(
#         distances, waypoints, times, current_time
#     )
#     # Should keep all times unchanged
#     assert jnp.array_equal(new_times, times)
#     assert cost == 0


def test_greedy_policy_no_feasible_cars():
    """Test that GreedyPolicy returns -1 when no cars can accept more trips"""
    n_cars = 2
    n_nodes = 4
    max_active_trips = 2
    env = rsp.RidesharePoolDispatch(
        n_cars=n_cars, n_nodes=n_nodes, n_events=1
    )
    
    # Create a state where all cars are at max capacity
    state = rsp.EnvState(
        time=0,
        waypoints=jnp.array([
            [0, 1, 2, 3],  # Car 0: 2 active trips (P1,D1,P2,D2)
            [1, 2, 3, 4],  # Car 1: 2 active trips (P1,D1,P2,D2)
        ]),
        times=jnp.array([
            [10, 20, 30, 40],  # All times in future = active
            [15, 25, 35, 45],  # All times in future = active
        ]),
        key=jax.random.PRNGKey(0),
        event=rs.RideshareEvent(
            t=jnp.array(5), src=jnp.array(0), dest=jnp.array(1)
        ),
    )

    # Create environment params
    env_params = rsp.EnvParams(
        events=state.event,
        distances=jnp.ones((n_nodes, n_nodes), dtype=jnp.int32),
        n_cars=n_cars,
        max_active_trips=max_active_trips
    )

    # Initialize policy
    policy = rsp.GreedyPolicy(n_cars=n_cars, temperature=0.0)

    # Get action from policy
    obs = env.get_obs(state)
    action, _ = policy.apply(env_params, {}, obs, jax.random.PRNGKey(1))

    # Should return -1 since no cars can accept more trips
    assert action == -1

def test_identical_trips_same_car():
    """Test that identical trips arriving in succession get assigned to the same car"""
    n_cars = 2
    n_nodes = 4
    n_events = 3
    env = rsp.RidesharePoolDispatch(
        n_cars=n_cars, n_nodes=n_nodes, n_events=n_events
    )

    # Simple distance matrix where all distances = 10
    distances = jnp.ones((n_nodes, n_nodes), dtype=jnp.int32) * 10
    distances = distances.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(
        0
    )  # Self-loops are 0

    # Create two identical events at t=0,1
    events = rs.RideshareEvent(
        t=jnp.array([1, 2, 3], dtype=jnp.int32),  # Three events
        src=jnp.array([0, 0, 0], dtype=jnp.int32),  # Same source
        dest=jnp.array([1, 1, 1], dtype=jnp.int32),  # Same destination
    )

    env_params = rsp.EnvParams(
        events=events, distances=distances, n_cars=n_cars, max_active_trips=2
    )
    policy = rsp.GreedyPolicy(n_cars=n_cars, temperature=0.0)  # Deterministic

    # Reset environment
    obs, state = env.reset(jax.random.PRNGKey(0), env_params)
    state = state.replace(
        waypoints=jnp.ones((n_cars, 4), dtype=jnp.int32) * 2,
    )

    # Get action for first trip
    action1, _ = policy.apply(env_params, {}, obs, jax.random.PRNGKey(1))
    obs1, state1, rew1, _, info1 = env.step(
        jax.random.PRNGKey(2), state, action1, env_params
    )

    # Get action for second trip
    action2, _ = policy.apply(env_params, {}, obs1, jax.random.PRNGKey(3))
    obs2, state2, rew2, _, info2 = env.step(
        jax.random.PRNGKey(2), state1, action2, env_params
    )

    # Both trips should be assigned to the same car
    assert action1 == action2
    assert info2["marginal_cost"] == 0
    assert jnp.array_equal(
        state2.times[0], jnp.array([11, 21, 11, 21], dtype=state2.times.dtype)
    )


# def test_rideshare_pool_dispatch():
#     """Test that dispatching a car in RidesharePoolDispatch works correctly"""
#     n_cars = 2
#     n_nodes = 4
#     n_events = 3
#     env = rsp.RidesharePoolDispatch(
#         n_cars=n_cars, n_nodes=n_nodes, n_events=n_events
#     )

#     # Simple distance matrix where all distances = 1 except self-loops
#     distances = (jnp.ones((n_nodes, n_nodes)) - jnp.eye(n_nodes)) * 20

#     # Create events: at t=0,1,2 with different src/dest
#     events = rs.RideshareEvent(
#         t=jnp.array([0, 1, 2]),
#         src=jnp.array([0, 1, 2]),
#         dest=jnp.array([1, 2, 3]),
#     )

#     env_params = rs.EnvParams(events=events, distances=distances, n_cars=n_cars)

#     # Reset environment
#     obs, state = env.reset(jax.random.PRNGKey(0), env_params)

#     # Test dispatching first car (action=0) for first event
#     new_obs, new_state, reward, done, info = env.step(
#         jax.random.PRNGKey(1), state, action=0, params=env_params
#     )

#     # Check that waypoints were updated correctly
#     assert jnp.array_equal(
#         new_state.waypoints[0][:2], jnp.array([0, 1])
#     )  # First car's first trip
#     assert jnp.array_equal(
#         new_state.waypoints[1], state.waypoints[1]
#     )  # Second car unchanged

#     # Check that times were updated
#     assert new_state.times[0][0] >= state.event.t  # Pickup time after event
#     assert new_state.times[0][1] > new_state.times[0][0]  # Dropoff after pickup
#     assert jnp.array_equal(
#         new_state.times[1], state.times[1]
#     )  # Second car unchanged

#     # Test adding second trip to same car
#     next_obs, next_state, next_reward, next_done, next_info = env.step(
#         jax.random.PRNGKey(2), new_state, action=0, params=env_params
#     )

#     # Check second trip waypoints were added
#     assert jnp.array_equal(next_state.waypoints[0][2:4], jnp.array([1, 2]))

#     # Verify ordering constraints
#     assert next_state.times[0][0] < next_state.times[0][1]  # P1 before D1
#     assert next_state.times[0][2] < next_state.times[0][3]  # P2 before D2
