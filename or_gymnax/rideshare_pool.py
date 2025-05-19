"""
Pricing and dispatch ridesharing environments
"""
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
from flax import linen as nn
import jax
from jax import Array
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from jaxtyping import Float, Integer, Bool

from . import rideshare as rs


MAX_ACTIVE_TRIPS = 2


@partial(jax.jit, static_argnums=(0, 1))
def obs_to_state(n_cars: int, max_waypoints: int, obs: Integer[Array, "o_dim"]):
    obs = obs.astype(int)
    event = rs.RideshareEvent(obs[0], obs[1], obs[2])
    waypoints = obs[3 : 3 + n_cars * max_waypoints].reshape(
        n_cars, max_waypoints
    )

    times = obs[3 + n_cars * max_waypoints :].reshape(n_cars, max_waypoints)
    return event, waypoints, times


@struct.dataclass
class EnvState(environment.EnvState):
    # Locations of currently assigned waypoints for each car.
    # Contiguous blocks of size waypoints_per_trip represent
    # waypoints for a single trip, but the ordering of trips does not matter
    # E.g., [P1, D1, P2, D2] or [P2, D2, P1, D1] for 2 pickups and 2 dropoffs
    waypoints: Integer[Array, "n_cars max_waypoints"]
    # Completion times for each waypoint, possibly in the future
    times: Integer[Array, "n_cars max_waypoints"]
    # Random key for sampling
    key: Integer[Array, "2"]
    # The current ride request event
    event: rs.RideshareEvent


def insert_and_optimize_trip(
    distances, waypoints, times, pickup_id, dropoff_id, time
):
    # Figure out where to insert the new trip
    # ------------------------------------------------------
    is_active = times > time
    pickup_is_active = is_active[jnp.array([0, 2])]
    dropoff_is_active = is_active[jnp.array([1, 3])]
    trip_is_active = jnp.logical_or(pickup_is_active, dropoff_is_active)
    num_active_trips = jnp.sum(trip_is_active)

    # Not worrying about whether there is space for a new trip
    # for now; will handle via cond at end of function
    # Add a new trip if there is an open slot
    first_inactive_trip = jnp.argmin(trip_is_active)

    new_waypoints = (
        waypoints.at[first_inactive_trip * 2]
        .set(pickup_id)
        .at[first_inactive_trip * 2 + 1]
        .set(dropoff_id)
    )
    new_times_draft = (
        times.at[first_inactive_trip * 2]
        .set(jnp.inf)
        .at[first_inactive_trip * 2 + 1]
        .set(jnp.inf)
    )  # Todo make this maxint

    is_active = times > time
    next_wp_idx = jax.lax.cond(
        jnp.any(is_active),  # If any waypoints are active...
        # ...find the next waypoint
        lambda: jnp.argmin(jnp.where(times >= time, times, jnp.inf)),
        # ...else, return the last completed waypoint
        lambda: jnp.argmax(times),
    )

    next_wp_time = jax.lax.cond(
        jnp.any(is_active),
        lambda: times[next_wp_idx],
        lambda: time,
    )

    # Figure out where to insert the new trip
    # ------------------------------------------------------
    new_times, marginal_cost = optimize_waypoints(
        distances,
        new_waypoints,
        new_times_draft,
        waypoints[next_wp_idx],
        next_wp_time,
    )

    marginal_cost_or_infeasible = jax.lax.cond(
        num_active_trips == MAX_ACTIVE_TRIPS,
        lambda: jnp.inf,
        lambda: marginal_cost,
    )

    return new_waypoints, new_times, marginal_cost_or_infeasible


def optimize_waypoints(
    distances: Integer[Array, "nodes nodes"],
    waypoints: Integer[Array, "max_waypoints"],
    times: Integer[Array, "max_waypoints"],
    start_waypoint: int,
    start_time: int,
) -> Tuple[
    Integer[Array, "max_waypoints"],  # Completion times
    Integer[Array, "1"],  # Marginal cost
]:
    """
    Return the shortest admissible sequence and its tour length.

    `times` is used to determine which waypoints are active, and which will
    be completed next.

    For efficiency, this simulator will assume that cars cannot be diverted from
    their current waypoint.
    This can be made increasingly realistic by increasing the number of waypoints
    recorded for each trip.
    To do this we will later need to add a "routes" element to the envstate of
    length points_per_segment * max_waypoints
    """
    is_active = times > start_time

    # Compute marginal times for each possible waypoint ordering
    # ----------------------------------------------------------------------
    # All admissible permutations that respect P≺D constraints
    seqs = jnp.array(
        [
            [0, 2, 1, 3],  # P1 P2 D1 D2
            [0, 2, 3, 1],  # P1 P2 D2 D1
            [2, 0, 1, 3],  # P2 P1 D1 D2
            [2, 0, 3, 1],  # P2 P1 D2 D1
            [0, 1, 2, 3],  # P1 D1 P2 D2
            [2, 3, 0, 1],  # P2 D2 P1 D1
        ]
    )  # shape (6,4)

    # Map sequence indices → actual node indices
    seq_is_active = is_active[seqs]
    # Replace inactive nodes with start_waypoint
    # so that the distance to complete them will be 0
    seqs_replace_inactive = jnp.where(seq_is_active, seqs, start_waypoint)
    seq_wps_with_start = jnp.concatenate(
        (
            jnp.repeat(start_waypoint, seqs.shape[0]).reshape(-1, 1),
            jnp.where(seq_is_active, waypoints[seqs], start_waypoint)
        ),
        axis=1
    )

    # Gather pair-wise edge lengths for every leg in every sequence
    src = seq_wps_with_start[:, :-1]
    dst = seq_wps_with_start[:, 1:]
    leg_dists = distances[src, dst]  # (6,4)
    leg_marginal_times = jnp.cumsum(leg_dists, axis=1)
    seq_completion_times = leg_marginal_times + start_time
    seq_marginal_time = leg_marginal_times[:, -1]
    wp_completion_times = jnp.where(
        is_active,
        jnp.zeros_like(seqs)
        .at[
            jnp.tile(
                jnp.expand_dims(jnp.arange(seqs.shape[0]), axis=1),
                seqs.shape[1],
            ).flatten(),
            seqs.flatten(),
        ]
        .set(seq_completion_times.flatten()),
        times,  # If already completed, don't change it
    )

    # Keep only valid waypoint orderings
    # -------------------------------------------------------------------------
    # Inactive nodes must come first
    is_valid_sequence = jnp.all(
        seq_is_active[:, :-1] <= seq_is_active[:, 1:], axis=1
    )

    # Get the best ordering and marginal cost
    # ----------------------------------------------------------------------
    best_sequence_idx = jnp.argmin(
        jnp.where(is_valid_sequence, seq_marginal_time, jnp.inf), axis=0
    )
    best_seq_times = wp_completion_times[best_sequence_idx]
    best_marginal_cost = seq_marginal_time[best_sequence_idx]

    return best_seq_times, best_marginal_cost


class RidesharePoolDispatch(rs.RideshareDispatch):
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: rs.EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        return jax.lax.cond(
            action >= 0,
            lambda: self.step_env_dispatch(key, state, action, params),
            lambda: self.step_env_unfulfill(key, state, action, params),
        )

    def step_env_unfulfill(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: rs.EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        next_event = rs.get_nth_event(params.events, state.time + 1)
        next_state = EnvState(
            time=state.time + 1,
            waypoints=state.waypoints,
            times=state.times,
            key=state.key,
            event=next_event,
        )
        done = self.is_terminal(next_state, params)
        reward = 0.0
        return (
            lax.stop_gradient(self.get_obs(next_state)),
            lax.stop_gradient(next_state),
            jnp.array(reward, dtype=float),
            done,
            {"discount": self.discount(state, params)},
        )

    def step_env_dispatch(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: rs.EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        new_car_wps, new_car_times, marginal_cost = insert_and_optimize_trip(
            params.distances,
            state.waypoints[action],
            state.times[action],
            state.event.src,
            state.event.dest,
            state.event.t,
        )
        next_state = self.dispatch_and_update_state(state, action, params)
        done = self.is_terminal(next_state, params)

        # TODO Should place these into envparams
        trip_cost = params.distances[state.event.src, state.event.dest]
        profit_margin = 0.3
        reward = (
            trip_cost * (1 + profit_margin) - marginal_cost  # price  # cost
        )

        return (
            lax.stop_gradient(self.get_obs(next_state)),
            lax.stop_gradient(next_state),
            jnp.array(reward, dtype=float),
            done,
            {"discount": self.discount(state, params)},
        )

    def dispatch_and_update_state(
        self, state: EnvState, car_id: int, params: rs.EnvParams
    ) -> EnvState:
        new_car_wps, new_car_times, _ = insert_and_optimize_trip(
            params.distances,
            state.waypoints[car_id],
            state.times[car_id],
            state.event.src,
            state.event.dest,
            state.event.t,
        )
        new_waypoints = state.waypoints.at[car_id].set(new_car_wps)
        new_times = state.times.at[car_id].set(new_car_times)

        next_event = rs.get_nth_event(params.events, state.time + 1)
        next_state = EnvState(
            time=state.time + 1,
            waypoints=new_waypoints,
            times=new_times,
            key=state.key,
            event=next_event,
        )
        return next_state

    def reset_env(
        self, key: chex.PRNGKey, params: rs.EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key, key_reset = jax.random.split(key)
        state = EnvState(
            time=0,
            # Initialize empty waypoints and times
            waypoints=jax.random.choice(
                key_reset,
                jnp.arange(self.n_nodes),
                (self.n_cars, MAX_ACTIVE_TRIPS * 2),
            ),
            times=jnp.zeros((self.n_cars, MAX_ACTIVE_TRIPS * 2), dtype=int),
            key=key,
            event=rs.get_nth_event(params.events, 0),
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        return jnp.concatenate(
            [
                jnp.reshape(state.event.t, (1,)),
                jnp.reshape(state.event.src, (1,)),
                jnp.reshape(state.event.dest, (1,)),
                state.waypoints.flatten(),
                state.times.flatten(),
            ]
        )


class ManhattanRidesharePoolDispatch(RidesharePoolDispatch):
    @property
    def name(self) -> str:
        """Environment name."""
        return "ManhattanRidesharePoolDispatch-v0"

    @property
    def default_params(self) -> rs.EnvParams:
        events, distances = rs.load_manhattan_data()
        return rs.EnvParams(
            events=events, distances=distances, n_cars=self.n_cars
        )


@struct.dataclass
class GreedyPolicy(rs.GreedyPolicy):
    """
    A simple greedy policy that selects the car with the lowest
    marginal cost for pooled rides, accounting for existing waypoints.
    """

    def get_costs(
        self,
        env_params: rs.EnvParams,
        rng: chex.PRNGKey,
        event: rs.RideshareEvent,
        waypoints: Integer[Array, "n_cars max_waypoints"],
        times: Integer[Array, "n_cars max_waypoints"],
        params: Dict,
    ):
        # For each car, compute marginal cost of adding this trip
        def get_car_cost(car_waypoints, car_times):
            _, _, marginal_cost = insert_and_optimize_trip(
                env_params.distances,
                car_waypoints,
                car_times,
                event.src,
                event.dest,
                event.t,
            )
            return marginal_cost

        costs = jax.vmap(get_car_cost)(waypoints, times)
        return costs

    def apply(
        self,
        env_params: rs.EnvParams,
        nn_params: Dict,
        obs: Integer[Array, "o_dim"],
        rng: chex.PRNGKey,
    ):
        event, waypoints, times = obs_to_state(
            self.n_cars, MAX_ACTIVE_TRIPS * 2, obs
        )
        rng, cost_rng = jax.random.split(rng)
        rewards = -self.get_costs(
            env_params, cost_rng, event, waypoints, times, nn_params
        )

        action = jax.random.choice(
            rng,
            jnp.arange(self.n_cars),
            p=jnp.exp((rewards - jnp.max(rewards)) / self.temperature),
        )
        return action, {}


if __name__ == "__main__":
    # Create a small test environment
    n_events = 10
    n_cars = 3
    key = jax.random.PRNGKey(10)
    src_key, dest_key = jax.random.split(key)

    # Initialize environment with simple distance matrix
    env = RidesharePoolDispatch(n_cars=n_cars, n_nodes=5, n_events=n_events)
    env_params = rs.EnvParams(
        events=rs.RideshareEvent(
            t=jnp.arange(n_events),
            src=jax.random.randint(src_key, (n_events,), 0, 5),
            dest=jax.random.randint(dest_key, (n_events,), 0, 5),
        ),
        distances=jnp.ones((5, 5)) - jnp.eye(5),  # Unit distances except self
        n_cars=n_cars,
    )

    # Initialize greedy policy
    policy = GreedyPolicy(n_cars=n_cars, temperature=0.1)

    # Run a few steps
    obs, state = env.reset(key, env_params)
    print("\nInitial state:")
    print(
        f"Event: t={state.event.t}, src={state.event.src}, dest={state.event.dest}"
    )
    print(f"Car waypoints:\n{state.waypoints}")
    print(f"Car times:\n{state.times}")

    for i in range(3):
        key, step_key = jax.random.split(key)
        action, _ = policy.apply(env_params, {}, obs, step_key)
        obs, state, reward, done, info = env.step(
            step_key, state, action, env_params
        )
        print(f"\nStep {i+1}:")
        print(f"Selected car: {action}")
        print(f"Reward: {reward}")
        print(
            f"Event: t={state.event.t}, src={state.event.src}, dest={state.event.dest}"
        )
        print(f"Car waypoints:\n{state.waypoints}")
        print(f"Car times:\n{state.times}")
