"""
Pricing and dispatch ridesharing environments
"""
from dataclasses import field
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
from gymnax.environments import spaces
from jaxtyping import Float, Integer, Bool
import funcy as f
import pooch
import numpy as np
import pandas as pd

from .nn import Policy, MLP
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


def insert_and_optimize_trip(distances, waypoints, times, pickup_id, dropoff_id, time):
    # Figure out where to insert the new trip
    # ------------------------------------------------------
    is_active = times > time
    pickup_mask = jnp.array([1, 0, 1, 0])
    pickup_is_active = is_active[pickup_mask]
    dropoff_mask = jnp.array([0, 1, 0, 1])
    dropoff_is_active = is_active[dropoff_mask]
    trip_is_active = jnp.logical_or(pickup_is_active, dropoff_is_active)
    num_active_trips = jnp.sum(trip_is_active)

    # Not worrying about whether there is space for a new trip
    # for now; will handle via cond at end of function
    # Add a new trip if there is an open slot
    first_inactive_trip = jnp.argmin(trip_is_active)
    new_waypoints = waypoints.at[
        first_inactive_trip * 2 : first_inactive_trip * 2 + 1
    ].set(jnp.array([pickup_id, dropoff_id]))
    new_times_draft = times.at[
        first_inactive_trip * 2 : first_inactive_trip * 2 + 1
    ].set(jnp.array([jnp.inf, jnp.inf]))  # Todo make this maxint

    # Figure out where to insert the new trip
    # ------------------------------------------------------
    new_times, marginal_cost = optimize_waypoints(
        distances,
        new_waypoints,
        new_times_draft,
        time,
    )

    marginal_cost_or_infeasible = jax.lax.cond(
        num_active_trips == MAX_ACTIVE_TRIPS,
        jnp.inf,
        marginal_cost,
    )

    return new_waypoints, new_times, marginal_cost_or_infeasible


def optimize_waypoints(
    distances: Integer[Array, "nodes nodes"],
    waypoints: Integer[Array, "max_waypoints"],
    times: Integer[Array, "max_waypoints"],
    time: int,
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
    # Replace inactive nodes with next_wp_idx,
    # so that the distance to complete them will be 0
    seqs_replace_inactive = jnp.where(seq_is_active, seqs, next_wp_idx)
    seqs_replace_inactive_wps = waypoints[seqs_replace_inactive]

    # Gather pair-wise edge lengths for every leg in every sequence
    src = seqs_replace_inactive_wps[:, :-1]
    dst = seqs_replace_inactive_wps[:, 1:]
    leg_dists = distances[src, dst]  # (6,4)
    leg_marginal_times = jnp.cumsum(leg_dists, axis=1)
    seq_completion_times = jnp.concatenate(
        (
            jnp.zeros((leg_marginal_times.shape[0], 1)),
            leg_marginal_times + next_wp_time,
        ),
        axis=1,
    )
    seq_marginal_time = leg_marginal_times[:, -1]
    wp_completion_times = jnp.where(
        is_active,
        jnp.zeros_like(seqs).at[seqs].set(seq_completion_times),
        times,  # If already completed, don't change it
    )

    # Keep only valid waypoint orderings
    # -------------------------------------------------------------------------
    # Inactive nodes must come first
    is_inactive_first = jnp.all(
        seq_is_active[:, :-1] <= seq_is_active[:, 1:], axis=1
    )
    # Cannot divert car from its current waypoint, so the first active node must
    # be next_wp_idx
    seq_is_active_or_next_wp = jnp.logical_or(
        seq_is_active, seqs == next_wp_idx
    )
    is_inactive_or_next_wp_first = jnp.all(
        seq_is_active_or_next_wp[:, :-1] <= seq_is_active_or_next_wp[:, 1:],
        axis=1,
    )
    is_valid_sequence = jnp.logical_and(
        is_inactive_first,
        is_inactive_or_next_wp_first,
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
    def dispatch_and_update_state(
        self, state: EnvState, car_id: int, params: rs.EnvParams
    ) -> EnvState:
        new_car_wps, new_car_times, marginal_costs = insert_and_optimize_trip(
            state.waypoints[car_id],
            state.times[car_id],
            state.event.src,
            state.event.dest,
            state.time,
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


if __name__ == "__main__":
    n_events = 100
    key = jax.random.PRNGKey(0)
    env = ManhattanRidesharePricing(n_cars=10000, n_events=n_events)
    env_params = env.default_params
    print(env_params)
    A = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.1)
    obs, state = env.reset(key, env_params)
    action, action_info = A.apply(env_params, dict(), obs, key)
    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)
