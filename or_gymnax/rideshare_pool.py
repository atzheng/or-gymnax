"""
Pricing and dispatch ridesharing environments
"""
from functools import partial
import itertools
import numpy as np
import chex
from flax import struct
from flax import linen as nn
import jax
from jax import Array
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from jaxtyping import Float, Integer, Bool
from typing import Tuple, Dict, Any, Union, Optional
import itertools as it
from jax.experimental import checkify


from . import rideshare as rs


def _num_wp(max_active_trips: int) -> int:
    """Return number of way-points stored per car."""
    return max_active_trips * 2  # P,D for every trip


def num_active_trips(
    waypoints: Integer[Array, "n_cars max_waypoints"],
    times: Integer[Array, "n_cars max_waypoints"],
    current_time: int,
) -> Integer[Array, "n_cars"]:
    """
    Count number of active trips for each car.

    Args:
        waypoints: Waypoint locations for each car
        times: Completion times for each waypoint
        current_time: Current simulation time

    Returns:
        Array of shape (n_cars,) containing number of active trips per car
    """
    # A waypoint is active if its completion time is in the future
    is_active = times > current_time

    # Reshape to (n_cars, max_active_trips, 2) to group P,D pairs
    n_cars, max_waypoints = waypoints.shape
    max_active_trips = max_waypoints // 2
    is_active_reshaped = is_active.reshape(n_cars, max_active_trips, 2)

    # A trip is active if either P or D is active
    trip_is_active = jnp.any(is_active_reshaped, axis=2)

    # Sum active trips per car
    return jnp.sum(trip_is_active, axis=1)


def admissible_sequences(max_active_trips: int) -> jnp.ndarray:
    """
    All permutations that satisfy P_i ≺ D_i for every trip i.
    Returns shape (n_perm, 2·m) with dtype int32.
    """
    m = max_active_trips
    indices = list(range(2 * m))
    sequences = [
        perm
        for perm in it.permutations(indices)
        if all(perm.index(2 * i) < perm.index(2 * i + 1) for i in range(m))
    ]
    return jnp.asarray(np.array(sequences), dtype=jnp.int32)


_SEQS = {m: admissible_sequences(m) for m in (2, 3)}


def get_sequences(max_active_trips: int) -> jnp.ndarray:
    return _SEQS[max_active_trips]


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


@struct.dataclass
class EnvParams(rs.EnvParams):
    max_active_trips: int = struct.field(pytree_node=False, default=4)
    profit_margin: float = 1


def insert_and_optimize_trip(
    distances, waypoints, times, pickup_id, dropoff_id, time, max_active_trips
):
    # Figure out where to insert the new trip
    # ------------------------------------------------------
    is_active = times > time
    pickup_is_active = is_active[::2]  # even indices → all pickups
    dropoff_is_active = is_active[1::2]  # odd indices → all drop-offs
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

    # Set the "draft" completion times to max so that
    # the waypoint optimizer knows that the trip is active
    new_times_draft = (
        times.at[first_inactive_trip * 2]
        .set(jnp.iinfo(times.dtype).max)
        .at[first_inactive_trip * 2 + 1]
        .set(jnp.iinfo(times.dtype).max)
    )

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

    current_marginal_cost = jax.lax.cond(
        jnp.any(is_active),
        lambda: jnp.max(times) - next_wp_time,
        lambda: 0,
    )

    # Figure out where to insert the new trip
    # ------------------------------------------------------
    new_times, new_marginal_cost = optimize_waypoints(
        distances,
        new_waypoints,
        new_times_draft,
        waypoints[next_wp_idx],
        next_wp_time,
        max_active_trips,
    )
    marginal_marginal_cost = new_marginal_cost - current_marginal_cost
    is_feasible = num_active_trips < max_active_trips
    marginal_cost_or_inf = jax.lax.cond(
        is_feasible,
        lambda: marginal_marginal_cost,
        lambda: jnp.iinfo(new_marginal_cost.dtype).max,
    )

    return (
        new_waypoints,
        new_times,
        marginal_cost_or_inf,
        is_feasible,
    )


def optimize_waypoints(
    distances: Integer[Array, "nodes nodes"],
    waypoints: Integer[Array, "max_waypoints"],
    times: Integer[Array, "max_waypoints"],
    start_waypoint: int,
    start_time: int,
    max_active_trips: int,
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
    # NOTE marginal times are not the additional time incurred by adding the
    # latest waypoint -- These are additional times relative to traveling to the first
    # waypoint (which cannot be modified)

    # All admissible permutations that respect P≺D constraints
    seqs = get_sequences(max_active_trips)  # shape (n_perm, 2·m)

    # Map sequence indices → actual node indices
    seq_is_active = is_active[seqs]
    # Replace inactive nodes with start_waypoint
    # so that the distance to complete them will be 0
    seqs_replace_inactive = jnp.where(seq_is_active, seqs, start_waypoint)
    seq_wps_with_start = jnp.concatenate(
        (
            jnp.repeat(start_waypoint, seqs.shape[0]).reshape(-1, 1),
            jnp.where(seq_is_active, waypoints[seqs], start_waypoint),
        ),
        axis=1,
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
    def __init__(
        self,
        n_cars: int,
        n_nodes: int,
        n_events: int,
    ):
        super().__init__(n_cars=n_cars, n_nodes=n_nodes, n_events=n_events)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
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
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        key, event_key = jax.random.split(state.key)
        next_event = rs.get_random_event(event_key, params.events, state.event.t)
        next_state = EnvState(
            time=state.time + 1,
            waypoints=state.waypoints,
            times=state.times,
            key=key,
            event=next_event,
        )
        done = self.is_terminal(next_state, params)
        reward = 0.0
        utilization = (
            jnp.sum(state.times > state.event.t)
            / _num_wp(params.max_active_trips)
            / params.n_cars
        )
        pct_cars_on_trip = jnp.any(state.times > state.event.t, axis=1).mean()

        return (
            lax.stop_gradient(self.get_obs(next_state)),
            lax.stop_gradient(next_state),
            jnp.array(reward, dtype=float),
            done,
            {
                "discount": self.discount(state, params),
                "is_unfulfill": True,
                # "is_match": False,
                "marginal_cost": 0,
                "utilization": utilization,
                "pct_cars_on_trip": pct_cars_on_trip,
                "t": state.event.t,
            },
        )

    def step_env_dispatch(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        (
            new_car_wps,
            new_car_times,
            marginal_cost,
            is_feasible,
        ) = insert_and_optimize_trip(
            params.distances,
            state.waypoints[action],
            state.times[action],
            state.event.src,
            state.event.dest,
            state.event.t,
            params.max_active_trips,
        )
        # checkify.check(is_feasible, "Trip being inserted should be feasible for car")
        new_waypoints = state.waypoints.at[action].set(new_car_wps)
        new_times = state.times.at[action].set(new_car_times)
        key, event_key = jax.random.split(state.key)
        next_event = rs.get_random_event(event_key, params.events, state.event.t)
        next_state = EnvState(
            time=state.time + 1,
            waypoints=new_waypoints,
            times=new_times,
            key=key,
            event=next_event,
        )
        done = self.is_terminal(next_state, params)
        trip_direct_cost = params.distances[state.event.src, state.event.dest]
        reward = trip_direct_cost * (1 + params.profit_margin) - marginal_cost
        utilization = (
            jnp.sum(state.times > state.event.t)
            / _num_wp(params.max_active_trips)
            / params.n_cars
        )

        pct_cars_on_trip = jnp.any(state.times > state.event.t, axis=1).mean()

        results = (
            lax.stop_gradient(self.get_obs(next_state)),
            lax.stop_gradient(next_state),
            jnp.array(reward, dtype=float),
            done,
            {
                "discount": self.discount(state, params),
                "is_unfulfill": False,
                "marginal_cost": marginal_cost,
                "utilization": utilization,
                "pct_cars_on_trip": pct_cars_on_trip,
                "t": state.event.t,
            },
        )

        return jax.lax.cond(
            is_feasible,
            lambda: results,
            lambda: self.step_env_unfulfill(key, state, action, params),
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key, key_reset = jax.random.split(key)
        state = EnvState(
            time=0,
            # Initialize empty waypoints and times
            waypoints=jax.random.choice(
                key_reset,
                jnp.arange(self.n_nodes),
                (self.n_cars, _num_wp(params.max_active_trips)),
            ),
            times=jnp.zeros(
                (self.n_cars, _num_wp(params.max_active_trips)), dtype=int
            ),
            key=key,
            event=rs.get_random_event(key_reset, params.events, 0),
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
    def __init__(self, *args, uniformize=False, **kwargs):
        super().__init__(*args, **kwargs, n_nodes=4333)
        self.uniformize = uniformize

    @property
    def name(self) -> str:
        """Environment name."""
        return "ManhattanRidesharePoolDispatch-v0"

    @property
    def default_params(self) -> EnvParams:
        events, distances = rs.load_manhattan_data(uniformize=self.uniformize)
        return EnvParams(
            events=jax.tree.map(lambda x: x[: self.n_events], events),
            distances=distances,
            n_cars=self.n_cars,
            max_active_trips=3,
        )


@struct.dataclass
class GreedyPolicy(rs.GreedyPolicy):
    """
    A simple greedy policy that selects the car with the lowest
    marginal cost for pooled rides, accounting for existing waypoints.
    """

    savings_threshold: Union[float, Float[Array, "n_cars"]] = 0.0

    def get_costs(
        self,
        env_params: EnvParams,
        rng: chex.PRNGKey,
        event: rs.RideshareEvent,
        waypoints: Integer[Array, "n_cars max_waypoints"],
        times: Integer[Array, "n_cars max_waypoints"],
        params: Dict,
    ):
        # For each car, compute marginal cost of adding this trip
        def get_car_cost(car_waypoints, car_times):
            is_solo = jnp.all(car_times <= event.t)
            _, _, marginal_cost, is_feasible = insert_and_optimize_trip(
                env_params.distances,
                car_waypoints,
                car_times,
                event.src,
                event.dest,
                event.t,
                env_params.max_active_trips,
            )
            return is_solo, marginal_cost, is_feasible

        is_solo, costs, is_feasible = jax.vmap(get_car_cost)(waypoints, times)
        maxint = jnp.iinfo(costs.dtype).max
        min_solo_cost = jnp.min(jnp.where(is_solo, costs, maxint))
        min_pool_cost = jnp.min(jnp.where(~is_solo, costs, maxint))

        # Check for eligibility based on savings threshold
        is_eligible = jnp.logical_or(
            is_solo,  # Solo trips are always eligible
            costs < min_solo_cost * (1 - self.savings_threshold),
        )

        # Only use pool if it saves savings_threshold % of the cost relative to solo
        thresholded_costs = jnp.where(is_eligible, costs, maxint)
        return thresholded_costs, is_feasible

    def apply(
        self,
        env_params: EnvParams,
        nn_params: Dict,
        obs: Integer[Array, "o_dim"],
        rng: chex.PRNGKey,
    ):
        event, waypoints, times = obs_to_state(
            self.n_cars, _num_wp(env_params.max_active_trips), obs
        )
        rng, cost_rng = jax.random.split(rng)
        costs, is_feasible = self.get_costs(
            env_params, cost_rng, event, waypoints, times, nn_params
        )

        # Assume positive part of reward is constant, so ignore
        rewards = -costs

        best_action = jax.random.choice(
            rng,
            jnp.arange(self.n_cars),
            p=jax.nn.softmax(
                (rewards - jnp.max(rewards)) / self.temperature,
                where=is_feasible,
            ),
        )

        action = jax.lax.cond(
            jnp.any(is_feasible),
            lambda: best_action,
            lambda: -1,  # No feasible action
        )

        info = {
            "best_cost": -rewards[best_action],
        }

        return action, info


if __name__ == "__main__":
    # Create a small test environment
    n_events = 10
    n_cars = 3
    key = jax.random.PRNGKey(10)
    src_key, dest_key = jax.random.split(key)

    # Initialize environment with simple distance matrix
    env = RidesharePoolDispatch(n_cars=n_cars, n_nodes=5, n_events=n_events)
    env_params = EnvParams(
        events=rs.RideshareEvent(
            t=jnp.arange(n_events),
            src=jax.random.randint(src_key, (n_events,), 0, 5),
            dest=jax.random.randint(dest_key, (n_events,), 0, 5),
        ),
        distances=jnp.ones((5, 5)) - jnp.eye(5),  # Unit distances except self
        n_cars=n_cars,
        max_active_trips=2,
    )

    # Initialize greedy policy
    policy = GreedyPolicy(n_cars=n_cars, temperature=0.1, savings_threshold=0.1)

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
