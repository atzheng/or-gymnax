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

    # Ghost car counterfactual tracking (fixed-size ring buffer)
    ghost_waypoints: Integer[Array, "max_ghosts max_waypoints"]
    ghost_times: Integer[Array, "max_ghosts max_waypoints"]
    ghost_birth_time: Integer[Array, "max_ghosts"]
    ghost_origin_step: Integer[Array, "max_ghosts"]
    ghost_type: Integer[Array, "max_ghosts"]  # 0=canonical-not-dispatched, 1=cf-dispatched
    ghost_active: Bool[Array, "max_ghosts"]
    ghost_excluded_cars: Integer[Array, "max_ghosts max_exclusions"]  # -1 padded
    ghost_n_excluded: Integer[Array, "max_ghosts"]
    ghost_threshold: Float[Array, "max_ghosts"]  # savings threshold stored per ghost
    ghost_write_idx: Integer[Array, ""]  # ring buffer pointer


@struct.dataclass
class EnvParams(rs.EnvParams):
    max_active_trips: int = struct.field(pytree_node=False, default=4)
    profit_margin: float = 1
    max_ghosts: int = struct.field(pytree_node=False, default=128)
    ghost_max_lifespan: int = 50
    ghost_max_branch_depth: int = struct.field(pytree_node=False, default=0)

    @property
    def max_exclusions(self):
        return self.ghost_max_branch_depth + 1


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


def _ghost_marginal_cost(distances, ghost_wp, ghost_t, event, max_active_trips):
    """Compute marginal cost of inserting event into a single ghost car."""
    _, _, cost, is_feasible = insert_and_optimize_trip(
        distances, ghost_wp, ghost_t,
        event.src, event.dest, event.t, max_active_trips,
    )
    return cost, is_feasible


def check_ghost_triggers(
    distances, state, event, real_costs, real_is_feasible, max_active_trips,
):
    """
    For each active ghost, check if either:
      (a) the ghost would beat the best eligible real car (excluding canonical), OR
      (b) a canonical (excluded) car is the best eligible real car overall.

    Eligibility uses the ghost's stored savings_threshold and direct pickup-to-dropoff
    cost as the solo baseline (not the cheapest available solo car).

    Returns:
        triggered: Bool[max_ghosts] — which ghosts triggered (ghost wins OR canonical wins)
        ghost_wins: Bool[max_ghosts] — which ghosts would be dispatched (ghost beats real fleet)
        ghost_costs: Integer[max_ghosts] — marginal cost per ghost
        ghost_feasible: Bool[max_ghosts] — whether ghost insertion is feasible
    """
    maxint = jnp.iinfo(real_costs.dtype).max
    direct_cost = distances[event.src, event.dest]
    real_is_solo = jnp.all(state.times <= event.t, axis=1)

    def check_one_ghost(ghost_wp, ghost_t, excluded_cars, n_excluded, active, threshold):
        cost, feasible = _ghost_marginal_cost(
            distances, ghost_wp, ghost_t, event, max_active_trips,
        )
        # Ghost eligibility: solo or passes savings threshold vs direct cost
        ghost_is_solo = jnp.all(ghost_t <= event.t)
        ghost_eligible = ghost_is_solo | (cost < direct_cost * (1 - threshold))

        # Build exclusion mask for this ghost
        exclude_mask = jnp.zeros(real_costs.shape[0], dtype=bool)
        exclude_mask = exclude_mask.at[excluded_cars].set(True)
        idx_range = jnp.arange(excluded_cars.shape[0])
        valid_exclusion = idx_range < n_excluded
        exclude_mask = jnp.where(
            jnp.any(valid_exclusion),
            exclude_mask,
            jnp.zeros_like(exclude_mask),
        )

        # Real car eligibility with this ghost's threshold
        real_eligible = real_is_feasible & (
            real_is_solo | (real_costs < direct_cost * (1 - threshold))
        )

        # Ghost wins: ghost beats eligible real fleet (excluding canonical cars)
        masked_real = jnp.where(exclude_mask | ~real_eligible, maxint, real_costs)
        best_real = jnp.min(masked_real)
        ghost_wins = active & feasible & ghost_eligible & (cost < best_real)

        # Canonical wins: excluded car is best eligible real car
        eligible_real_costs = jnp.where(real_eligible, real_costs, maxint)
        best_real_car = jnp.argmin(eligible_real_costs)
        canonical_wins = (
            active & exclude_mask[best_real_car] & (jnp.min(eligible_real_costs) < maxint)
        )
        triggered = ghost_wins | canonical_wins
        return triggered, ghost_wins, cost, feasible

    triggered, ghost_wins, ghost_costs, ghost_feasible = jax.vmap(check_one_ghost)(
        state.ghost_waypoints,
        state.ghost_times,
        state.ghost_excluded_cars,
        state.ghost_n_excluded,
        state.ghost_active,
        state.ghost_threshold,
    )
    return triggered, ghost_wins, ghost_costs, ghost_feasible


def update_triggered_ghosts(
    distances, state, event, triggered, max_active_trips,
):
    """Update ghost car state for triggered ghosts (dispatch the current trip to them)."""
    def update_one(ghost_wp, ghost_t, was_triggered):
        new_wp, new_t, _, _ = insert_and_optimize_trip(
            distances, ghost_wp, ghost_t,
            event.src, event.dest, event.t, max_active_trips,
        )
        # Only update if triggered
        out_wp = jnp.where(was_triggered, new_wp, ghost_wp)
        out_t = jnp.where(was_triggered, new_t, ghost_t)
        return out_wp, out_t

    new_ghost_wps, new_ghost_ts = jax.vmap(update_one)(
        state.ghost_waypoints, state.ghost_times, triggered,
    )
    return new_ghost_wps, new_ghost_ts


def apply_ghost_a(pre_state, next_state, canonical_car, threshold_a, max_exclusions, max_ghosts):
    """Write Ghost A (canonical car pre-dispatch) into next_state's ghost buffer.

    Ghost A represents the canonical car's state had it NOT been dispatched.
    pre_state supplies the car's waypoints and the origin step; next_state supplies
    the ghost buffer fields (already updated by trigger/expiry) and the write index.
    """
    wp = pre_state.waypoints[canonical_car]
    t = pre_state.times[canonical_car]
    excluded = jnp.full(max_exclusions, -1, dtype=jnp.int32).at[0].set(canonical_car)
    idx = next_state.ghost_write_idx % max_ghosts
    return next_state.replace(
        ghost_waypoints=next_state.ghost_waypoints.at[idx].set(wp),
        ghost_times=next_state.ghost_times.at[idx].set(t),
        ghost_birth_time=next_state.ghost_birth_time.at[idx].set(pre_state.event.t),
        ghost_origin_step=next_state.ghost_origin_step.at[idx].set(pre_state.time),
        ghost_type=next_state.ghost_type.at[idx].set(jnp.array(0, dtype=jnp.int32)),
        ghost_active=next_state.ghost_active.at[idx].set(True),
        ghost_excluded_cars=next_state.ghost_excluded_cars.at[idx].set(excluded),
        ghost_n_excluded=next_state.ghost_n_excluded.at[idx].set(jnp.array(1, dtype=jnp.int32)),
        ghost_threshold=next_state.ghost_threshold.at[idx].set(threshold_a),
        ghost_write_idx=(next_state.ghost_write_idx + 1) % max_ghosts,
    )


def apply_ghost_b(pre_state, next_state, cf_car, threshold_b, distances,
                  max_active_trips, max_exclusions, max_ghosts):
    """Write Ghost B (cf car with trip inserted) into next_state's ghost buffer.

    Ghost B represents what would have happened had the cf car been dispatched.
    pre_state supplies the car's waypoints and the origin step; next_state supplies
    the ghost buffer fields (already updated by Ghost A write if applicable).
    """
    wp, t, _, _ = insert_and_optimize_trip(
        distances, pre_state.waypoints[cf_car], pre_state.times[cf_car],
        pre_state.event.src, pre_state.event.dest, pre_state.event.t, max_active_trips,
    )
    excluded = jnp.full(max_exclusions, -1, dtype=jnp.int32).at[0].set(cf_car)
    idx = next_state.ghost_write_idx % max_ghosts
    return next_state.replace(
        ghost_waypoints=next_state.ghost_waypoints.at[idx].set(wp),
        ghost_times=next_state.ghost_times.at[idx].set(t),
        ghost_birth_time=next_state.ghost_birth_time.at[idx].set(pre_state.event.t),
        ghost_origin_step=next_state.ghost_origin_step.at[idx].set(pre_state.time),
        ghost_type=next_state.ghost_type.at[idx].set(jnp.array(1, dtype=jnp.int32)),
        ghost_active=next_state.ghost_active.at[idx].set(True),
        ghost_excluded_cars=next_state.ghost_excluded_cars.at[idx].set(excluded),
        ghost_n_excluded=next_state.ghost_n_excluded.at[idx].set(jnp.array(1, dtype=jnp.int32)),
        ghost_threshold=next_state.ghost_threshold.at[idx].set(threshold_b),
        ghost_write_idx=(next_state.ghost_write_idx + 1) % max_ghosts,
    )


def expire_ghosts(state, current_time, ghost_max_lifespan):
    """Deactivate ghosts that have exceeded their lifespan."""
    expired = (current_time - state.ghost_birth_time) >= ghost_max_lifespan
    return state.ghost_active & ~expired


def compute_real_car_costs(distances, waypoints, times, event, max_active_trips):
    """Compute marginal cost for each real car. Returns (costs, is_feasible)."""
    def cost_one(car_wp, car_t):
        _, _, cost, feasible = insert_and_optimize_trip(
            distances, car_wp, car_t,
            event.src, event.dest, event.t, max_active_trips,
        )
        return cost, feasible
    return jax.vmap(cost_one)(waypoints, times)


def greedy_select_car(
    distances,
    waypoints: Integer[Array, "n_cars max_waypoints"],
    times: Integer[Array, "n_cars max_waypoints"],
    event: rs.RideshareEvent,
    max_active_trips: int,
    savings_threshold: float,
    exclude_car: Integer[Array, ""] = jnp.array(-1, dtype=jnp.int32),
) -> Tuple[Integer[Array, ""], Bool[Array, ""]]:
    """Select the cheapest eligible car under the savings threshold.

    A car is eligible if it is solo (no active trips) or its marginal cost
    is below direct_cost * (1 - savings_threshold), where direct_cost is
    the pickup-to-dropoff distance. Car exclude_car is always ineligible
    (use -1 to exclude nothing).

    Returns (car_idx, found); car_idx is valid only when found=True.
    """
    direct_cost = distances[event.src, event.dest]
    car_indices = jnp.arange(waypoints.shape[0], dtype=jnp.int32)

    def cost_one(car_wp, car_t):
        is_solo = jnp.all(car_t <= event.t)
        _, _, cost, feasible = insert_and_optimize_trip(
            distances, car_wp, car_t,
            event.src, event.dest, event.t, max_active_trips,
        )
        eligible = feasible & (is_solo | (cost < direct_cost * (1 - savings_threshold)))
        return cost, eligible

    costs, eligible = jax.vmap(cost_one)(waypoints, times)
    eligible = eligible & (car_indices != exclude_car)

    maxint = jnp.iinfo(costs.dtype).max
    masked_costs = jnp.where(eligible, costs, maxint)
    car_idx = jnp.argmin(masked_costs)
    found = jnp.any(eligible)
    return car_idx, found


class RidesharePoolDispatch(rs.RideshareDispatch):
    def __init__(
        self,
        n_cars: int,
        n_nodes: int,
        n_events: int,
    ):
        super().__init__(n_cars=n_cars, n_nodes=n_nodes, n_events=n_events)

    def _ghost_step(self, state, params, real_costs, real_is_feasible):
        """Run ghost trigger checks and expiry. Returns updated ghost fields and trigger info."""
        event = state.event

        # Check which ghosts trigger
        triggered, ghost_wins, ghost_costs, ghost_feasible = check_ghost_triggers(
            params.distances, state, event,
            real_costs, real_is_feasible, params.max_active_trips,
        )

        # Only dispatch trip to ghosts that actually won (not canonical-wins triggers)
        new_ghost_wps, new_ghost_ts = update_triggered_ghosts(
            params.distances, state, event, ghost_wins, params.max_active_trips,
        )

        # Expire old ghosts
        ghost_active = expire_ghosts(state, event.t, params.ghost_max_lifespan)

        # Build trigger info
        ghost_info = {
            "ghost_triggered": triggered,
            "ghost_trigger_origin_steps": jnp.where(
                triggered, state.ghost_origin_step, -1,
            ),
            "n_ghost_triggers": jnp.sum(triggered),
            "n_active_ghosts": jnp.sum(ghost_active),
            "step": state.time,
        }

        return new_ghost_wps, new_ghost_ts, ghost_active, ghost_info

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Float[Array, "2"],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment.

        action[0] = savings_threshold for the canonical (treatment A) policy.
        action[1] = savings_threshold for the counterfactual (treatment B) policy.

        The environment internally selects the canonical car (cheapest eligible
        under threshold_A) and the counterfactual car (cheapest eligible under
        threshold_B). Dispatches the canonical car if found, otherwise unfulfills.

        Ghost creation is symmetric and independent:
          Ghost A is written iff A dispatches (canonical_found and feasible).
          Ghost B is written iff B would dispatch (cf_found) and cf != canonical.
        """
        threshold_a = action[0].astype(jnp.float32)
        threshold_b = action[1].astype(jnp.float32)

        canonical_car, canonical_found = greedy_select_car(
            params.distances, state.waypoints, state.times, state.event,
            params.max_active_trips, threshold_a,
        )
        canonical_car = jnp.where(canonical_found, canonical_car, jnp.array(-1, dtype=jnp.int32))
        # cf search excludes canonical_car so the two arms always select different cars.
        # When canonical_car=-1 (A unfulfills), exclude_car=-1 excludes nothing.
        cf_car, cf_found = greedy_select_car(
            params.distances, state.waypoints, state.times, state.event,
            params.max_active_trips, threshold_b, exclude_car=canonical_car,
        )
        cf_car = jnp.where(cf_found, cf_car, jnp.array(-1, dtype=jnp.int32))

        obs, next_state, reward, done, info = jax.lax.cond(
            canonical_found,
            lambda: self.step_env_dispatch(key, state, canonical_car, params),
            lambda: self.step_env_unfulfill(key, state, params),
        )

        # Ghost A: canonical car pre-dispatch. Written iff A dispatched.
        next_state = jax.lax.cond(
            canonical_found,
            lambda ns: apply_ghost_a(
                state, ns, canonical_car, threshold_a,
                params.max_exclusions, params.max_ghosts,
            ),
            lambda ns: ns,
            next_state,
        )

        # Ghost B: cf car with trip inserted. Written iff B found.
        next_state = jax.lax.cond(
            cf_found,
            lambda ns: apply_ghost_b(
                state, ns, cf_car, threshold_b, params.distances,
                params.max_active_trips, params.max_exclusions, params.max_ghosts,
            ),
            lambda ns: ns,
            next_state,
        )

        return (
            obs,
            lax.stop_gradient(next_state),
            reward,
            done,
            {
                **info,
                "action_A": canonical_car,  # already -1 when canonical_found=False
                "action_B": cf_car,
            },
        )

    def step_env_unfulfill(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # Compute real car costs for ghost comparison
        real_costs, real_is_feasible = compute_real_car_costs(
            params.distances, state.waypoints, state.times,
            state.event, params.max_active_trips,
        )

        # Ghost step: check triggers, update, expire
        new_ghost_wps, new_ghost_ts, ghost_active, ghost_info = self._ghost_step(
            state, params, real_costs, real_is_feasible,
        )

        key, event_key = jax.random.split(state.key)
        next_event = rs.get_random_event(event_key, params.events, state.event.t)
        next_state = EnvState(
            time=state.time + 1,
            waypoints=state.waypoints,
            times=state.times,
            key=key,
            event=next_event,
            ghost_waypoints=new_ghost_wps,
            ghost_times=new_ghost_ts,
            ghost_birth_time=state.ghost_birth_time,
            ghost_origin_step=state.ghost_origin_step,
            ghost_type=state.ghost_type,
            ghost_active=ghost_active,
            ghost_excluded_cars=state.ghost_excluded_cars,
            ghost_n_excluded=state.ghost_n_excluded,
            ghost_threshold=state.ghost_threshold,
            ghost_write_idx=state.ghost_write_idx,
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
            next_state,
            jnp.array(reward, dtype=float),
            done,
            {
                "discount": self.discount(state, params),
                "is_unfulfill": True,
                "marginal_cost": 0,
                "utilization": utilization,
                "pct_cars_on_trip": pct_cars_on_trip,
                "t": state.event.t,
                **ghost_info,
            },
        )

    def step_env_dispatch(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        canonical_car: Integer[Array, ""],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # Compute real car costs for ghost comparison
        real_costs, real_is_feasible = compute_real_car_costs(
            params.distances, state.waypoints, state.times,
            state.event, params.max_active_trips,
        )

        # Ghost trigger checks and updates on existing ghosts
        new_ghost_wps, new_ghost_ts, ghost_active, ghost_info = self._ghost_step(
            state, params, real_costs, real_is_feasible,
        )

        # Execute canonical dispatch
        (
            new_car_wps,
            new_car_times,
            marginal_cost,
            is_feasible,
        ) = insert_and_optimize_trip(
            params.distances,
            state.waypoints[canonical_car],
            state.times[canonical_car],
            state.event.src,
            state.event.dest,
            state.event.t,
            params.max_active_trips,
        )
        new_waypoints = state.waypoints.at[canonical_car].set(new_car_wps)
        new_times = state.times.at[canonical_car].set(new_car_times)

        key, event_key = jax.random.split(state.key)
        next_event = rs.get_random_event(event_key, params.events, state.event.t)
        next_state = EnvState(
            time=state.time + 1,
            waypoints=new_waypoints,
            times=new_times,
            key=key,
            event=next_event,
            ghost_waypoints=new_ghost_wps,
            ghost_times=new_ghost_ts,
            ghost_birth_time=state.ghost_birth_time,
            ghost_origin_step=state.ghost_origin_step,
            ghost_type=state.ghost_type,
            ghost_active=ghost_active,
            ghost_excluded_cars=state.ghost_excluded_cars,
            ghost_n_excluded=state.ghost_n_excluded,
            ghost_threshold=state.ghost_threshold,
            ghost_write_idx=state.ghost_write_idx,
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
            next_state,
            jnp.array(reward, dtype=float),
            done,
            {
                "discount": self.discount(state, params),
                "is_unfulfill": False,
                "marginal_cost": marginal_cost,
                "utilization": utilization,
                "pct_cars_on_trip": pct_cars_on_trip,
                "t": state.event.t,
                **ghost_info,
            },
        )

        return jax.lax.cond(
            is_feasible,
            lambda: results,
            lambda: self.step_env_unfulfill(key, state, params),
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key, key_reset = jax.random.split(key)
        max_wp = _num_wp(params.max_active_trips)
        state = EnvState(
            time=0,
            waypoints=jax.random.choice(
                key_reset,
                jnp.arange(self.n_nodes),
                (self.n_cars, max_wp),
            ),
            times=jnp.zeros((self.n_cars, max_wp), dtype=int),
            key=key,
            event=rs.get_random_event(key_reset, params.events, 0),
            ghost_waypoints=jnp.zeros((params.max_ghosts, max_wp), dtype=int),
            ghost_times=jnp.zeros((params.max_ghosts, max_wp), dtype=int),
            ghost_birth_time=jnp.zeros(params.max_ghosts, dtype=int),
            ghost_origin_step=jnp.full(params.max_ghosts, -1, dtype=int),
            ghost_type=jnp.zeros(params.max_ghosts, dtype=int),
            ghost_active=jnp.zeros(params.max_ghosts, dtype=bool),
            ghost_excluded_cars=jnp.full(
                (params.max_ghosts, params.max_exclusions), -1, dtype=int,
            ),
            ghost_n_excluded=jnp.zeros(params.max_ghosts, dtype=int),
            ghost_threshold=jnp.zeros(params.max_ghosts, dtype=jnp.float32),
            ghost_write_idx=jnp.array(0, dtype=int),
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
        # Use direct pickup-to-dropoff distance as the solo baseline.
        # This is independent of which solo cars happen to be available.
        min_solo_cost = env_params.distances[event.src, event.dest]

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
        """Return [savings_threshold, savings_threshold] for both treatment arms.

        The environment uses these thresholds to select canonical and counterfactual
        cars internally via greedy_select_car.
        """
        threshold = jnp.array(self.savings_threshold, dtype=jnp.float32)
        action = jnp.array([threshold, threshold])
        return action, {}


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
        print(f"Selected car: {action[0]} (counterfactual: {action[1]})")
        print(f"Reward: {reward}")
        print(
            f"Event: t={state.event.t}, src={state.event.src}, dest={state.event.dest}"
        )
        print(f"Car waypoints:\n{state.waypoints}")
        print(f"Car times:\n{state.times}")
