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

    # Ghost group state (ring buffer of groups)
    group_origin_step: Integer[Array, "max_groups"]
    group_birth_time: Integer[Array, "max_groups"]
    group_active: Bool[Array, "max_groups"]
    group_threshold: Float[Array, "max_groups"]  # creation-time threshold (logging only)
    group_n_ghosts: Integer[Array, "max_groups"]  # 0, 1, or 2 ghosts per group
    group_excluded_cars: Integer[Array, "max_groups max_ghosts_per_group"]  # -1 padded
    group_ghost_car_ids: Integer[Array, "max_groups max_ghosts_per_group"]  # -1 padded
    group_write_idx: Integer[Array, ""]  # ring buffer pointer for groups

    # Ghost slot state (contiguous blocks: group g owns slots
    # [g*max_ghosts_per_group, (g+1)*max_ghosts_per_group))
    ghost_waypoints: Integer[Array, "max_ghost_slots max_waypoints"]
    ghost_times: Integer[Array, "max_ghost_slots max_waypoints"]


@struct.dataclass
class EnvParams(rs.EnvParams):
    max_active_trips: int = struct.field(pytree_node=False, default=4)
    profit_margin: float = 1
    max_groups: int = struct.field(pytree_node=False, default=64)
    max_ghosts_per_group: int = struct.field(pytree_node=False, default=2)
    ghost_max_lifespan: int = 50

    @property
    def max_ghost_slots(self):
        return self.max_groups * self.max_ghosts_per_group


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


def _flatten_groups_to_slots(state, max_groups, max_ghosts_per_group):
    """Expand group-level fields to per-slot arrays for vmap.

    Returns per-slot views of: waypoints, times, excluded_cars, n_excluded, active.
    Total number of slots = max_groups * max_ghosts_per_group.
    """
    n_slots = max_groups * max_ghosts_per_group
    g_idx = jnp.repeat(jnp.arange(max_groups), max_ghosts_per_group)
    local_idx = jnp.tile(jnp.arange(max_ghosts_per_group), max_groups)

    # Per-slot validity: slot is valid if local_idx < group's n_ghosts
    is_valid_slot = local_idx < state.group_n_ghosts[g_idx]

    # Slot active = group active AND slot is valid
    slot_active = state.group_active[g_idx] & is_valid_slot

    # Per-ghost exclusion: each ghost excludes only its own car_id
    # (preserves per-ghost trigger semantics; per-group dispatch is issue 89g)
    slot_own_car = state.group_ghost_car_ids[g_idx, local_idx]  # (n_slots,)
    slot_excluded_cars = jnp.full((n_slots, max_ghosts_per_group), -1, dtype=jnp.int32)
    slot_excluded_cars = slot_excluded_cars.at[:, 0].set(slot_own_car)
    slot_n_excluded = jnp.where(is_valid_slot, jnp.int32(1), jnp.int32(0))

    # Ghost data is already stored in contiguous blocks
    # slot index = g * max_ghosts_per_group + local_offset (which is just linear 0..n_slots-1)

    return (
        state.ghost_waypoints[:n_slots],  # (n_slots, max_wp)
        state.ghost_times[:n_slots],  # (n_slots, max_wp)
        slot_excluded_cars,  # (n_slots, max_ghosts_per_group)
        slot_n_excluded,  # (n_slots,)
        slot_active,  # (n_slots,)
        g_idx,  # (n_slots,) — group index for each slot
    )


def check_ghost_triggers(
    distances, state, event, real_costs, real_is_feasible, max_active_trips,
    threshold, max_groups, max_ghosts_per_group,
):
    """
    For each active ghost slot, check if either:
      (a) the ghost would beat the best eligible real car (excluding group's cars), OR
      (b) a group-excluded car is the best eligible real car overall.

    Eligibility uses the current action's threshold (not the stored per-group
    creation-time threshold) and direct pickup-to-dropoff cost as the solo
    baseline. Both canonical and counterfactual fleets use the same threshold
    to isolate the causal effect of car-state divergence.

    Returns per-slot arrays (size max_groups * max_ghosts_per_group):
        triggered: Bool — which slots triggered
        ghost_wins: Bool — which slots would be dispatched
        ghost_costs: Integer — marginal cost per slot
        ghost_feasible: Bool — whether insertion is feasible
    """
    maxint = jnp.iinfo(real_costs.dtype).max
    direct_cost = distances[event.src, event.dest]
    real_is_solo = jnp.all(state.times <= event.t, axis=1)

    slot_wps, slot_ts, slot_excluded, slot_n_excluded, slot_active, _ = \
        _flatten_groups_to_slots(state, max_groups, max_ghosts_per_group)

    def check_one_ghost(ghost_wp, ghost_t, excluded_cars, n_excluded, active):
        cost, feasible = _ghost_marginal_cost(
            distances, ghost_wp, ghost_t, event, max_active_trips,
        )
        # Ghost eligibility: solo or passes savings threshold vs direct cost
        ghost_is_solo = jnp.all(ghost_t <= event.t)
        ghost_eligible = ghost_is_solo | (cost < direct_cost * (1 - threshold))

        # Build exclusion mask using comparison (avoids -1 wrap-around indexing)
        car_ids = jnp.arange(real_costs.shape[0])  # (n_cars,)
        valid_excl = jnp.arange(excluded_cars.shape[0]) < n_excluded  # (max_exc,)
        exclude_mask = jnp.any(
            (car_ids[:, None] == excluded_cars[None, :]) & valid_excl[None, :],
            axis=1,
        )

        # Real car eligibility
        real_eligible = real_is_feasible & (
            real_is_solo | (real_costs < direct_cost * (1 - threshold))
        )

        # Ghost wins: ghost beats eligible real fleet (excluding group's cars)
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
        slot_wps, slot_ts, slot_excluded, slot_n_excluded, slot_active,
    )
    return triggered, ghost_wins, ghost_costs, ghost_feasible


def update_triggered_ghosts(
    distances, state, event, triggered, max_active_trips,
    max_groups, max_ghosts_per_group,
):
    """Update ghost car state for triggered ghosts (dispatch the current trip to them)."""
    n_slots = max_groups * max_ghosts_per_group

    def update_one(ghost_wp, ghost_t, was_triggered):
        new_wp, new_t, _, _ = insert_and_optimize_trip(
            distances, ghost_wp, ghost_t,
            event.src, event.dest, event.t, max_active_trips,
        )
        out_wp = jnp.where(was_triggered, new_wp, ghost_wp)
        out_t = jnp.where(was_triggered, new_t, ghost_t)
        return out_wp, out_t

    new_ghost_wps, new_ghost_ts = jax.vmap(update_one)(
        state.ghost_waypoints[:n_slots],
        state.ghost_times[:n_slots],
        triggered,
    )
    return new_ghost_wps, new_ghost_ts


def create_ghost_group(
    pre_state, next_state, canonical_car, cf_car,
    write_a, write_b, threshold_a,
    distances, max_active_trips, max_ghosts_per_group, max_groups,
):
    """Create a ghost group with Ghost A and/or Ghost B.

    Ghost A = canonical car's pre-dispatch state (what if it wasn't dispatched).
    Ghost B = cf car with the trip inserted (what if it was dispatched instead).

    Slot layout within the group: slot 0 = first ghost, slot 1 = second.
    When only one ghost is written, it occupies slot 0.
    """
    g_idx = next_state.group_write_idx % max_groups
    base_slot = g_idx * max_ghosts_per_group

    # Ghost A data: canonical car's pre-dispatch state
    a_wp = pre_state.waypoints[canonical_car]
    a_t = pre_state.times[canonical_car]

    # Ghost B data: cf car with trip inserted
    b_wp, b_t, _, _ = insert_and_optimize_trip(
        distances, pre_state.waypoints[cf_car], pre_state.times[cf_car],
        pre_state.event.src, pre_state.event.dest, pre_state.event.t,
        max_active_trips,
    )

    n_ghosts = write_a.astype(jnp.int32) + write_b.astype(jnp.int32)

    # Slot 0: A if write_a, else B
    slot0_wp = jnp.where(write_a, a_wp, b_wp)
    slot0_t = jnp.where(write_a, a_t, b_t)
    slot0_car = jnp.where(write_a, canonical_car, cf_car)

    # Slot 1: B (only meaningful when both write)
    both = write_a & write_b
    slot1_wp = jnp.where(both, b_wp, jnp.zeros_like(b_wp))
    slot1_t = jnp.where(both, b_t, jnp.zeros_like(b_t))
    slot1_car = jnp.where(both, cf_car, jnp.int32(-1))

    # Write ghost slots
    ghost_wps = (next_state.ghost_waypoints
        .at[base_slot].set(slot0_wp)
        .at[base_slot + 1].set(slot1_wp))
    ghost_ts = (next_state.ghost_times
        .at[base_slot].set(slot0_t)
        .at[base_slot + 1].set(slot1_t))

    # Exclusion set and car IDs
    excluded = jnp.full(max_ghosts_per_group, -1, dtype=jnp.int32).at[0].set(slot0_car)
    excluded = excluded.at[1].set(slot1_car)
    car_ids = jnp.full(max_ghosts_per_group, -1, dtype=jnp.int32).at[0].set(slot0_car)
    car_ids = car_ids.at[1].set(slot1_car)

    return next_state.replace(
        ghost_waypoints=ghost_wps,
        ghost_times=ghost_ts,
        group_origin_step=next_state.group_origin_step.at[g_idx].set(pre_state.time),
        group_birth_time=next_state.group_birth_time.at[g_idx].set(pre_state.event.t),
        group_active=next_state.group_active.at[g_idx].set(True),
        group_threshold=next_state.group_threshold.at[g_idx].set(threshold_a),
        group_n_ghosts=next_state.group_n_ghosts.at[g_idx].set(n_ghosts),
        group_excluded_cars=next_state.group_excluded_cars.at[g_idx].set(excluded),
        group_ghost_car_ids=next_state.group_ghost_car_ids.at[g_idx].set(car_ids),
        group_write_idx=(next_state.group_write_idx + 1) % max_groups,
    )


def expire_groups(state, current_time, ghost_max_lifespan):
    """Deactivate groups that have exceeded their lifespan."""
    expired = (current_time - state.group_birth_time) >= ghost_max_lifespan
    return state.group_active & ~expired


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
) -> Tuple[Integer[Array, ""], Bool[Array, ""]]:
    """Select the cheapest eligible car under the savings threshold.

    A car is eligible if it is solo (no active trips) or its marginal cost
    is below direct_cost * (1 - savings_threshold), where direct_cost is
    the pickup-to-dropoff distance.

    Returns (car_idx, found); car_idx is valid only when found=True.
    """
    direct_cost = distances[event.src, event.dest]

    def cost_one(car_wp, car_t):
        is_solo = jnp.all(car_t <= event.t)
        _, _, cost, feasible = insert_and_optimize_trip(
            distances, car_wp, car_t,
            event.src, event.dest, event.t, max_active_trips,
        )
        eligible = feasible & (is_solo | (cost < direct_cost * (1 - savings_threshold)))
        return cost, eligible

    costs, eligible = jax.vmap(cost_one)(waypoints, times)
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

    def _ghost_step(self, state, params, real_costs, real_is_feasible, threshold):
        """Run ghost trigger checks and expiry. Returns updated ghost fields and trigger info."""
        event = state.event
        max_groups = params.max_groups
        mpg = params.max_ghosts_per_group
        n_slots = max_groups * mpg

        # Check which ghost slots trigger
        triggered, ghost_wins, ghost_costs, ghost_feasible = check_ghost_triggers(
            params.distances, state, event,
            real_costs, real_is_feasible, params.max_active_trips,
            threshold, max_groups, mpg,
        )

        # Only dispatch trip to ghosts that actually won (not canonical-wins triggers)
        new_ghost_wps, new_ghost_ts = update_triggered_ghosts(
            params.distances, state, event, ghost_wins, params.max_active_trips,
            max_groups, mpg,
        )

        # Expire old groups
        group_active = expire_groups(state, event.t, params.ghost_max_lifespan)

        # Map slot-level triggers back to group-level for info
        g_idx = jnp.repeat(jnp.arange(max_groups), mpg)
        slot_origin_steps = state.group_origin_step[g_idx]

        # Build trigger info (per-slot, matching old interface shape)
        group_ages = jnp.where(group_active, event.t - state.group_birth_time, jnp.int32(0))
        ghost_info = {
            "ghost_triggered": triggered,
            "ghost_trigger_origin_steps": jnp.where(
                triggered, slot_origin_steps, -1,
            ),
            "n_ghost_triggers": jnp.sum(triggered),
            "n_active_ghosts": jnp.sum(group_active),
            "oldest_ghost_age": jnp.max(group_ages),
            "step": state.time,
        }

        return new_ghost_wps, new_ghost_ts, group_active, ghost_info

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

        Ghost group creation:
          A group is created iff the two policies pick different cars (or one
          fulfills while the other doesn't). The group contains 1 or 2 ghosts
          depending on which policies dispatch.
        """
        threshold_a = action[0].astype(jnp.float32)
        threshold_b = action[1].astype(jnp.float32)

        canonical_car, canonical_found = greedy_select_car(
            params.distances, state.waypoints, state.times, state.event,
            params.max_active_trips, threshold_a,
        )
        canonical_car = jnp.where(canonical_found, canonical_car, jnp.array(-1, dtype=jnp.int32))
        cf_car, cf_found = greedy_select_car(
            params.distances, state.waypoints, state.times, state.event,
            params.max_active_trips, threshold_b,
        )
        cf_car = jnp.where(cf_found, cf_car, jnp.array(-1, dtype=jnp.int32))

        obs, next_state, reward, done, info = jax.lax.cond(
            canonical_found,
            lambda: self.step_env_dispatch(key, state, canonical_car, params, threshold_a),
            lambda: self.step_env_unfulfill(key, state, params, threshold_a),
        )

        different_cars = canonical_car != cf_car
        write_a = canonical_found & different_cars
        write_b = cf_found & different_cars
        create_group = write_a | write_b

        next_state = jax.lax.cond(
            create_group,
            lambda ns: create_ghost_group(
                state, ns, canonical_car, cf_car,
                write_a, write_b, threshold_a,
                params.distances, params.max_active_trips,
                params.max_ghosts_per_group, params.max_groups,
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
        threshold: float = 0.0,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # Compute real car costs for ghost comparison
        real_costs, real_is_feasible = compute_real_car_costs(
            params.distances, state.waypoints, state.times,
            state.event, params.max_active_trips,
        )

        # Ghost step: check triggers, update, expire
        new_ghost_wps, new_ghost_ts, group_active, ghost_info = self._ghost_step(
            state, params, real_costs, real_is_feasible, threshold,
        )

        key, event_key = jax.random.split(state.key)
        next_event = rs.get_random_event(event_key, params.events, state.event.t)
        next_state = EnvState(
            time=state.time + 1,
            waypoints=state.waypoints,
            times=state.times,
            key=key,
            event=next_event,
            group_origin_step=state.group_origin_step,
            group_birth_time=state.group_birth_time,
            group_active=group_active,
            group_threshold=state.group_threshold,
            group_n_ghosts=state.group_n_ghosts,
            group_excluded_cars=state.group_excluded_cars,
            group_ghost_car_ids=state.group_ghost_car_ids,
            group_write_idx=state.group_write_idx,
            ghost_waypoints=new_ghost_wps,
            ghost_times=new_ghost_ts,
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
        threshold: float = 0.0,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # Compute real car costs for ghost comparison
        real_costs, real_is_feasible = compute_real_car_costs(
            params.distances, state.waypoints, state.times,
            state.event, params.max_active_trips,
        )

        # Ghost trigger checks and updates on existing ghosts
        new_ghost_wps, new_ghost_ts, group_active, ghost_info = self._ghost_step(
            state, params, real_costs, real_is_feasible, threshold,
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
            group_origin_step=state.group_origin_step,
            group_birth_time=state.group_birth_time,
            group_active=group_active,
            group_threshold=state.group_threshold,
            group_n_ghosts=state.group_n_ghosts,
            group_excluded_cars=state.group_excluded_cars,
            group_ghost_car_ids=state.group_ghost_car_ids,
            group_write_idx=state.group_write_idx,
            ghost_waypoints=new_ghost_wps,
            ghost_times=new_ghost_ts,
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
            lambda: self.step_env_unfulfill(key, state, params, threshold),
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key, key_reset = jax.random.split(key)
        max_wp = _num_wp(params.max_active_trips)
        mg = params.max_groups
        mpg = params.max_ghosts_per_group
        n_slots = mg * mpg
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
            group_origin_step=jnp.full(mg, -1, dtype=int),
            group_birth_time=jnp.zeros(mg, dtype=int),
            group_active=jnp.zeros(mg, dtype=bool),
            group_threshold=jnp.zeros(mg, dtype=jnp.float32),
            group_n_ghosts=jnp.zeros(mg, dtype=int),
            group_excluded_cars=jnp.full((mg, mpg), -1, dtype=int),
            group_ghost_car_ids=jnp.full((mg, mpg), -1, dtype=int),
            group_write_idx=jnp.array(0, dtype=int),
            ghost_waypoints=jnp.zeros((n_slots, max_wp), dtype=int),
            ghost_times=jnp.zeros((n_slots, max_wp), dtype=int),
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
