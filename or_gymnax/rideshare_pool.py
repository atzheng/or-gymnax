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


def check_group_triggers(
    distances, state, event, canonical_car, canonical_found,
    threshold, max_active_trips, max_groups, max_ghosts_per_group,
):
    """Per-group trigger check: build counterfactual fleet, dispatch, compare to canonical.

    For each active group:
    1. Build cf fleet by swapping ghost states into real fleet at their car positions
    2. Run greedy_select_car on the cf fleet with the current threshold
    3. Trigger fires if cf dispatch differs from canonical dispatch

    Both dispatches use the current action's threshold (not stored group threshold)
    to isolate the causal effect of car-state divergence.

    Returns per-group arrays:
        triggered: Bool[max_groups]
        cf_cars: Integer[max_groups] — cf dispatch result (-1 if not found)
        cf_founds: Bool[max_groups]
    """
    n_slots = max_groups * max_ghosts_per_group
    max_wp = state.waypoints.shape[1]
    all_ghost_wps = state.ghost_waypoints[:n_slots].reshape(
        max_groups, max_ghosts_per_group, max_wp)
    all_ghost_ts = state.ghost_times[:n_slots].reshape(
        max_groups, max_ghosts_per_group, max_wp)

    def check_one_group(group_active, group_n_ghosts, group_ghost_car_ids,
                         group_ghost_wps, group_ghost_ts):
        # Build cf fleet: real fleet with ghost states swapped in
        def swap_one(carry, j):
            cw, ct = carry
            valid = j < group_n_ghosts
            car_id = group_ghost_car_ids[j]
            new_cw = cw.at[car_id].set(group_ghost_wps[j])
            new_ct = ct.at[car_id].set(group_ghost_ts[j])
            cw = jnp.where(valid, new_cw, cw)
            ct = jnp.where(valid, new_ct, ct)
            return (cw, ct), None

        (cf_wp, cf_t), _ = jax.lax.scan(
            swap_one, (state.waypoints, state.times), jnp.arange(max_ghosts_per_group)
        )

        cf_car, cf_found = greedy_select_car(
            distances, cf_wp, cf_t, event, max_active_trips, threshold,
        )

        # Normalize cf_car to -1 when not found (canonical_car is already -1 when not found)
        cf_car = jnp.where(cf_found, cf_car, jnp.int32(-1))

        # Trigger: active group where dispatches differ
        triggered = group_active & (
            (cf_car != canonical_car) | (cf_found != canonical_found)
        )

        return triggered, cf_car, cf_found

    triggered, cf_cars, cf_founds = jax.vmap(check_one_group)(
        state.group_active,
        state.group_n_ghosts,
        state.group_ghost_car_ids,
        all_ghost_wps,
        all_ghost_ts,
    )
    return triggered, cf_cars, cf_founds


def update_triggered_ghost_groups(
    distances, state, event, triggered, cf_cars,
    max_active_trips, max_groups, max_ghosts_per_group,
):
    """Update ghost state for triggered groups where cf dispatched a group ghost."""
    n_slots = max_groups * max_ghosts_per_group
    max_wp = state.waypoints.shape[1]
    all_ghost_wps = state.ghost_waypoints[:n_slots].reshape(
        max_groups, max_ghosts_per_group, max_wp)
    all_ghost_ts = state.ghost_times[:n_slots].reshape(
        max_groups, max_ghosts_per_group, max_wp)

    def update_one_group(group_triggered, cf_car, group_n_ghosts,
                          group_ghost_car_ids, group_ghost_wps, group_ghost_ts):
        def update_one_slot(ghost_wp, ghost_t, car_id, j):
            is_match = group_triggered & (j < group_n_ghosts) & (car_id == cf_car)
            new_wp, new_t, _, _ = insert_and_optimize_trip(
                distances, ghost_wp, ghost_t,
                event.src, event.dest, event.t, max_active_trips,
            )
            return jnp.where(is_match, new_wp, ghost_wp), jnp.where(is_match, new_t, ghost_t)

        js = jnp.arange(max_ghosts_per_group)
        updated_wps, updated_ts = jax.vmap(update_one_slot)(
            group_ghost_wps, group_ghost_ts, group_ghost_car_ids, js,
        )
        return updated_wps, updated_ts

    updated_wps, updated_ts = jax.vmap(update_one_group)(
        triggered, cf_cars, state.group_n_ghosts, state.group_ghost_car_ids,
        all_ghost_wps, all_ghost_ts,
    )
    return updated_wps.reshape(n_slots, max_wp), updated_ts.reshape(n_slots, max_wp)


def grow_group_on_trigger(
    car_waypoints,        # (n_cars, max_wp) - real fleet pre-dispatch states
    car_times,            # (n_cars, max_wp)
    ghost_waypoints,      # (n_slots, max_wp) - ghost states after update_triggered step
    ghost_times,          # (n_slots, max_wp)
    group_n_ghosts,       # (max_groups,)
    group_excluded_cars,  # (max_groups, mpg)
    group_ghost_car_ids,  # (max_groups, mpg)
    triggered,            # (max_groups,) bool
    cf_cars,              # (max_groups,) int - cf dispatch car per group (-1 if not found)
    cf_founds,            # (max_groups,) bool
    canonical_car,        # scalar int32
    canonical_found,      # scalar bool
    distances, event, max_active_trips, max_groups, max_ghosts_per_group,
):
    """Grow ghost groups on trigger by adding ghosts for newly divergent cars.

    For each triggered group (canonical ≠ cf dispatch):
    - Car X (canonical): add Ghost X (pre-dispatch state) if X not already in exclusion set.
    - Car Y (cf): add Ghost Y (with-trip state) if Y not already in exclusion set.
      If Y is already a ghost, it was updated by update_triggered_ghost_groups; no duplicate.

    Growth is capped at max_ghosts_per_group per group.
    """
    mpg = max_ghosts_per_group
    max_wp = ghost_waypoints.shape[-1]
    n_slots = max_groups * mpg

    all_ghost_wps = ghost_waypoints[:n_slots].reshape(max_groups, mpg, max_wp)
    all_ghost_ts = ghost_times[:n_slots].reshape(max_groups, mpg, max_wp)

    # Canonical car's pre-dispatch state (shared across all groups)
    canonical_car_wp = car_waypoints[canonical_car]
    canonical_car_t = car_times[canonical_car]

    # Cf car states per group (safe indexing: clamp invalid -1 to 0)
    safe_cf_cars = jnp.where(cf_cars >= 0, cf_cars, jnp.int32(0))
    cf_car_wps = car_waypoints[safe_cf_cars]  # (max_groups, max_wp)
    cf_car_ts = car_times[safe_cf_cars]        # (max_groups, max_wp)

    def grow_one_group(
        group_triggered, cf_car, cf_found,
        group_n_ghosts, group_excluded_cars, group_ghost_car_ids,
        group_ghost_wps, group_ghost_ts,
        cf_car_wp, cf_car_t,
    ):
        # Check if canonical car (X) is already in this group's exclusion set
        x_in_set = jnp.any(group_excluded_cars == canonical_car)
        add_x = group_triggered & canonical_found & ~x_in_set & (group_n_ghosts < mpg)

        # Check if cf car (Y) is already in this group's exclusion set
        y_in_set = jnp.any(group_excluded_cars == cf_car)
        n_after_x = group_n_ghosts + add_x.astype(jnp.int32)
        add_y = group_triggered & cf_found & ~y_in_set & (n_after_x < mpg)

        # Local slot indices within the group's contiguous block
        slot_x = group_n_ghosts
        slot_y = n_after_x

        # Ghost Y state: cf car with the trip inserted
        y_wp_new, y_t_new, _, _ = insert_and_optimize_trip(
            distances, cf_car_wp, cf_car_t,
            event.src, event.dest, event.t, max_active_trips,
        )

        # Write ghost X (canonical car's pre-dispatch state)
        new_group_wps = jnp.where(
            add_x,
            group_ghost_wps.at[slot_x].set(canonical_car_wp),
            group_ghost_wps,
        )
        new_group_ts = jnp.where(
            add_x,
            group_ghost_ts.at[slot_x].set(canonical_car_t),
            group_ghost_ts,
        )

        # Write ghost Y (cf car with trip)
        new_group_wps = jnp.where(
            add_y,
            new_group_wps.at[slot_y].set(y_wp_new),
            new_group_wps,
        )
        new_group_ts = jnp.where(
            add_y,
            new_group_ts.at[slot_y].set(y_t_new),
            new_group_ts,
        )

        # Update n_ghosts
        new_n_ghosts = group_n_ghosts + add_x.astype(jnp.int32) + add_y.astype(jnp.int32)

        # Update exclusion set and car ID arrays
        new_excluded = jnp.where(
            add_x,
            group_excluded_cars.at[slot_x].set(canonical_car),
            group_excluded_cars,
        )
        new_excluded = jnp.where(
            add_y,
            new_excluded.at[slot_y].set(cf_car),
            new_excluded,
        )
        new_car_ids = jnp.where(
            add_x,
            group_ghost_car_ids.at[slot_x].set(canonical_car),
            group_ghost_car_ids,
        )
        new_car_ids = jnp.where(
            add_y,
            new_car_ids.at[slot_y].set(cf_car),
            new_car_ids,
        )

        return new_n_ghosts, new_excluded, new_car_ids, new_group_wps, new_group_ts

    new_n_ghosts, new_excluded, new_car_ids, new_group_wps, new_group_ts = jax.vmap(
        grow_one_group
    )(
        triggered, cf_cars, cf_founds,
        group_n_ghosts, group_excluded_cars, group_ghost_car_ids,
        all_ghost_wps, all_ghost_ts,
        cf_car_wps, cf_car_ts,
    )

    new_ghost_wps = new_group_wps.reshape(n_slots, max_wp)
    new_ghost_ts = new_group_ts.reshape(n_slots, max_wp)

    return new_n_ghosts, new_excluded, new_car_ids, new_ghost_wps, new_ghost_ts


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

    def _ghost_step(self, state, params, canonical_car, canonical_found, threshold):
        """Run ghost trigger checks, growth, and expiry. Returns updated ghost fields and trigger info."""
        event = state.event
        max_groups = params.max_groups
        mpg = params.max_ghosts_per_group

        # Per-group trigger check: build cf fleet, dispatch, compare to canonical
        triggered, cf_cars, cf_founds = check_group_triggers(
            params.distances, state, event,
            canonical_car, canonical_found, threshold,
            params.max_active_trips, max_groups, mpg,
        )

        # Update ghost state for triggered groups where cf dispatched a group ghost
        new_ghost_wps, new_ghost_ts = update_triggered_ghost_groups(
            params.distances, state, event, triggered, cf_cars,
            params.max_active_trips, max_groups, mpg,
        )

        # Grow groups: add ghosts for newly divergent cars on trigger
        new_group_n_ghosts, new_group_excluded, new_group_car_ids, new_ghost_wps, new_ghost_ts = (
            grow_group_on_trigger(
                state.waypoints, state.times,
                new_ghost_wps, new_ghost_ts,
                state.group_n_ghosts, state.group_excluded_cars, state.group_ghost_car_ids,
                triggered, cf_cars, cf_founds,
                canonical_car, canonical_found,
                params.distances, event, params.max_active_trips, max_groups, mpg,
            )
        )

        # Expire old groups
        group_active = expire_groups(state, event.t, params.ghost_max_lifespan)

        group_ages = jnp.where(group_active, event.t - state.group_birth_time, jnp.int32(0))
        ghost_info = {
            "group_triggered": triggered,
            "group_trigger_origin_steps": jnp.where(
                triggered, state.group_origin_step, -1,
            ),
            "n_group_triggers": jnp.sum(triggered),
            "n_active_groups": jnp.sum(group_active),
            "oldest_ghost_age": jnp.max(group_ages),
            "step": state.time,
        }

        return (
            new_ghost_wps, new_ghost_ts, group_active,
            new_group_n_ghosts, new_group_excluded, new_group_car_ids,
            ghost_info,
        )

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
        # Ghost step: check triggers, update, grow, expire
        (
            new_ghost_wps, new_ghost_ts, group_active,
            new_group_n_ghosts, new_group_excluded, new_group_car_ids,
            ghost_info,
        ) = self._ghost_step(state, params, jnp.int32(-1), jnp.bool_(False), threshold)

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
            group_n_ghosts=new_group_n_ghosts,
            group_excluded_cars=new_group_excluded,
            group_ghost_car_ids=new_group_car_ids,
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
        # Ghost trigger checks, updates on existing ghosts, and group growth
        (
            new_ghost_wps, new_ghost_ts, group_active,
            new_group_n_ghosts, new_group_excluded, new_group_car_ids,
            ghost_info,
        ) = self._ghost_step(state, params, canonical_car, jnp.bool_(True), threshold)

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
            group_n_ghosts=new_group_n_ghosts,
            group_excluded_cars=new_group_excluded,
            group_ghost_car_ids=new_group_car_ids,
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
