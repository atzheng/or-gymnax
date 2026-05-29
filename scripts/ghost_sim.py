"""
Single simulation: 1000 steps, 10 cars, ghost cars enabled.
Logs canonical and ghost fleet state at each timestep to ghost_sim_log.jsonl.
"""
import json
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from or_gymnax import rideshare_pool as rp
from or_gymnax import rideshare as rs

N_CARS = 1
N_NODES = 3
N_EVENTS = 30
N_STEPS = 10
MAX_ACTIVE_TRIPS = 2
MAX_GROUPS = 2 * N_NODES
OUTPUT = "ghost_sim_log.jsonl"

key = jax.random.PRNGKey(1)
k1, k2 = jax.random.split(key, 2)

# Build synthetic events with monotone increasing times.
# Three nodes: 0 (node 1), 1 (node 2), 2 (node 3).
# All trips originate at node 0; destination is randomly node 1 or node 2.
event_times = jnp.cumsum(jax.random.randint(k1, (N_EVENTS,), 1, 5))
event_srcs  = jnp.zeros(N_EVENTS, dtype=jnp.int32)
event_dests = (1 + jnp.arange(N_EVENTS).astype(jnp.int32)) % 2 + 1

events = rs.RideshareEvent(t=event_times, src=event_srcs, dest=event_dests)

# Distance matrix: d(0,1)=1, d(1,2)=1, d(0,2)=2
distances = jnp.array([
    [0, 1, 2],
    [1, 0, 1],
    [2, 1, 0],
], dtype=jnp.int32) * 100

env = rp.RidesharePoolDispatch(n_cars=N_CARS, n_nodes=N_NODES, n_events=N_EVENTS)
env_params = rp.EnvParams(
    events=events,
    distances=distances,
    n_cars=N_CARS,
    max_active_trips=MAX_ACTIVE_TRIPS,
    max_groups=MAX_GROUPS,
    ghost_max_lifespan=50,
)

THRESHOLD_A = 0.0   # canonical
THRESHOLD_B = 0.6   # counterfactual
action = jnp.array([THRESHOLD_A, THRESHOLD_B])

key = jax.random.PRNGKey(10)
obs, state = env.reset(key, env_params)
# Initialize each car's first waypoint to node 1
state = state.replace(waypoints=state.waypoints.at[:, 0].set(1))

print(f"Starting simulation: {N_STEPS} steps, {N_CARS} cars, {MAX_GROUPS} ghost groups")
print(f"Logging to {OUTPUT}")

with open(OUTPUT, "w") as f:
    for step in range(N_STEPS):
        key, step_key = jax.random.split(key)
        obs, next_state, reward, done, info = env.step(step_key, state, action, env_params)

        # Canonical fleet
        canonical_cars = []
        for car_idx in range(N_CARS):
            canonical_cars.append({
                "car": int(car_idx),
                "waypoints": state.waypoints[car_idx].tolist(),
                "times": state.times[car_idx].tolist(),
            })

        # Ghost fleet (per-group)
        ghost_cars = []
        mpg = env_params.max_ghosts_per_group
        for g in range(MAX_GROUPS):
            if bool(state.group_active[g]):
                ghosts_in_group = []
                for j in range(int(state.group_n_ghosts[g])):
                    slot = g * mpg + j
                    ghosts_in_group.append({
                        "slot": int(slot),
                        "car_id": int(state.group_ghost_car_ids[g, j]),
                        "waypoints": state.ghost_waypoints[slot].tolist(),
                        "times": state.ghost_times[slot].tolist(),
                    })
                ghost_cars.append({
                    "group_idx": int(g),
                    "birth_time": int(state.group_birth_time[g]),
                    "origin_step": int(state.group_origin_step[g]),
                    "excluded_cars": state.group_excluded_cars[g].tolist(),
                    "n_ghosts": int(state.group_n_ghosts[g]),
                    "ghosts": ghosts_in_group,
                })

        def _scalar(v):
            return v.tolist() if hasattr(v, "tolist") else v

        record = {
            "step": step,
            "event_t": int(state.event.t),
            "event_src": int(state.event.src),
            "event_dest": int(state.event.dest),
            "action_canonical": int(action[0]),
            "action_counterfactual": int(action[1]),
            "reward": float(reward),
            "solo_cost": int(distances[state.event.src, state.event.dest]),
            # Full info dict
            "is_unfulfill": bool(info.get("is_unfulfill", False)),
            "marginal_cost": _scalar(info.get("marginal_cost", 0)),
            "utilization": _scalar(info.get("utilization", 0.0)),
            "pct_cars_on_trip": _scalar(info.get("pct_cars_on_trip", 0.0)),
            "discount": _scalar(info.get("discount", 0.0)),
            "n_active_groups": int(info.get("n_active_groups", 0)),
            "n_group_triggers": int(info.get("n_group_triggers", 0)),
            "oldest_ghost_age": _scalar(info.get("oldest_ghost_age", 0)),
            "group_triggered": _scalar(info.get("group_triggered", [])),
            "group_trigger_origin_steps": _scalar(info.get("group_trigger_origin_steps", [])),
            "action_A": _scalar(info.get("action_A", -1)),
            "action_B": _scalar(info.get("action_B", -1)),
            "info_t": _scalar(info.get("t", -1)),
            "info_step": _scalar(info.get("step", -1)),
            "canonical_fleet": canonical_cars,
            "ghost_fleet": ghost_cars,
        }
        f.write(json.dumps(record) + "\n")

        state = next_state

        if step % 100 == 0:
            n_ghosts = sum(1 for g in ghost_cars if True)
            print(f"  step={step:4d}  event_t={record['event_t']:5d}  "
                  f"action={record['action_canonical']}  "
                  f"reward={record['reward']:6.1f}  "
                  f"active_groups={record['n_active_groups']:3d}  "
                  f"triggers={record['n_group_triggers']}")

print(f"Done. Log written to {OUTPUT}")
