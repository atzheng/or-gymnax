from jax.experimental import sparse
from functools import partial
from or_gymnax.rideshare_pool import (
    EnvState,
    ManhattanRidesharePoolDispatch,
    GreedyPolicy,
    EnvParams,
    obs_to_state,
)
from or_gymnax.rideshare import RideshareEvent
import or_gymnax.rideshare_pool as rsp
from or_gymnax.nn import Policy
from jax import numpy as jnp
from typing import Dict, Callable, Tuple, Literal, Any
import chex
from jax import Array
from jaxtyping import Integer, Float, Bool
from flax import struct
from sacred import Experiment
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import funcy as f
import pandas as pd
import jax
import pooch
from jax.tree_util import Partial
from lstd import (
    DQLSTDEstimatorState,
    dqlstd_update,
    dqlstd,
    opelstd,
    opelstd_update,
    OPELSTDEstimatorState,
)
from mc import (
    DQMCEstimatorState,
    dqmc_update,
    dqmc,
)

ex = Experiment("rideshares")


@ex.config
def config():
    n_cars = 300  # Number of cars
    savings_threshold_A = (
        0.0  # Minimum savings required for pooling in policy A
    )
    savings_threshold_B = (
        0.2  # Minimum savings required for pooling in policy B
    )
    n_events = 10000  # Number of events to simulate per trial
    k = 1000  # Total number of trials
    batch_size = 100  # Number of environments to run in parallel
    p = 0.5  # Treatment probability
    output = "results.csv"
    config_output = "config.csv"
    uniformize = True


@struct.dataclass
class NaiveEstimatorState:
    """
    Contains the current state of an estimator
    """

    counts: Integer[Array, "2"]
    rewards: Float[Array, "2"]

    @classmethod
    def init(cls):
        return cls(
            counts=jnp.zeros((2,), dtype=jnp.int32),
            rewards=jnp.zeros((2,), dtype=jnp.float32),
        )


def naive_update(
    est: NaiveEstimatorState,
    reward: float,
    obs: Integer[Array, "o_dim"],
    z: Bool,
):
    return NaiveEstimatorState(
        est.counts.at[z.astype(jnp.uint8)].add(1),
        est.rewards.at[z.astype(jnp.uint8)].add(reward),
    )


def naive(est: NaiveEstimatorState):
    avg_rewards = est.rewards / (est.counts + 1e-8)  # Avoid division by zero
    return avg_rewards[1] - avg_rewards[0]

@f.partial(jax.jit, static_argnames=("n_cars", "max_waypoints", "n_zones"))
def obs_to_repr(
    obs: Integer[Array, "o_dim"],
    n_cars: int,
    max_waypoints: int,
    n_zones: int,
    nodes_to_zones: Integer[Array, "n_nodes"],
) -> Float[Array, "repr_dim"]:
    """
    Convert observation to state representation counting cars by location and active trips.

    Args:
        obs: Raw observation from environment
        n_cars: Number of cars in system
        max_waypoints: Maximum number of waypoints per car
        n_zones: Number of spatial zones (clusters) in the environment
        nodes_zones: DataFrame mapping node IDs to zone IDs

    Returns:
        Array of shape (3 * n_zones, ) containing counts of cars in each
        zone with 0, 1, or 2 active trips respectively
    """
    # Get waypoints and times from observation
    _, waypoints, times = obs_to_state(n_cars, max_waypoints, obs)

    # Get current time
    current_time = obs[0]

    # Count active trips per car
    active_trips = rsp.num_active_trips(waypoints, times, current_time)

    is_active = times > current_time
    next_wp_idx = jnp.where(
        jnp.any(is_active, axis=1),  # If any waypoints are active...
        # ...find the next waypoint
        jnp.argmin(jnp.where(times >= current_time, times, jnp.inf), axis=1),
        # ...else, return the last completed waypoint
        jnp.argmax(times, axis=1),
    )

    next_locations = jnp.take_along_axis(
        waypoints, jnp.expand_dims(next_wp_idx, 1), axis=1
    ).squeeze()

    # Convert node IDs to zone IDs
    zone_ids = nodes_to_zones[next_locations]
    zones_and_active_trips_idx = active_trips * n_zones + zone_ids

    repr = jnp.zeros(3 * n_zones).at[zones_and_active_trips_idx].add(1)
    # Add log features
    repr_aug = jnp.concatenate((repr, jnp.log(repr + 1)))

    # Intercept
    return repr_aug


estimator_fns = {
    "naive": naive,
    "dqlstd": dqlstd,
    "opelstd": opelstd,
    "dqmc-100": dqmc,
    "dqmc-300": dqmc,
}

def collect_step(
    env,
    env_params,
    A: Policy,
    B: Policy,
    obs: Float[Array, "o_dim"],
    state: rsp.EnvState,
    key: chex.PRNGKey,
    p: float,
):
    key, policy_key = jax.random.split(key)
    key, treat_key = jax.random.split(key)
    is_treat = jax.random.bernoulli(treat_key, p=p)
    action, action_info = jax.lax.cond(
        is_treat,
        lambda: B.apply(env_params, dict(), obs, policy_key),
        lambda: A.apply(env_params, dict(), obs, policy_key),
    )

    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)

    return (
        is_treat,
        action,
        reward,
        new_obs,
        new_state,
    )


def stepper(
    estimators: Dict[str, Callable],
    env,
    env_params,
    A: Policy,
    B: Policy,
    carry: Tuple[Array, Array, Dict[str, NaiveEstimatorState], chex.PRNGKey],
    unused: Any,
    p: float = 0.5,
):
    obs, state, ests, key = carry
    key, subkey = jax.random.split(key)
    is_treat, action, reward, new_obs, new_state = collect_step(
        env, env_params, A, B, obs, state, subkey, p
    )
    new_ests = {
        est_name: est_update(ests[est_name], reward, obs, is_treat)
        for est_name, est_update in estimators.items()
    }
    return ((new_obs, new_state, new_ests, key), None)


def init_estimator_states(phi, r, z, p):
    d = phi.shape[0]
    return {
        "naive": NaiveEstimatorState.init(),
        "dqlstd": DQLSTDEstimatorState.init(d, phi, r, z, p),
        "opelstd": OPELSTDEstimatorState.init(d, phi, r),
        "dqmc-1000": DQMCEstimatorState.init(1000),
        "dqmc-3000": DQMCEstimatorState.init(3000),
    }


@f.partial(
    jax.jit,
    static_argnames=(
        "env",
        "A",
        "B",
        "n_zones",
        "p",
        "n_envs",
        "n_steps",
    ),
)
def run_trials(
    env,
    env_params,
    A,
    B,
    key,
    nodes_to_zones,
    n_zones,
    p,
    n_envs=10,
    n_steps=1000,
):
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_envs)
    obs_to_repr_fn = Partial(
        obs_to_repr,
        n_cars=env_params.n_cars,
        max_waypoints=env_params.max_active_trips * 2,
        n_zones=n_zones,
        nodes_to_zones=nodes_to_zones,
    )

    estimators = {
        "naive": naive_update,
        "dqlstd": Partial(dqlstd_update, obs_to_repr_fn, p),
        "opelstd": Partial(opelstd_update, obs_to_repr_fn),
        "dqmc-100": Partial(dqmc_update, p),
        "dqmc-300": Partial(dqmc_update, p),
    }

    def scanner(carry):
        return jax.lax.scan(
            Partial(
                stepper,
                estimators,
                env,
                env_params,
                A,
                B,
                p=p,
            ),
            carry,
            jnp.arange(n_steps),
            unroll=1,
        )[0][2]

    # Collect first step results to initialize estimators
    obs0, state0 = jax.vmap(env.reset, in_axes=(0, None))(
        reset_keys, env_params
    )

    key, init_step_key = jax.random.split(key)
    init_step_keys = jax.random.split(init_step_key, n_envs)
    is_treat, _, rewards, obs1, state1 = jax.vmap(
        collect_step, in_axes=(None, None, None, None, 0, 0, 0, None)
    )(env, env_params, A, B, obs0, state0, init_step_keys, p)

    init_ests = jax.vmap(
        init_estimator_states,
        in_axes=(0, 0, 0, None),
    )(
        jax.vmap(obs_to_repr_fn, in_axes=(0,))(obs1),
        rewards,
        is_treat,
        p,
    )

    step_keys = jax.random.split(key, n_envs)
    init_carry = (obs1, state1, init_ests, step_keys)
    vmap_scan = jax.vmap(scanner, in_axes=(0,))
    estimator_results = vmap_scan(init_carry)

    return {
        est_name: jax.vmap(est, in_axes=(0,))(estimator_results[est_name])
        for est_name, est in estimator_fns.items()
    }


def load_taxi_zones():
    zones = pd.read_parquet("taxi-zones.parquet")
    # TODO not sure if nodes are correctly mapped
    unq_zones, unq_zone_ids = np.unique(zones["zone"], return_inverse=True)
    zones["zone_id"] = unq_zone_ids
    max_zone_id = zones["zone_id"].max()
    nodes_fname = pooch.retrieve(
        "https://github.com/atzheng/nyc-taxi-simulator-data/releases/download/initial-release/manhattan-nodes.parquet",
        known_hash="md5:4d75202c20f3b7816d45b6e068f684b6",
    )
    nodes = pd.read_parquet(nodes_fname)
    nodes["lng"] = nodes["lng"].astype(float)
    nodes["lat"] = nodes["lat"].astype(float)
    nodes_zones = nodes.merge(zones, on="osmid")

    # Create a vector mapping nodes to zones
    nodes_to_zones = (
        (jnp.ones(len(nodes), dtype=jnp.int32) * (max_zone_id + 1))
        .at[nodes_zones["idx"].values]
        .set(nodes_zones["zone_id"].values)
    )
    return nodes_to_zones


@ex.automain
def main(
    n_cars,
    savings_threshold_A,
    savings_threshold_B,
    n_events,
    seed,
    k,
    batch_size,
    p,
    output,
    uniformize,
    _config,
):
    key = jax.random.PRNGKey(seed)
    env = ManhattanRidesharePoolDispatch(
        n_cars=n_cars, n_events=n_events, uniformize=uniformize
    )
    env_params = env.default_params

    nodes_to_zones = load_taxi_zones()

    A = GreedyPolicy(
        n_cars=env.n_cars,
        temperature=0.01,
        savings_threshold=savings_threshold_A,
    )
    B = GreedyPolicy(
        n_cars=env.n_cars,
        temperature=0.01,
        savings_threshold=savings_threshold_B,
    )
    print(
        "Simulation time (mins)",
        (env_params.events.t.max() - env_params.events.t[5]) / 60,
    )
    print(
        "Simulation time (hrs)",
        (env_params.events.t.max() - env_params.events.t[5]) / 3600,
    )

    all_results = []
    keys = jax.random.split(key, k // batch_size + 1)

    for key in tqdm(keys):
        ests = run_trials(
            env,
            env_params,
            A,
            B,
            key,
            n_zones=int(nodes_to_zones.max() + 1),
            n_envs=batch_size,
            n_steps=n_events,
            nodes_to_zones=nodes_to_zones,
            p=p,
        )
        all_results.append(ests)

    pd.DataFrame.from_dict([_config]).to_csv(
        _config["config_output"], index=False
    )
    results_df = pd.concat(map(pd.DataFrame, all_results))
    results_df.to_csv(output, index=False)
