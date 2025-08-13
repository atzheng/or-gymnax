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
    DQLSTDPGEstimatorState,
    dqlstdpg_update,
    dqlstdpg,
    opelstd,
    opelstd_update,
    OPELSTDEstimatorState,
)
from mc_v2 import (
    DQMCEstimatorState,
    dqmc_update,
    dqmc,
)
from diffgq1 import (
    OPEDiffGQ1EstimatorState,
    opediffgq1_update,
    opediffgq1,
)
from step import StepInfo

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
    k = 10  # Total number of trials
    batch_size = 10  # Number of environments to run in parallel
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
    obs: Integer[Array, "o_dim"],
    stepinfo: StepInfo,
):
    return NaiveEstimatorState(
        est.counts.at[stepinfo.is_treat.astype(jnp.uint8)].add(1),
        est.rewards.at[stepinfo.is_treat.astype(jnp.uint8)].add(stepinfo.reward),
    )


def naive(est: NaiveEstimatorState):
    avg_rewards = est.rewards / (est.counts + 1e-8)  # Avoid division by zero
    return avg_rewards[1] - avg_rewards[0]


@f.partial(jax.jit, static_argnames=("n_cars", "max_waypoints", "n_zones"))
# def obs_to_repr(
#     obs: Integer[Array, "o_dim"],
#     n_cars: int,
#     max_waypoints: int,
#     n_zones: int,
#     nodes_to_zones: Integer[Array, "n_nodes"],
# ) -> Float[Array, "repr_dim"]:
#     """
#     Convert observation to state representation counting cars by location and active trips.

#     Args:
#         obs: Raw observation from environment
#         n_cars: Number of cars in system
#         max_waypoints: Maximum number of waypoints per car
#         n_zones: Number of spatial zones (clusters) in the environment
#         nodes_zones: DataFrame mapping node IDs to zone IDs

#     Returns:
#         Array of shape (3 * n_zones, ) containing counts of cars in each
#         zone with 0, 1, or 2 active trips respectively
#     """
#     # Get waypoints and times from observation
#     _, waypoints, times = obs_to_state(n_cars, max_waypoints, obs)

#     # Get current time
#     current_time = obs[0]

#     # Count active trips per car
#     active_trips = rsp.num_active_trips(waypoints, times, current_time)

#     final_wp_idx = jnp.argmax(times, axis=1)

#     is_active = times > current_time
#     is_solo = jnp.all(~is_active, axis=1)

#     next_wp_idx = jnp.argmin(
#         jnp.where(times >= current_time, times, jnp.inf), axis=1
#     )

#     next_locations = jnp.take_along_axis(
#         waypoints, jnp.expand_dims(next_wp_idx, 1), axis=1
#     ).squeeze()

#     final_locations = jnp.take_along_axis(
#         waypoints, jnp.expand_dims(final_wp_idx, 1), axis=1
#     ).squeeze()

#     next_zones = nodes_to_zones[next_locations]
#     final_zones = nodes_to_zones[final_locations]

#     pool_repr = (
#         jnp.zeros((n_zones, n_zones))
#         .at[next_zones, final_zones]
#         .add(
#             ~is_solo * (max_waypoints // 2 - active_trips)
#         )  # Remaining seats, solo dealt with separately
#         .reshape(-1)
#     )

#     solo_repr = jnp.zeros((n_zones,)).at[final_zones].add(is_solo)

#     # Intercept
#     return jnp.sqrt(jnp.concatenate((pool_repr, solo_repr)))


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
    repr = jnp.zeros((4,), dtype=jnp.float32).at[active_trips].add(1)
    return repr


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
    action_A = A.apply(env_params, dict(), obs, policy_key)
    action_B = B.apply(env_params, dict(), obs, policy_key)
    action, action_info = jax.lax.cond(
        is_treat, lambda: action_B, lambda: action_A
    )
    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)

    return (
        StepInfo(
            is_treat=is_treat,
            action_A=action_A,
            action_B=action_B,
            reward=reward,
        ),
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
    step_info, new_obs, new_state = collect_step(env, env_params, A, B, obs, state, subkey, p)
    new_ests = {
        est_name: est_update(
            ests[est_name], obs, step_info
        )
        for est_name, est_update in estimators.items()
    }
    return ((new_obs, new_state, new_ests, key), None)


def init_estimator_states(phi, r, z, p):
    d = phi.shape[0]
    return {
        "naive": NaiveEstimatorState.init(),
        "dqlstd": DQLSTDEstimatorState.init(d, phi, r, z, p),
        "dqlstdpg": DQLSTDPGEstimatorState.init(d, phi, r, z, p),
        "opelstd": OPELSTDEstimatorState.init(d, phi, r),
        # "opediffgq1": OPEDiffGQ1EstimatorState.init(d, phi, r),
        "dqmc-100": DQMCEstimatorState.init(100),
        "dqmc-100-99": DQMCEstimatorState.init(100),
        "dqmc-100-999": DQMCEstimatorState.init(100),
        "dqmc-100-995": DQMCEstimatorState.init(100),
        "dqmc-300": DQMCEstimatorState.init(300),
        "dqmc-300-99": DQMCEstimatorState.init(300),
        "dqmc-300-999": DQMCEstimatorState.init(300),
        "dqmc-300-995": DQMCEstimatorState.init(300),
        "dqmc-1000-20": DQMCEstimatorState.init(1000),
        "dqmc-3000-20": DQMCEstimatorState.init(3000),
        "dqmc-1000-30": DQMCEstimatorState.init(1000),
        "dqmc-3000-30": DQMCEstimatorState.init(3000),
    }


estimator_fns = {
    "naive": naive,
    "dqlstd": dqlstd,
    "dqlstdpg": dqlstdpg,
    "opelstd": opelstd,
    # "opediffgq1": opediffgq1,
    "dqmc-100": dqmc,
    "dqmc-100-99": dqmc,
    "dqmc-100-999": dqmc,
    "dqmc-100-995": dqmc,
    "dqmc-300": dqmc,
    "dqmc-300-99": dqmc,
    "dqmc-300-999": dqmc,
    "dqmc-300-995": dqmc,
    "dqmc-1000-20": dqmc,
    "dqmc-3000-20": dqmc,
    "dqmc-1000-30": dqmc,
    "dqmc-3000-30": dqmc,
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
        # "opediffgq1": Partial(
        #     opediffgq1_update, obs_to_repr_fn, 0.001, eta=0.01
        # ),
        "dqmc-100": Partial(dqmc_update, env_params.distances, p),
        "dqmc-100-99": Partial(
            dqmc_update, env_params.distances, p, gamma=0.99
        ),
        "dqmc-100-999": Partial(
            dqmc_update, env_params.distances, p, gamma=0.999
        ),
        "dqmc-100-995": Partial(
            dqmc_update, env_params.distances, p, gamma=0.995
        ),
        "dqmc-300": Partial(dqmc_update, env_params.distances, p),
        "dqmc-300-99": Partial(
            dqmc_update, env_params.distances, p, gamma=0.99
        ),
        "dqmc-300-999": Partial(
            dqmc_update, env_params.distances, p, gamma=0.999
        ),
        "dqmc-300-995": Partial(
            dqmc_update, env_params.distances, p, gamma=0.995
        ),
        "dqmc-1000-30": Partial(
            dqmc_update, env_params.distances, p, max_distance=30 * 60
        ),
        "dqmc-1000-20": Partial(
            dqmc_update, env_params.distances, p, max_distance=20 * 60
        ),
        "dqmc-3000-30": Partial(
            dqmc_update, env_params.distances, p, max_distance=30 * 60
        ),
        "dqmc-3000-20": Partial(
            dqmc_update, env_params.distances, p, max_distance=20 * 60
        ),
        "dqlstdpg": Partial(dqlstdpg_update, obs_to_repr_fn, p),
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
    step_infos_tuple = jax.vmap(
        collect_step, in_axes=(None, None, None, None, 0, 0, 0, None)
    )(env, env_params, A, B, obs0, state0, init_step_keys, p)
    step_infos, new_obs_batch, new_state_batch = step_infos_tuple

    init_ests = jax.vmap(
        init_estimator_states,
        in_axes=(0, 0, 0, None),
    )(
        jax.vmap(obs_to_repr_fn, in_axes=(0,))(new_obs_batch),
        step_infos.reward,
        step_infos.is_treat,
        p,
    )

    step_keys = jax.random.split(key, n_envs)
    init_carry = (
        new_obs_batch,
        new_state_batch,
        init_ests,
        step_keys,
    )
    vmap_scan = jax.vmap(scanner, in_axes=(0,))
    estimator_results = vmap_scan(init_carry)
    results = {
        est_name: jax.vmap(est, in_axes=(0,))(estimator_results[est_name])
        for est_name, est in estimator_fns.items()
    }
    for lam2 in range(12):
        lam = lam2 / 2
        for gam in [1.0, 0.999, 0.995, 0.99]:
            results[f"dqlstd-lam=1e{lam}-gam={gam}"] = jax.vmap(
                dqlstd, in_axes=(0, None, None)
            )(estimator_results["dqlstd"], 10**lam, gam)
            results[f"opelstd-lam=1e{lam}-gam={gam}"] = jax.vmap(
                opelstd, in_axes=(0, None, None)
            )(estimator_results["opelstd"], 10**lam, gam)

    return results


def load_taxi_zones():
    zones = pd.read_parquet("taxi-zones.parquet")
    neighborhoods = pd.read_csv("neighborhoods.csv")
    ids = pd.DataFrame({"neighborhood": neighborhoods["neighborhood"].unique()})
    ids["neighborhood_id"] = ids.index

    max_id = ids["neighborhood_id"].max()
    zones = pd.merge(zones, neighborhoods, on="zone", how="left").merge(
        ids, on="neighborhood", how="left"
    )
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
        (jnp.ones(len(nodes), dtype=jnp.int32) * (max_id + 1))
        .at[nodes_zones["idx"].values]
        .set(nodes_zones["neighborhood_id"].values)
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
    n_batches = int(np.ceil(k / batch_size))
    keys = jax.random.split(key, n_batches)

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
