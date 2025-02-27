import jax
from jax.experimental import sparse
from functools import partial
from picard.environments.rideshare_dispatch import (
    ManhattanRideshareDispatch,
    ManhattanRidesharePricing,
    GreedyPolicy,
    SimplePricingPolicy,
    EnvParams,
    obs_to_state,
    RideshareEvent,
)
from or_gymnax.nn import Policy
from jax import numpy as jnp
from typing import Dict, Callable, Tuple
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
import haversine


ex = Experiment("rideshares")


@ex.config
def config():
    n_cars = 300  # Number of cars
    # Pricing choice model parameters
    w_price = -0.3
    w_eta = -0.005
    w_intercept = 4
    n_events = 10000  # Number of events to simulate per trial
    k = 1000  # Total number of trials
    batch_size = 100  # Number of environments to run in parallel
    switch_every = 1000  # Switchback duration
    p = 0.5  # Treatment probability
    output = "results.csv"
    config_output = "config.csv"
    max_km = 2
    lookahead_seconds = 600


@struct.dataclass
class ExperimentInfo:
    """
    Contains treatment assignment and cluster information for each step
    """

    t: Integer[Array, "n_steps"]
    space_id: Integer[Array, "n_steps"]
    cluster_id: Integer[Array, "n_steps"]
    is_treat: Bool[Array, "n_steps"]
    key: chex.PRNGKey


@struct.dataclass
class EstimatorState:
    """
    Contains the current state of an estimator
    """

    counts: Integer[Array, "n_clusters"]
    estimates: Float[Array, "n_clusters"]


def naive_update(
    est: EstimatorState, reward: float, info: ExperimentInfo, p=0.5
):
    return EstimatorState(
        est.counts.at[info.cluster_id].add(1),
        est.estimates.at[info.cluster_id].add(reward),
    )


def naive(est: EstimatorState, z: Bool[Array, "n_clusters"], p=0.5):
    eta = z / p - (1 - z) / (1 - p)
    N = est.counts.sum()
    return (eta * est.estimates).sum() / N


def dn_update(
    time_ids: Integer[Array, "n_clusters"],
    space_ids: Integer[Array, "n_clusters"],
    space_adj: Bool[Array, "n_spaces n_spaces"],
    est: EstimatorState,
    reward: float,
    info: ExperimentInfo,
    p=0.5,
    lookahead_seconds=600,
    switch_every=1
):
    # Eta to be included later
    z = info.is_treat
    xi = z * (1 - p) / p + (1 - z) * p / (1 - p)
    update_val = xi * reward
    is_adjacent_t = (time_ids >= info.t - lookahead_seconds) & (
        time_ids <= (info.t // switch_every + 1) * switch_every
    )
    is_adjacent_space = space_adj[space_ids, info.space_id]
    # jax.debug.print("{eea}", eea=(is_adjacent_space & is_adjacent_t).sum())
    update_ests = est.estimates + jnp.where(
        is_adjacent_t & is_adjacent_space,
        update_val,
        0.0,
    )
    update_ests = update_ests.at[info.cluster_id].add(reward - update_val)
    return EstimatorState(est.counts.at[info.cluster_id].add(1), update_ests)


def dn(est: EstimatorState, z: Bool[Array, "n_clusters"], p=0.5):
    N = est.counts.sum()
    mask = est.counts > 0
    eta = z / p - (1 - z) / (1 - p)
    avg_y = (mask * est.estimates).sum() / est.counts.sum()
    baseline = est.counts * avg_y
    return (mask * eta * (est.estimates - baseline)).sum() / N


estimator_fns = {
    "naive": naive,
    "dn": dn,
}


def stepper(
    estimators: Dict[str, Callable],
    env,
    env_params,
    A: Policy,
    B: Policy,
    carry: Tuple[Array, Array, Dict[str, EstimatorState]],
    info: ExperimentInfo,
):
    obs, state, ests = carry
    key, policy_key = jax.random.split(info.key)
    action, action_info = jax.lax.cond(
        info.is_treat,
        lambda: B.apply(env_params, dict(), obs, policy_key),
        lambda: A.apply(env_params, dict(), obs, policy_key),
    )

    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)

    new_ests = {
        est_name: est_update(ests[est_name], reward, info)
        for est_name, est_update in estimators.items()
    }

    return ((new_obs, new_state, new_ests), None)


def run_trials(
    env,
    env_params,
    A,
    B,
    key,
    n_envs=10,
    n_steps=1000,
    switch_every=1,
    p=0.5,
    spatial_clusters=None,
    space_adj=None,
    lookahead_seconds=600,
):
    time_ids = (
        env_params.events.t // switch_every + 1
    ) * switch_every  # Identifies the end of the period
    space_ids = spatial_clusters.loc[env_params.events.src]["zone_id"].values
    ab_key, key = jax.random.split(key)
    unq_times, unq_time_ids = jnp.unique(time_ids, return_inverse=True)
    # unq_spaces, unq_space_ids = jnp.unique(space_ids, return_inverse=True)
    unq_spaces = jnp.arange(spatial_clusters["zone_id"].max() + 1)
    cluster_ids = unq_time_ids * len(unq_spaces) + space_ids
    n_clusters = len(unq_times) * len(unq_spaces)
    cluster_treats = jax.random.bernoulli(ab_key, p, (n_envs, n_clusters))
    is_treat = cluster_treats[:, cluster_ids]

    reset_key, step_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_envs)
    step_keys = jax.random.split(step_key, (n_envs, n_steps))

    estimators = {
        "naive": partial(naive_update, p=p),
        "dn": partial(
            dn_update,
            jnp.repeat(unq_times, len(unq_spaces)),
            jnp.tile(unq_spaces, len(unq_times)),
            space_adj,
            p=p,
            lookahead_seconds=lookahead_seconds,
            switch_every=switch_every,
        ),
    }

    # Initial inputs
    init_ests = {
        est_name: EstimatorState(
            jnp.zeros((n_envs, n_clusters)), jnp.zeros((n_envs, n_clusters))
        )
        for est_name in estimators.keys()
    }

    infos = ExperimentInfo(
        t=jnp.tile(env_params.events.t.reshape(1, -1), (n_envs, 1)),
        space_id=jnp.tile(space_ids.reshape(1, -1), (n_envs, 1)),
        cluster_id=jnp.tile(cluster_ids.reshape(1, -1), (n_envs, 1)),
        is_treat=is_treat,
        key=step_keys,
    )

    def scanner(obs_and_state_and_ests, infos):
        return jax.lax.scan(
            partial(stepper, estimators, env, env_params, A, B),
            obs_and_state_and_ests,
            infos,
        )[0][2]

    obs_and_states = jax.vmap(env.reset, in_axes=(0, None))(
        reset_keys, env_params
    )
    obs_and_states_and_ests = (*obs_and_states, init_ests)
    vmap_scan = jax.vmap(scanner, in_axes=(0, 0))
    estimator_results = vmap_scan(obs_and_states_and_ests, infos)
    return {
        est_name: jax.vmap(estimator_fns[est_name], in_axes=(0, 0, None))(
            est_state, cluster_treats, p
        )
        for est_name, est_state in estimator_results.items()
    }


def load_spatial_clusters():
    zones = pd.read_parquet("taxi-zones.parquet")
    unq_zones, unq_zone_ids = np.unique(zones["zone"], return_inverse=True)
    zones["zone_id"] = unq_zone_ids
    nodes = pd.read_parquet("manhattan-nodes.parquet")
    nodes["lng"] = nodes["lng"].astype(float)
    nodes["lat"] = nodes["lat"].astype(float)
    nodes_zones = nodes.merge(zones, on="osmid")

    centroids = nodes_zones.groupby("zone_id").aggregate(
        {"lat": "mean", "lng": "mean"}
    )
    dist = np.zeros((len(centroids), len(centroids)))
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            dist[i, j] = haversine.haversine(
                (centroids.iloc[i]["lat"], centroids.iloc[i]["lng"]),
                (centroids.iloc[j]["lat"], centroids.iloc[j]["lng"]),
            )
    return nodes_zones, dist


@ex.automain
def main(
    n_cars,
    w_price,
    w_eta,
    w_intercept,
    n_events,
    seed,
    k,
    batch_size,
    switch_every,
    p,
    output,
    _config,
    max_km,
    lookahead_seconds,
):
    key = jax.random.PRNGKey(seed)
    env = ManhattanRidesharePricing(n_cars=n_cars, n_events=n_events)
    env_params = env.default_params
    env_params = env_params.replace(
        w_price=w_price, w_eta=w_eta, w_intercept=w_intercept
    )

    nodes_zones, zone_dists = load_spatial_clusters()

    A = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.01)
    B = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.02)
    print(
        "Simulation time (mins)",
        (env_params.events.t.max() - env_params.events.t[5]) / 60,
    )
    print(
        "Simulation time (hrs)",
        (env_params.events.t.max() - env_params.events.t[5]) / 3600,
    )

    unq_times = jnp.unique(env_params.events.t // switch_every)
    n_times = unq_times.shape[0]
    n_spaces = zone_dists.shape[0]
    n_clusters = n_times * n_spaces

    all_results = []
    keys = jax.random.split(key, k // batch_size + 1)
    for key in tqdm(keys):
        ests = run_trials(
            env,
            env_params,
            A,
            B,
            key,
            n_envs=batch_size,
            n_steps=n_events,
            switch_every=switch_every,
            p=p,
            spatial_clusters=nodes_zones,
            space_adj=jnp.asarray(zone_dists < max_km),
            lookahead_seconds=lookahead_seconds,
        )
        all_results.append(ests)

    pd.DataFrame.from_dict([_config]).to_csv(
        _config["config_output"], index=False
    )
    results_df = pd.concat(map(pd.DataFrame, all_results))
    results_df.to_csv(output, index=False)
