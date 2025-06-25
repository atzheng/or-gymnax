from sacred import Experiment
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import pandas as pd
from birth_death import BirthDeath
from flax.struct import dataclass
from lstd import (
    DQLSTDEstimatorState,
    dqlstd_update,
    dqlstd,
    OPELSTDEstimatorState,
    opelstd_update,
    opelstd,
)
from sparse_lstd import (
    SparseDQLSTDEstimatorState,
    sparse_dqlstd_update,
    sparse_dqlstd,
    SparseOPELSTDEstimatorState,
    sparse_opelstd_update,
    sparse_opelstd,
)
from typing import Dict, Tuple, Any
import chex
from functools import partial

ex = Experiment("birth-death")

@ex.config
def config():
    N = 1000  # Maximum population
    lam = 1.0  # Arrival rate
    mu = 1.0  # Service rate
    p0 = 0.315  # Success probability for action 0
    p1 = 0.3937  # Success probability for action 1
    episode_length = 1000  # Steps per episode
    num_episodes = 10  # Number of episodes to run
    p = 0.5  # Treatment probability
    output = "birth_death_results.csv"  # Output file for results
    config_output = "birth_death_config.csv"  # Output file for configuration


@dataclass
class NaiveEstimatorState:
    """Contains the current state of naive estimator"""

    counts: jnp.ndarray  # Shape (2,) for counting treatments
    rewards: jnp.ndarray  # Shape (2,) for summing rewards

    @classmethod
    def init(cls):
        return cls(
            counts=jnp.zeros(2, dtype=jnp.int32),
            rewards=jnp.zeros(2, dtype=jnp.float32),
        )


def naive_update(
    est: NaiveEstimatorState, reward: float, obs: jnp.ndarray, z: bool
):
    return NaiveEstimatorState(
        counts=est.counts.at[z.astype(jnp.int32)].add(1),
        rewards=est.rewards.at[z.astype(jnp.int32)].add(reward),
    )


def naive(est: NaiveEstimatorState):
    avg_rewards = est.rewards / (est.counts + 1e-8)
    return avg_rewards[1] - avg_rewards[0]


def obs_to_repr(N, obs):
    """Convert observation to sparse one-hot representation"""
    s = obs[0].astype(jnp.int32)
    # Create sparse one-hot vector
    indices = jnp.array([[s]])
    data = jnp.ones(1, dtype=jnp.float32)
    return jsparse.BCOO((data, indices), shape=(N + 1,))


def init_estimator_states(phi, r, z, p):
    d = phi.shape[0]
    return {
        "naive": NaiveEstimatorState.init(),
        # "dqlstd": DQLSTDEstimatorState.init(d, phi, r, z, p),
        # "opelstd": OPELSTDEstimatorState.init(d, phi, r),
        "sparse_dqlstd": SparseDQLSTDEstimatorState.init(d, phi, r, z, p),
        "sparse_opelstd": SparseOPELSTDEstimatorState.init(d, phi, r),
    }


@partial(
    jax.jit, static_argnames=("env", "params", "num_episodes", "episode_length")
)
def run_episodes(env, params, num_episodes=100, episode_length=1000, p=0.5):
    """Run multiple episodes and return dictionary of estimator results."""

    estimators = {
        "naive": naive_update,
        # "dqlstd": partial(dqlstd_update, partial(obs_to_repr, params.N), p),
        # "opelstd": partial(opelstd_update, partial(obs_to_repr, params.N)),
        "sparse_dqlstd": partial(sparse_dqlstd_update, partial(obs_to_repr, params.N), p),
        "sparse_opelstd": partial(sparse_opelstd_update, partial(obs_to_repr, params.N)),
    }

    estimator_fns = {
        "naive": naive,
        # "dqlstd": dqlstd,
        # "opelstd": opelstd,
        "sparse_dqlstd": sparse_dqlstd,
        "sparse_opelstd": sparse_opelstd,
    }

    def run_episode(key):
        init_key, scan_key = jax.random.split(key)
        obs, state = env.reset(init_key, params)

        # First step to initialize estimators
        key, key_action = jax.random.split(scan_key)
        z = jax.random.bernoulli(key_action, p=p)
        action = z.astype(jnp.int32)
        key, key_step = jax.random.split(key)
        next_obs, next_state, reward, _, _ = env.step(
            key_step, state, action, params
        )

        # Initialize estimators
        ests = init_estimator_states(
            obs_to_repr(params.N, next_obs), jnp.array(reward), z, p
        )

        def step_fn(carry, t):
            obs, state, ests, key = carry
            key, key_action = jax.random.split(key)
            z = jax.random.bernoulli(key_action, p=p)
            action = z.astype(jnp.int32)
            key, key_step = jax.random.split(key)
            next_obs, next_state, reward, _, _ = env.step(
                key_step, state, action, params
            )

            # Update all estimators
            new_ests = {
                name: update_fn(ests[name], reward, obs, z)
                for name, update_fn in estimators.items()
            }

            return (next_obs, next_state, new_ests, key), None

        scan_keys = jax.random.split(scan_key, episode_length)
        (_, _, final_ests, _), _ = jax.lax.scan(
            step_fn,
            (next_obs, next_state, ests, scan_keys[0]),
            jnp.arange(episode_length),
        )

        results = {
            name: fn(final_ests[name]) for name, fn in estimator_fns.items()
        }

        # Add LSTD estimates with different parameters
        for lam2 in range(12):  # -6 to +5 in log scale
            lam = 10 ** ((lam2 - 6) / 2)  # Ranges from 1e-3 to 1e+2.5
            for gamma in [1.0, 0.999, 0.995, 0.99]:
                # DQLSTD with different parameters
                dqlstd_name = f"dqlstd-lam={lam:.1e}-gam={gamma}"
                results[dqlstd_name] = dqlstd(final_ests["dqlstd"], lam, gamma)

                # OPELSTD with different parameters
                opelstd_name = f"opelstd-lam={lam:.1e}-gam={gamma}"
                results[opelstd_name] = opelstd(final_ests["opelstd"], lam, gamma)
                
                # Sparse DQLSTD with different parameters
                sparse_dqlstd_name = f"sparse_dqlstd-lam={lam:.1e}-gam={gamma}"
                results[sparse_dqlstd_name] = sparse_dqlstd(final_ests["sparse_dqlstd"], lam, gamma)
                
                # Sparse OPELSTD with different parameters
                sparse_opelstd_name = f"sparse_opelstd-lam={lam:.1e}-gam={gamma}"
                results[sparse_opelstd_name] = sparse_opelstd(final_ests["sparse_opelstd"], lam, gamma)

        return results


    keys = jax.random.split(jax.random.PRNGKey(0), num_episodes)
    episode_results = jax.vmap(run_episode)(keys)

    # Average results across episodes
    return episode_results


@ex.automain
def main(N, lam, mu, p0, p1, episode_length, num_episodes, p, output, config_output, _config):
    env = BirthDeath()
    params = env.default_params.replace(
        N=N,
        lam=lam,
        mu=mu,
        p0=p0,
        p1=p1
    )
    
    results = run_episodes(
        env, 
        params, 
        episode_length=episode_length, 
        num_episodes=num_episodes, 
        p=p
    )
    
    # Save configuration
    pd.DataFrame.from_dict([_config]).to_csv(config_output, index=False)
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(
        {name: results[name] for name in results.keys()}
    )
    results_df.to_csv(output, index=False)
    
    return results
