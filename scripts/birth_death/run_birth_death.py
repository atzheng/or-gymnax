from sacred import Experiment
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import pandas as pd
from birth_death import BirthDeath
from flax.struct import dataclass
from step import StepInfo
# from lstd import (
#     DQLSTDEstimatorState,
#     dqlstd_update,
#     dqlstd,
#     OPELSTDEstimatorState,
#     opelstd_update,
#     opelstd,
# )
from tridiagonal_lstd import (
    DQTridiagLSTDEstimatorState,
    dq_tridiag_lstd_update,
    dq_tridiag_lstd,
    OPETridiagLSTDEstimatorState,
    ope_tridiag_lstd_update,
    ope_tridiag_lstd,
)
from typing import Dict, Tuple, Any
import chex
from functools import partial

ex = Experiment("birth-death")


@ex.config
def config():
    N = 500  # Maximum population
    lam = 1.0  # Arrival rate
    mu = 1.0  # Service rate
    p0 = 0.315  # Success probability for action 0
    p1 = 0.3937  # Success probability for action 1
    episode_length = 10000  # Steps per episode
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


def init_estimator_states(N, obs, r, z, p):
    return {
        "naive": NaiveEstimatorState.init(),
        "dqtridiag": DQTridiagLSTDEstimatorState.init(N + 1, obs.astype(jnp.int32), r, z, p),
        "opetridiag": OPETridiagLSTDEstimatorState.init(N + 1, obs.astype(jnp.int32), r, z),
    }


@partial(
    jax.jit, static_argnames=("env", "params", "num_episodes", "episode_length")
)
def run_episodes(env, params, seed, num_episodes=100, episode_length=1000, p=0.5):
    """Run multiple episodes and return dictionary of estimator results."""

    estimators = {
        "naive": naive_update,
        # Memory requirements for typical DQLSTD / OPELSTD too high
        "dqtridiag": partial(dq_tridiag_lstd_update, p),
        "opetridiag": ope_tridiag_lstd_update,
    }

    estimator_fns = {
        "naive": naive,
        "dqtridiag": dq_tridiag_lstd,
        "opetridiag": ope_tridiag_lstd,
    }

    def run_episode(key):
        init_key, scan_key = jax.random.split(key)
        obs, state = env.reset(init_key, params)

        # First step to initialize estimators
        key, key_action = jax.random.split(scan_key)
        z = jax.random.bernoulli(key_action, p=p)
        action = z #.astype(jnp.int32)
        key, key_step = jax.random.split(key)
        next_obs, next_state, reward, _, _ = env.step(
            key_step, state, action, params
        )

        # Initialize estimators
        ests = init_estimator_states(params.N, next_obs, jnp.array(reward), z, p)

        def step_fn(carry, t):
            obs, state, ests, key = carry
            key, key_action = jax.random.split(key)
            z = jax.random.bernoulli(key_action, p=p)
            action = z
            key, key_step = jax.random.split(key)
            next_obs, next_state, reward, _, _ = env.step(
                key_step, state, action, params
            )

            # Update all estimators
            new_ests = {
                name: update_fn(ests[name], reward, obs.astype(jnp.int32), z)
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
        for lam2 in range(-12, 6):  # -5 to +3 in log scale
            lam = 10 ** (lam2 / 2)  # Ranges from 1e-3 to 1e+2.5
            opetridiag_name = f"opetridiag-lam={lam:.1e}"
            results[opetridiag_name] = ope_tridiag_lstd(
                final_ests["opetridiag"], lam
            )
            for gamma in [1.0, 0.999, 0.995, 0.99]:
                # Tridiagonal DQLSTD with different parameters
                dqtridiag_name = f"dqtridiag-lam={lam:.1e}-gam={gamma}"
                results[dqtridiag_name] = dq_tridiag_lstd(
                    final_ests["dqtridiag"], lam, gamma
                )

                # Tridiagonal OPELSTD with different parameters

        return results

    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    episode_results = jax.vmap(run_episode)(keys)

    # Average results across episodes
    return episode_results


@ex.automain
def main(
    N,
    lam,
    mu,
    p0,
    p1,
    episode_length,
    num_episodes,
    p,
    output,
    config_output,
    _config,
    _seed
):
    env = BirthDeath()
    params = env.default_params.replace(N=N, lam=lam, mu=mu, p0=p0, p1=p1)

    results = run_episodes(
        env,
        params,
        seed=_seed,
        episode_length=episode_length,
        num_episodes=num_episodes,
        p=p,
    )

    # Save configuration
    pd.DataFrame.from_dict([_config]).to_csv(config_output, index=False)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame({name: results[name] for name in results.keys()})
    results_df.to_csv(output, index=False)

    return results
