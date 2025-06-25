from sacred import Experiment
import jax
import jax.numpy as jnp
import pandas as pd
from birth_death import BirthDeath
from typing import Dict, Tuple, Any
import chex
from functools import partial

ex = Experiment("compute-ate-birth-death")

@ex.config
def config():
    N = 1000  # Maximum population
    lam = 1.0  # Arrival rate
    mu = 1.0  # Service rate
    p0 = 0.315  # Success probability for action 0
    p1 = 0.3937  # Success probability for action 1
    episode_length = 1000  # Steps per episode
    num_episodes = 10  # Number of episodes to run
    output = "birth_death_ate_results.csv"  # Output file for results
    config_output = "birth_death_ate_config.csv"  # Output file for configuration

@partial(jax.jit, static_argnames=("env", "params", "episode_length"))
def run_policy(env, params, key, episode_length, p):
    """Run a single episode with fixed policy (p=0 or p=1)."""
    
    init_key, scan_key = jax.random.split(key)
    obs, state = env.reset(init_key, params)
    
    def step_fn(carry, t):
        obs, state, key = carry
        key, key_step = jax.random.split(key)
        # Fixed action based on p (0 or 1)
        action = jnp.array(p, dtype=jnp.int32)
        next_obs, next_state, reward, _, _ = env.step(key_step, state, action, params)
        return (next_obs, next_state, key), reward
    
    scan_keys = jax.random.split(scan_key, episode_length)
    (_, _, _), rewards = jax.lax.scan(
        step_fn, 
        (obs, state, scan_keys[0]),
        jnp.arange(episode_length)
    )
    
    return jnp.mean(rewards)

@ex.automain
def main(N, lam, mu, p0, p1, episode_length, num_episodes, output, config_output, _config):
    env = BirthDeath()
    params = env.default_params.replace(
        N=N,
        lam=lam,
        mu=mu,
        p0=p0,
        p1=p1
    )
    
    # Generate keys for all episodes
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_episodes)
    
    # Run policy with action=0 for all episodes
    rewards_0 = jax.vmap(run_policy, in_axes=(None, None, 0, None, None))(
        env, params, keys, episode_length, 0
    )
    
    # Run policy with action=1 for all episodes
    rewards_1 = jax.vmap(run_policy, in_axes=(None, None, 0, None, None))(
        env, params, keys, episode_length, 1
    )
    
    # Compute ATE
    ate = jnp.mean(rewards_1 - rewards_0)
    
    # Create results dataframe
    results = {
        "episode": jnp.arange(num_episodes),
        "reward_0": rewards_0,
        "reward_1": rewards_1,
        "diff": rewards_1 - rewards_0
    }
    
    # Save configuration
    pd.DataFrame.from_dict([_config]).to_csv(config_output, index=False)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output, index=False)
    
    # Print summary
    print(f"Average reward (action=0): {jnp.mean(rewards_0):.4f} ± {jnp.std(rewards_0):.4f}")
    print(f"Average reward (action=1): {jnp.mean(rewards_1):.4f} ± {jnp.std(rewards_1):.4f}")
    print(f"Average Treatment Effect: {ate:.4f}")
    
    return {
        "reward_0_mean": float(jnp.mean(rewards_0)),
        "reward_0_std": float(jnp.std(rewards_0)),
        "reward_1_mean": float(jnp.mean(rewards_1)),
        "reward_1_std": float(jnp.std(rewards_1)),
        "ate": float(ate)
    }
