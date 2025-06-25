import jax
from functools import partial
from or_gymnax.rideshare_pool import (
    ManhattanRidesharePoolDispatch,
    GreedyPolicy,
)
from sacred import Experiment
import pandas as pd
from tqdm import tqdm
from jax.experimental import checkify

ex = Experiment("compute-ate")


@ex.config
def config():
    n_cars = 300  # Number of cars
    # Pricing choice model parameters
    max_active_trips = 3
    n_events = 10000  # Number of events to simulate per trial
    batch_size = 100  # Number of environments to run in parallel
    k = 100  # Total number of trials
    output = "ate.csv"
    savings_threshold_A = 0.0
    savings_threshold_B = 0.5


def stepper(env, env_params, policy, carry, key):
    obs, state, total_reward = carry
    key, policy_key = jax.random.split(key)
    action, action_info = policy.apply(env_params, dict(), obs, policy_key)
    new_obs, new_state, reward, done, info = env.step(
        key, state, action, env_params
    )
    info["reward"] = reward
    info["action"] = action
    return (new_obs, new_state, total_reward + reward), info


def run(env, env_params, policy, key, n_steps):
    keys = jax.random.split(key, n_steps)
    obs, state = env.reset(key, env_params)

    final, _ = jax.lax.scan(
        partial(stepper, env, env_params, policy),
        (obs, state, 0),
        keys,
    )
    _, _, total_reward = final
    return total_reward / n_steps


vmap_run = jax.vmap(run, in_axes=(None, None, None, 0, None))


def run_batch(env, env_params, A, B, key, n_steps, batch_size):
    keys = jax.random.split(key, batch_size)
    results_A = vmap_run(env, env_params, A, keys, n_steps)
    results_B = vmap_run(env, env_params, B, keys, n_steps)
    return {
        "A": results_A,
        "B": results_B,
    }


@ex.automain
def main(
    n_cars,
    n_events,
    max_active_trips,
    k,
    output,
    seed,
    batch_size,
    savings_threshold_A,
    savings_threshold_B,
):
    env = ManhattanRidesharePoolDispatch(
        n_cars=n_cars, n_events=n_events, uniformize=True
    )
    env_params = env.default_params
    env_params = env_params.replace(max_active_trips=max_active_trips)

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

    n_batches = k // batch_size
    keys = jax.random.split(jax.random.PRNGKey(seed), n_batches)
    results = [
        run_batch(env, env_params, A, B, key, n_events, batch_size)
        for key in tqdm(keys)
    ]
    results_df = pd.concat(map(pd.DataFrame, results))
    results_df.to_csv(output, index=False)

    # keys = jax.random.split(jax.random.PRNGKey(seed), n_events)
    # obs, state = env.reset(jax.random.PRNGKey(seed), env_params)

    # final, infos = jax.lax.scan(
    #     partial(stepper, env, env_params, B),
    #     (obs, state, 0),
    #     keys,
    # )

    # pd.DataFrame(infos).to_csv(output)
