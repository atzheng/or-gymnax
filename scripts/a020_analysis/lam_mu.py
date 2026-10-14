"""Market balance lambda/mu for the pooled rideshare env under the all-A (threshold 0.2) policy.
Little's law: lambda/mu = (arrivals per car) x (car busy time per request)
            = (mean # busy cars / n_cars) / (fraction of requests served) ... per served request,
and as Johari et al. define lambda per listing over ALL arrivals: lambda/mu = util / served_frac * served_frac
-> we report both util and served fraction."""
import sys, numpy as np, jax, jax.numpy as jnp
from xp_gym.environments.rideshare import XPRidesharePoolDispatchEnv
from or_gymnax.rideshare_pool import obs_to_state
sta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
env = XPRidesharePoolDispatchEnv(n_cars=300, n_events=500000, savings_threshold_A=sta, savings_threshold_B=sta)
p0 = env.default_params
params = p0.replace(env_params=p0.env_params.replace(max_active_trips=2, max_groups=1, max_ghosts_per_group=1, ghost_max_lifespan=1))
obs, state = env.reset(jax.random.PRNGKey(0), params)

@jax.jit
def run(state, keys):
    def f(s, k):
        o, s2, r, _, info = env.step(k, s, jnp.array(0), params)
        ev, W, T = obs_to_state(300, 4, o)
        busy = (T.max(axis=1) > ev.t).sum()
        car = info["action_A"].reshape(-1)[0]
        return s2, (busy, car >= 0, ev.t)
    return jax.lax.scan(f, state, keys)

_, (busy, served, t) = run(state, jax.random.split(jax.random.PRNGKey(1), N))
busy, served, t = (np.asarray(x) for x in (busy, served, t))
b = 5 * 300  # burn-in
util = busy[b:].mean() / 300
dt = np.diff(t[b:]).mean()
print(f"A={sta}: mean inter-arrival {dt:.3f}s, utilization {util:.3f}, served frac {served[b:].mean():.3f}")
print("lambda/mu = util / served =", round(util / served[b:].mean(), 3))
print("per 50k window: util, served, lambda/mu:", [(round(busy[i:i+50000].mean()/300, 3), round(served[i:i+50000].mean(), 3), round(busy[i:i+50000].mean()/300 / served[i:i+50000].mean(), 2)) for i in range(0, N, 50000)])
