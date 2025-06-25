import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from jaxtyping import Float, Integer, Bool
from typing import Tuple


@struct.dataclass
class DQMCEstimatorState:
    count: Integer[Array, "1"]  # Number of updates so far
    r: Float[Array, "window_size"]  # Rewards in the current window
    z: Float[Array, "window_size"]  # Treatment indicators in the current window
    sum_r1: Float[Array, "1"]  # Sum of all rewards seen
    sum_ipw1: Float[Array, "1"]
    sum_r0: Float[Array, "1"]  # Sum of all rewards seen
    sum_ipw0: Float[Array, "1"]
    sum_Q1: Float[Array, "1"]  # Sum of Q1 values seen
    sum_Q0: Float[Array, "1"]  # Sum of Q0 values seen
    sum_Qdiff: Float[Array, "1"]

    @classmethod
    def init(cls, window_size: int):
        return cls(
            count=jnp.array(0, dtype=jnp.int32),
            r=jnp.zeros(window_size),
            z=jnp.zeros(window_size),
            sum_r1=jnp.zeros(1),
            sum_ipw1=jnp.zeros(1),
            sum_r0=jnp.zeros(1),
            sum_ipw0=jnp.zeros(1),
            sum_Q1=jnp.zeros(1),
            sum_Q0=jnp.zeros(1),
            sum_Qdiff=jnp.zeros(1),
        )


def dqmc_update(
    # window_size: int,  # Size of the sliding window
    p: float,  # Probability of treatment
    est: DQMCEstimatorState,
    r: Float[Array, "1"],  # Reward at time t
    phi,
    z: Float[Array, "1"],  # Treatment indicator at time t
    Qbar=0  # Baseline for DR
) -> DQMCEstimatorState:
    window_size = est.r.shape[0]  # Use the size of the r array as window size
    ptr = est.count % window_size

    # Only start updating Q values after the first window is filled
    should_update = est.count >= window_size
    r1 = est.r * est.z / p
    sum_Q1 = est.sum_Q1 + should_update * (
        Qbar + est.z[ptr] / p * ((est.r[ptr] + (jnp.sum(r1) - r1[ptr])) - Qbar)
    )

    r0 = est.r * (1 - est.z) / (1 - p)
    sum_Q0 = est.sum_Q0 + should_update * (
        Qbar + (1 - est.z[ptr]) / (1 - p) * ((est.r[ptr] + (jnp.sum(r0) - r0[ptr])) - Qbar)
    )

    sum_Qdiff = est.sum_Qdiff + should_update * (
        jnp.sum(r1 - r0) - (r1[ptr] - r0[ptr])
    )


    return DQMCEstimatorState(
        count=est.count + 1,
        r=est.r.at[ptr].set(r),
        z=est.z.at[ptr].set(z),
        sum_r1=est.sum_r1 + r * z / p,
        sum_ipw1=est.sum_ipw1 + z / p,
        sum_r0=est.sum_r1 + r * (1 - z) / (1 - p),
        sum_ipw0=est.sum_ipw0 + (1 - z) / (1 - p),
        sum_Q1=sum_Q1,
        sum_Q0=sum_Q0,
        sum_Qdiff=sum_Qdiff,
    )


def dqmc(est: DQMCEstimatorState):
    window_size = est.r.shape[0]
    r1bar = est.sum_r1 / est.count
    Q1 = est.sum_Q1 - est.sum_ipw1 * r1bar * (window_size - 1)
    r0bar = est.sum_r0 / est.count
    Q0 = est.sum_Q0 - est.sum_ipw0 * r0bar * (window_size - 1)
    rdiff_bar = (est.sum_r1 - est.sum_r0) / est.count
    Qdiff = est.sum_Qdiff - est.count * rdiff_bar * (window_size - 1)
    return (Q1 - Q0 - Qdiff)[0] / (est.count - window_size)  # Average treatment effect
