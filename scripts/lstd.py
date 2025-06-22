import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from jaxtyping import Float, Integer, Bool
from typing import Tuple, Callable, Optional


@struct.dataclass
class LSTDEstimatorState:
    """
    State required for V function estimation using LSTD, with
    recursive updates via the Sherman-Morrison formula.
    """

    phi: Float[Array, "d"]  # Feature vector for time t
    r: Float[Array, "1"]  # reward at time t
    A: Float[Array, "d d"]
    sum_phi_r: Float[Array, "d"]
    sum_phi: Float[Array, "d"]
    sum_r: Float[Array, "1"]
    count: Integer[Array, "1"]

    @classmethod
    def init(cls, d, phi, r):
        """
        Initialize the LSTD estimator state.

        Args:
            d: Dimension of the feature vector
        """
        return cls(
            phi=phi,
            r=r,
            A=jnp.zeros((d, d)),
            sum_phi_r=jnp.zeros(d),
            sum_phi=jnp.zeros(d),
            sum_r=jnp.zeros(1),
            count=jnp.array(0, dtype=jnp.int32),
        )


def lstd_update(
    est: LSTDEstimatorState,
    phi: Float[Array, "d"],
    r: Float[Array, "1"],
):
    """
    Update LSTD parameters by directly updating A matrix

    Args:
        est: Current estimator state containing A matrix, b, and feature vector
        phi: Feature vector for current state
        r: Observed reward
    """
    return LSTDEstimatorState(
        phi=phi,
        r=r,
        A=est.A + jnp.outer(est.phi, est.phi - phi),
        sum_phi_r=est.sum_phi_r + est.phi * est.r,
        sum_phi=est.sum_phi + est.phi,
        sum_r=est.sum_r + est.r,
        count=est.count + 1,
    )


def lstd(
    est: LSTDEstimatorState,
) -> Tuple[Float[Array, "d"], Float[Array, "1"]]:
    A_aug = jnp.concatenate((est.A, jnp.expand_dims(est.sum_phi, 1)), axis=1)
    beta_and_rbar = jnp.linalg.pinv(A_aug) @ est.sum_phi_r
    beta = beta_and_rbar[:-1]
    rbar = beta_and_rbar[-1]
    return beta, rbar

    # Version using the average to estimate r
    # b = est.sum_phi_r - est.sum_phi * est.sum_r / est.count
    # beta = jnp.linalg.pinv(est.A) @ b
    # return beta


@struct.dataclass
class DQLSTDEstimatorState:
    lstd_tr: LSTDEstimatorState
    sum_phi_tr: Float[Array, "d"]
    sum_r_tr: Float[Array, "1"]
    lstd_co: LSTDEstimatorState
    sum_phi_co: Float[Array, "d"]
    sum_r_co: Float[Array, "1"]
    prev_z: Bool[Array, "1"]
    prev_r: Float[Array, "1"]

    @classmethod
    def init(cls, d: int, phi, r, z, p):
        """
        Initialize the DQLSTD estimator state.

        Args:
            d: Dimension of the feature vector
        """
        return cls(
            lstd_tr=LSTDEstimatorState.init(d, phi, z / p * r),
            sum_phi_tr=jnp.zeros(d),
            sum_r_tr=jnp.zeros((1,), dtype=jnp.float32),
            lstd_co=LSTDEstimatorState.init(d, phi, (1 - z) / (1 - p) * r),
            sum_phi_co=jnp.zeros(d),
            sum_r_co=jnp.zeros((1,), dtype=jnp.float32),
            prev_z=z,
            prev_r=r,
        )


def dqlstd_update(
    obs_to_repr: Callable[[Float[Array, "d_obs"]], Float[Array, "d"]],
    p: float,
    est: DQLSTDEstimatorState,
    reward: float,
    obs: Float[Array, "d_obs"],
    z: Bool,
) -> DQLSTDEstimatorState:
    """
    Update LSTD Q-learning parameters using Sherman-Morrison formula

    Args:
        est: Current estimator state containing A^-1, b, and feature vector
        reward: Observed reward
        obs: Feature vector for current state
    """
    phi = obs_to_repr(obs)

    prev_ips_tr = est.prev_z / p
    prev_r_tr = prev_ips_tr * est.prev_r
    phi_tr = prev_ips_tr * phi
    sum_phi_tr = est.sum_phi_tr + phi_tr
    sum_r_tr = est.lstd_tr.sum_r + prev_r_tr
    lstd_tr = lstd_update(est.lstd_tr, phi, z / p * reward)

    prev_ips_co = (1 - est.prev_z) / (1 - p)
    prev_r_co = prev_ips_co * est.prev_r
    phi_co = prev_ips_co * phi
    sum_phi_co = est.sum_phi_co + phi_co
    sum_r_co = est.lstd_co.sum_r + prev_r_co
    lstd_co = lstd_update(est.lstd_co, phi, (1 - z) / (1 - p) * reward)

    return DQLSTDEstimatorState(
        lstd_tr=lstd_tr,
        sum_phi_tr=sum_phi_tr,
        sum_r_tr=sum_r_tr,
        lstd_co=lstd_co,
        sum_phi_co=sum_phi_co,
        sum_r_co=sum_r_co,
        prev_z=z,
        prev_r=reward,
    )


def dqlstd(est: DQLSTDEstimatorState) -> Float[Array, "1"]:
    beta_tr, _ = lstd(est.lstd_tr)
    # V estimate is only correct up to constant, need to constrain so rho'V = 0
    bias_tr = est.lstd_tr.sum_phi @ beta_tr
    Q_tr = (
        est.sum_r_tr + est.sum_phi_tr @ beta_tr - bias_tr
    ) / est.lstd_tr.count

    beta_co, _ = lstd(est.lstd_co)
    bias_co = est.lstd_co.sum_phi @ beta_co
    Q_co = (
        est.sum_r_co + est.sum_phi_co @ beta_co - bias_co
    ) / est.lstd_co.count

    return (Q_tr - Q_co)[0]  # Difference between treatment and control
