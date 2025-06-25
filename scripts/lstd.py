import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from jaxtyping import Float, Integer, Bool
from typing import Tuple, Callable, Optional


# Base LSTD
# -------------------------------------------------------------------------


@struct.dataclass
class LSTDEstimatorState:
    """
    State required for V function estimation using LSTD, with
    recursive updates via the Sherman-Morrison formula.
    """

    phi: Float[Array, "d"]  # Feature vector for time t
    r: Float[Array, "1"]  # reward at time t
    phitphi: Float[Array, "d d"]
    phitphi_next: Float[Array, "d d"]
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
            phitphi=jnp.zeros((d, d)),
            phitphi_next=jnp.zeros((d, d)),
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
        phitphi=est.phitphi + jnp.outer(est.phi, est.phi),
        phitphi_next=est.phitphi_next + jnp.outer(est.phi, phi),
        sum_phi_r=est.sum_phi_r + est.phi * est.r,
        sum_phi=est.sum_phi + est.phi,
        sum_r=est.sum_r + est.r,
        count=est.count + 1,
    )


def lstd(
    est: LSTDEstimatorState,
    lam=1e-4,  # Regularization parameter
    gamma=1.0,  # Discount factor for next state
) -> Tuple[Float[Array, "d"], Float[Array, "1"]]:
    # import jax
    # import jax.numpy as jnp
    # A_aug = jnp.concatenate(
    #     (
    #         est.A + lam * jnp.eye(est.A.shape[0]),
    #         jnp.expand_dims(est.sum_phi, 1),
    #     ),
    #     axis=1,
    # )
    # beta_and_rbar = jnp.linalg.pinv(A_aug) @ est.sum_phi_r
    # beta = beta_and_rbar[:-1]
    # rbar = beta_and_rbar[-1]

    # Version using the average to estimate r
    rbar = est.sum_r / est.count  # TODO This will be wrong for OPE, need to fix
    b = est.sum_phi_r - est.sum_phi * est.sum_r / est.count
    beta = jnp.linalg.lstsq(
        est.phitphi
        - gamma * est.phitphi_next
        + lam * jnp.eye(est.phitphi.shape[0]),
        b,
    )[0]
    # jax.debug.breakpoint()
    return beta, rbar

    # Version using the average to estimate r
    # b = est.sum_phi_r - est.sum_phi * est.sum_r / est.count
    # beta = jnp.linalg.pinv(est.A) @ b
    # return beta


def ope_lstd(
    est: LSTDEstimatorState,
    lam=1e-4,  # Regularization parameter
    gamma=1.0,  # Discount factor for next state
) -> Tuple[Float[Array, "d"], Float[Array, "1"]]:
    A_aug = jnp.concatenate(
        (
            est.phitphi
            - gamma * est.phitphi_next
            + lam * jnp.eye(est.phitphi.shape[0]),
            jnp.expand_dims(est.sum_phi, 1),
        ),
        axis=1,
    )
    beta_and_rbar = jnp.linalg.lstsq(A_aug, est.sum_phi_r)[0]
    beta = beta_and_rbar[:-1]
    rbar = beta_and_rbar[-1]
    return beta, rbar


# DQ LSTD
# -------------------------------------------------------------------------
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


def dqlstd(est: DQLSTDEstimatorState, lam=1e-4, gamma=1.0) -> Float[Array, "1"]:
    beta_tr, _ = lstd(est.lstd_tr, lam=lam, gamma=gamma)
    # V estimate is only correct up to constant, need to constrain so rho'V = 0
    bias_tr = est.lstd_tr.sum_phi @ beta_tr
    Q_tr = (
        est.sum_r_tr + est.sum_phi_tr @ beta_tr - bias_tr
    ) / est.lstd_tr.count

    beta_co, _ = lstd(est.lstd_co, lam=lam, gamma=gamma)
    bias_co = est.lstd_co.sum_phi @ beta_co
    Q_co = (
        est.sum_r_co + est.sum_phi_co @ beta_co - bias_co
    ) / est.lstd_co.count

    return (Q_tr - Q_co)[0]  # Difference between treatment and control


# DQLSTD-PG
# -------------------------------------------------------------------------
@struct.dataclass
class DQLSTDPGEstimatorState:
    lstd: LSTDEstimatorState
    sum_r_tr: Float[Array, "1"]
    sum_r_co: Float[Array, "1"]
    sum_phi_tr: Float[Array, "d"]
    sum_phi_co: Float[Array, "d"]
    prev_z: Bool[Array, "1"]

    @classmethod
    def init(cls, d: int, phi, r, z, p):
        """
        Initialize the DQLSTD estimator state.

        Args:
            d: Dimension of the feature vector
        """
        return cls(
            lstd=LSTDEstimatorState.init(d, phi, r),
            sum_r_tr=jnp.zeros((1,), dtype=jnp.float32),
            sum_r_co=jnp.zeros((1,), dtype=jnp.float32),
            sum_phi_tr=jnp.zeros(d),
            sum_phi_co=jnp.zeros(d),
            prev_z=jnp.array(z, dtype=jnp.bool_),
        )


def dqlstdpg_update(
    obs_to_repr: Callable[[Float[Array, "d_obs"]], Float[Array, "d"]],
    p: float,
    est: DQLSTDPGEstimatorState,
    reward: Float[Array, "1"],
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
    phi_tr = prev_ips_tr * phi
    sum_phi_tr = est.sum_phi_tr + phi_tr
    sum_r_tr = est.sum_r_tr + prev_ips_tr * est.lstd.r

    prev_ips_co = (1 - est.prev_z) / (1 - p)
    phi_co = prev_ips_co * phi
    sum_phi_co = est.sum_phi_co + phi_co
    sum_r_co = est.sum_r_co + prev_ips_co * est.lstd.r

    lstd = lstd_update(est.lstd, phi, reward)

    return DQLSTDPGEstimatorState(
        lstd=lstd,
        sum_r_tr=sum_r_tr,
        sum_r_co=sum_r_co,
        sum_phi_tr=sum_phi_tr,
        sum_phi_co=sum_phi_co,
        prev_z=z,
    )


def dqlstdpg(est: DQLSTDPGEstimatorState, lam=1e-4) -> Float[Array, "1"]:
    beta, _ = lstd(est.lstd, lam=lam)
    # V estimate is only correct up to constant, need to constrain so rho'V = 0
    return (
        (est.sum_r_tr - est.sum_r_co) / est.lstd.count
        + ((est.sum_phi_tr - est.sum_phi_co) / est.lstd.count) @ beta
    )[0]


# OPE LSTD
# -------------------------------------------------------------------------
@struct.dataclass
class OPELSTDEstimatorState:
    lstd_tr: LSTDEstimatorState
    lstd_co: LSTDEstimatorState
    prev_z: Bool[Array, "1"]
    prev_r: Float[Array, "1"]
    prev_phi: Float[Array, "d"]

    @classmethod
    def init(cls, d: int, phi, r):
        """
        Initialize the OPELSTD estimator state.

        Args:
            d: Dimension of the feature vector
        """
        return cls(
            lstd_tr=LSTDEstimatorState.init(d, phi, r),
            lstd_co=LSTDEstimatorState.init(d, phi, r),
            prev_z=jnp.array(False, dtype=jnp.bool_),
            prev_r=jnp.array(0.0, dtype=jnp.float32),
            prev_phi=jnp.zeros(d, dtype=jnp.float32),
        )


def opelstd_update(
    obs_to_repr: Callable[[Float[Array, "d_obs"]], Float[Array, "d"]],
    est: OPELSTDEstimatorState,
    reward: Float[Array, "1"],
    obs: Float[Array, "d_obs"],
    z: Bool,
):
    phi = obs_to_repr(obs)

    new_lstd_tr = lstd_update(
        est.lstd_tr.replace(phi=est.prev_phi, r=est.prev_r), phi, reward
    )

    new_lstd_co = lstd_update(
        est.lstd_co.replace(phi=est.prev_phi, r=est.prev_r), phi, reward
    )

    return OPELSTDEstimatorState(
        lstd_tr=jax.lax.cond(
            est.prev_z, lambda: new_lstd_tr, lambda: est.lstd_tr
        ),
        lstd_co=jax.lax.cond(
            ~est.prev_z, lambda: new_lstd_co, lambda: est.lstd_co
        ),
        prev_z=z,
        prev_r=reward,
        prev_phi=phi,
    )


def opelstd(est: OPELSTDEstimatorState, lam=1e-4, gamma=1.) -> Float[Array, "1"]:
    _, rho_tr = ope_lstd(est.lstd_tr, lam=lam, gamma=gamma)
    _, rho_co = ope_lstd(est.lstd_co, lam=lam, gamma=gamma)
    return rho_tr - rho_co
