import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from jaxtyping import Float, Integer, Bool
from typing import Tuple, Callable, Optional


# Base LSTD
# -------------------------------------------------------------------------
@struct.dataclass
class TridiagLSTDEstimatorState:
    """
    State required for V function estimation using LSTD, with
    recursive updates via the Sherman-Morrison formula.
    """

    phi: Integer[Array, "1"]
    r: Float[Array, "1"]  # reward at time t
    phitphi_next: Float[Array, "N 3"]
    sum_phi_r: Float[Array, "N"]
    sum_phi: Float[Array, "N"]
    sum_r: Float[Array, "1"]
    count: Integer[Array, "1"]

    @classmethod
    def init(cls, N, phi, r):
        """
        Initialize the LSTD estimator state.

        Args:
            d: Dimension of the feature vector
        """
        return cls(
            phi=phi,
            r=r,
            phitphi_next=jnp.zeros((N, 3)),
            sum_phi_r=jnp.zeros(N),
            sum_phi=jnp.zeros(N),
            sum_r=jnp.zeros(1),
            count=jnp.array(0, dtype=jnp.int32),
        )


def tridiag_lstd_update(
    est: TridiagLSTDEstimatorState,
    phi: Integer[Array, "1"],
    r: Float[Array, "1"],
):
    return TridiagLSTDEstimatorState(
        phi=phi,
        r=r,
        phitphi_next=est.phitphi_next.at[est.phi, phi - est.phi + 1].add(1),
        sum_phi_r=est.sum_phi_r.at[est.phi].add(est.r),
        sum_phi=est.sum_phi.at[est.phi].add(1),
        sum_r=est.sum_r + r,
        count=est.count + 1,
    )


def tridiag_lstd(est: TridiagLSTDEstimatorState, lam=1e-4, gamma=1.0):
    dl = -gamma * est.phitphi_next[:, 0]
    d = est.sum_phi - est.phitphi_next[:, 1] + lam
    du = -gamma * est.phitphi_next[:, 2]
    b = est.sum_phi_r - est.sum_phi * est.sum_r / est.count
    beta_head = jax.lax.linalg.tridiagonal_solve(
        dl=dl[:-1], d=d[:-1], du=du[:-1].at[-1].set(0), b=b[:-1].reshape(-1, 1)
    ).reshape(-1)
    beta = jnp.concatenate((beta_head, jnp.array([0.0])))
    return beta, est.sum_r / est.count


# DQ LSTD
# -------------------------------------------------------------------------
@struct.dataclass
class DQTridiagLSTDEstimatorState:
    """
    State required for doubly robust V function estimation using tridiagonal LSTD.
    """

    lstd_tr: TridiagLSTDEstimatorState
    sum_phi_tr: Float[Array, "N"]
    sum_r_tr: Float[Array, "1"]
    lstd_co: TridiagLSTDEstimatorState
    sum_phi_co: Float[Array, "N"]
    sum_r_co: Float[Array, "1"]
    prev_z: Bool[Array, "1"]
    prev_r: Float[Array, "1"]

    @classmethod
    def init(cls, N: int, phi, r, z, p):
        """
        Initialize the DQTridiagLSTD estimator state.

        Args:
            N: Dimension of the state space
        """
        return cls(
            lstd_tr=TridiagLSTDEstimatorState.init(N, phi, z / p * r),
            sum_phi_tr=jnp.zeros(N),
            sum_r_tr=jnp.zeros((1,), dtype=jnp.float32),
            lstd_co=TridiagLSTDEstimatorState.init(
                N, phi, (1 - z) / (1 - p) * r
            ),
            sum_phi_co=jnp.zeros(N),
            sum_r_co=jnp.zeros((1,), dtype=jnp.float32),
            prev_z=z,
            prev_r=r,
        )


def dq_tridiag_lstd_update(
    p: float,
    est: DQTridiagLSTDEstimatorState,
    reward: float,
    phi: Integer[Array, "1"],
    z: Bool,
) -> DQTridiagLSTDEstimatorState:
    """
    Update DQ Tridiagonal LSTD parameters

    Args:
        p: Probability of treatment
        est: Current estimator state
        reward: Observed reward
        phi: State index for current state
        z: Treatment indicator
    """
    prev_ips_tr = est.prev_z / p
    prev_r_tr = prev_ips_tr * est.prev_r
    phi_tr = jnp.zeros_like(est.sum_phi_tr).at[phi].set(prev_ips_tr)
    sum_phi_tr = est.sum_phi_tr + phi_tr
    sum_r_tr = est.sum_r_tr + prev_r_tr
    lstd_tr = tridiag_lstd_update(est.lstd_tr, phi, z / p * reward)

    prev_ips_co = (1 - est.prev_z) / (1 - p)
    prev_r_co = prev_ips_co * est.prev_r
    phi_co = jnp.zeros_like(est.sum_phi_co).at[phi].set(prev_ips_co)
    sum_phi_co = est.sum_phi_co + phi_co
    sum_r_co = est.sum_r_co + prev_r_co
    lstd_co = tridiag_lstd_update(est.lstd_co, phi, (1 - z) / (1 - p) * reward)

    return DQTridiagLSTDEstimatorState(
        lstd_tr=lstd_tr,
        sum_phi_tr=sum_phi_tr,
        sum_r_tr=sum_r_tr,
        lstd_co=lstd_co,
        sum_phi_co=sum_phi_co,
        sum_r_co=sum_r_co,
        prev_z=z,
        prev_r=reward,
    )


def dq_tridiag_lstd(
    est: DQTridiagLSTDEstimatorState, lam=1e-4, gamma=1.0
) -> Float[Array, "1"]:
    """
    Compute the doubly robust treatment effect estimate using tridiagonal LSTD.

    Args:
        est: The estimator state
        lam: Regularization parameter
        gamma: Discount factor

    Returns:
        The estimated treatment effect
    """
    beta_tr, _ = tridiag_lstd(est.lstd_tr, lam=lam, gamma=gamma)
    # V estimate is only correct up to constant, need to constrain so rho'V = 0
    bias_tr = jnp.sum(est.lstd_tr.sum_phi * beta_tr)
    Q_tr = (
        est.sum_r_tr + jnp.sum(est.sum_phi_tr * beta_tr) - bias_tr
    ) / est.lstd_tr.count

    beta_co, _ = tridiag_lstd(est.lstd_co, lam=lam, gamma=gamma)
    bias_co = jnp.sum(est.lstd_co.sum_phi * beta_co)
    Q_co = (
        est.sum_r_co + jnp.sum(est.sum_phi_co * beta_co) - bias_co
    ) / est.lstd_co.count

    return (Q_tr - Q_co)[0]  # Difference between treatment and control


# OPE Tridiagonal LSTD
# -------------------------------------------------------------------------
@struct.dataclass
class OPETridiagLSTDEstimatorState:
    """
    State required for off-policy evaluation using tridiagonal LSTD.
    """

    lstd_tr: TridiagLSTDEstimatorState
    lstd_co: TridiagLSTDEstimatorState
    prev_z: Bool[Array, "1"]
    prev_r: Float[Array, "1"]
    prev_phi: Integer[Array, "1"]

    @classmethod
    def init(cls, N: int, phi, r):
        """
        Initialize the OPETridiagLSTD estimator state.

        Args:
            N: Dimension of the state space
        """
        return cls(
            lstd_tr=TridiagLSTDEstimatorState.init(N, phi, r),
            lstd_co=TridiagLSTDEstimatorState.init(N, phi, r),
            prev_z=jnp.array(False, dtype=jnp.bool_),
            prev_r=jnp.array(0.0, dtype=jnp.float32),
            prev_phi=jnp.array(0, dtype=jnp.int32),
        )


def ope_tridiag_lstd_update(
    est: OPETridiagLSTDEstimatorState,
    reward: Float[Array, "1"],
    phi: Integer[Array, "1"],
    z: Bool,
) -> OPETridiagLSTDEstimatorState:
    """
    Update OPE Tridiagonal LSTD parameters

    Args:
        est: Current estimator state
        reward: Observed reward
        phi: State index for current state
        z: Treatment indicator
    """
    new_lstd_tr = tridiag_lstd_update(
        est.lstd_tr.replace(phi=est.prev_phi, r=est.prev_r), phi, reward
    )

    new_lstd_co = tridiag_lstd_update(
        est.lstd_co.replace(phi=est.prev_phi, r=est.prev_r), phi, reward
    )

    return OPETridiagLSTDEstimatorState(
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


def ope_tridiag_lstd(est: OPETridiagLSTDEstimatorState, lam=0.0):
    """
    Compute the off-policy evaluation estimate using tridiagonal LSTD.

    Args:
        est: The estimator state
        lam: Regularization parameter

    Returns:
        The estimated treatment effect
    """
    # For treatment group
    tr_probs = (est.lstd_tr.phitphi_next + lam) / (
        est.lstd_tr.sum_phi.reshape(-1, 1) + lam
    )
    tr_log_wts = jnp.log(tr_probs[:, 2][:-1]) - jnp.log(tr_probs[:, 0][1:])
    tr_unnorm_rho = jnp.exp(
        jnp.concatenate((jnp.zeros(1), jnp.cumsum(tr_log_wts)))
    )
    tr_rho = tr_unnorm_rho / jnp.sum(tr_unnorm_rho)
    tr_r = est.lstd_tr.sum_phi_r / jnp.maximum(est.lstd_tr.sum_phi, 1.0)
    tr_value = tr_rho @ tr_r

    # For control group
    co_probs = (est.lstd_co.phitphi_next + lam) / (
        est.lstd_co.sum_phi.reshape(-1, 1) + lam
    )
    co_log_wts = jnp.log(co_probs[:, 2][:-1]) - jnp.log(co_probs[:, 0][1:])
    co_unnorm_rho = jnp.exp(
        jnp.concatenate((jnp.zeros(1), jnp.cumsum(co_log_wts)))
    )
    co_rho = co_unnorm_rho / jnp.sum(co_unnorm_rho)
    co_r = est.lstd_co.sum_phi_r / jnp.maximum(est.lstd_co.sum_phi, 1.0)
    co_value = co_rho @ co_r

    return tr_value - co_value
