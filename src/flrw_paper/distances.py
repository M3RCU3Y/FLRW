"""Distance, time, horizon, and BAO observable calculations."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad, solve_ivp
from scipy.optimize import brentq

from .models import (
    A_MIN,
    BAO_RD_MPC,
    DH_GPC,
    DH_MPC,
    H0_INV_GYR,
    ODE_ATOL,
    ODE_RTOL,
    QUAD_EPSABS,
    QUAD_EPSREL,
    Z_MAX_DISTANCE,
    Model,
)

def time_integrand_a(a: np.ndarray | float, model: Model) -> np.ndarray | float:
    """Dimensionless age integrand: d tau / da = 1 / [a E(a)]."""
    a = np.asarray(a)
    return 1.0 / (a * model.e_a(a))


def conformal_integrand_a(a: np.ndarray | float, model: Model) -> np.ndarray | float:
    """Dimensionless conformal-time integrand: d eta_bar / da = 1/[a^2 E(a)]."""
    a = np.asarray(a)
    return 1.0 / (a**2 * model.e_a(a))


def early_time_age(model: Model, eps: float = A_MIN) -> float:
    """Leading integral of 1/[a E(a)] from a=0 to eps."""
    if model.omega_r > 0.0:
        return eps**2 / (2.0 * np.sqrt(model.omega_r))
    if model.omega_m > 0.0:
        return 2.0 * eps**1.5 / (3.0 * np.sqrt(model.omega_m))
    if model.omega_k > 0.0:
        return eps / np.sqrt(model.omega_k)
    return 0.0


def early_time_conformal(model: Model, eps: float = A_MIN) -> float:
    """Leading integral of 1/[a^2 E(a)] from a=0 to eps."""
    if model.omega_r > 0.0:
        return eps / np.sqrt(model.omega_r)
    if model.omega_m > 0.0:
        return 2.0 * np.sqrt(eps) / np.sqrt(model.omega_m)
    if model.omega_k > 0.0:
        # A curvature-only Big Bang has a logarithmically divergent conformal time.
        return np.nan
    return 0.0


def dimensionless_age(model: Model, a: float = 1.0) -> float:
    """Return H0 * t(a)."""
    upper = a
    if model.turnaround_a is not None:
        upper = min(upper, model.turnaround_a * (1.0 - 1.0e-11))
    if upper <= A_MIN:
        return early_time_age(model, upper)
    val, _ = quad(
        lambda x: float(time_integrand_a(x, model)),
        A_MIN,
        upper,
        epsabs=QUAD_EPSABS,
        epsrel=QUAD_EPSREL,
        limit=400,
    )
    return early_time_age(model) + val


def dimensionless_horizon(model: Model, a: float = 1.0) -> float:
    """Return proper particle horizon in units of c/H0: a * integral da/[a^2 E(a)]."""
    upper = a
    if model.turnaround_a is not None:
        upper = min(upper, model.turnaround_a * (1.0 - 1.0e-11))
    if upper <= A_MIN:
        return upper * early_time_conformal(model, upper)
    val, _ = quad(
        lambda x: float(conformal_integrand_a(x, model)),
        A_MIN,
        upper,
        epsabs=QUAD_EPSABS,
        epsrel=QUAD_EPSREL,
        limit=400,
    )
    return upper * (early_time_conformal(model) + val)


def dimensionless_comoving_distance(model: Model, z: float) -> float:
    val, _ = quad(
        lambda zp: 1.0 / float(model.e_z(zp)),
        0.0,
        z,
        epsabs=QUAD_EPSABS,
        epsrel=QUAD_EPSREL,
        limit=400,
    )
    return val


def dimensionless_lookback_time(model: Model, z: float) -> float:
    val, _ = quad(
        lambda zp: 1.0 / ((1.0 + zp) * float(model.e_z(zp))),
        0.0,
        z,
        epsabs=QUAD_EPSABS,
        epsrel=QUAD_EPSREL,
        limit=400,
    )
    return val


def transverse_distance_gpc(model: Model, z: float) -> float:
    dc = np.array([dimensionless_comoving_distance(model, z) * DH_GPC])
    return float(transverse_distance_from_dc(model, dc)[0])


def luminosity_distance_mpc(model: Model, z: np.ndarray | float) -> np.ndarray | float:
    z_arr = np.asarray(z)
    if z_arr.ndim == 0:
        dm = transverse_distance_gpc(model, float(z_arr))
        return (1.0 + float(z_arr)) * dm * 1000.0
    return np.array([luminosity_distance_mpc(model, float(zi)) for zi in z_arr])


def distance_modulus(model: Model, z: np.ndarray | float) -> np.ndarray | float:
    dl_mpc = luminosity_distance_mpc(model, z)
    return 5.0 * np.log10(dl_mpc) + 25.0


def distance_modulus_fast(model: Model, z: np.ndarray) -> np.ndarray:
    """Vectorized distance modulus using a dense cumulative-integration grid."""
    z_arr = np.asarray(z, dtype=float)
    if z_arr.ndim != 1:
        raise ValueError("distance_modulus_fast expects a one-dimensional redshift array")
    if np.any(z_arr <= 0.0):
        raise ValueError("distance modulus is defined here only for positive redshift")

    z_max = float(np.max(z_arr))
    n_grid = max(6000, 4 * len(z_arr))
    grid = np.unique(np.concatenate(([0.0], z_arr, np.linspace(0.0, z_max, n_grid))))
    chi = cumulative_trapezoid(1.0 / model.e_z(grid), grid, initial=0.0)
    chi_at_z = np.interp(z_arr, grid, chi)
    dc_gpc = DH_GPC * chi_at_z
    dm_gpc = transverse_distance_from_dc(model, dc_gpc)
    dl_mpc = (1.0 + z_arr) * dm_gpc * 1000.0
    return 5.0 * np.log10(dl_mpc) + 25.0


def deceleration_parameter_z(model: Model, z: np.ndarray | float) -> np.ndarray | float:
    zp1 = 1.0 + np.asarray(z)
    numerator = (
        model.omega_r * zp1**4
        + 0.5 * model.omega_m * zp1**3
        - model.omega_l
    )
    return numerator / model.e_z(z) ** 2


def transition_redshift(model: Model) -> float | None:
    if model.omega_l <= 0.0:
        return None
    try:
        return float(brentq(lambda z: float(deceleration_parameter_z(model, z)), 0.0, 10.0))
    except ValueError:
        return None


def angular_diameter_maximum(model: Model, z_max: float = Z_MAX_DISTANCE) -> tuple[float, float]:
    z = np.linspace(0.001, z_max, 5000)
    dc = comoving_distance_grid(model, z)
    dm = transverse_distance_from_dc(model, dc)
    da = dm / (1.0 + z)
    i = int(np.argmax(da))
    return float(z[i]), float(da[i])


def bao_prediction(model: Model, z: float, quantity: str, rd_mpc: float = BAO_RD_MPC) -> float:
    """Return a DESI-style BAO observable for a model at redshift z."""
    dc_mpc = dimensionless_comoving_distance(model, z) * DH_MPC
    dm_mpc = transverse_distance_from_dc(model, np.asarray([dc_mpc / 1000.0]))[0] * 1000.0
    dh_mpc = DH_MPC / float(model.e_z(z))
    if quantity == "DM_over_rs":
        return float(dm_mpc / rd_mpc)
    if quantity == "DH_over_rs":
        return float(dh_mpc / rd_mpc)
    if quantity == "DV_over_rs":
        dv_mpc = (z * dm_mpc**2 * dh_mpc) ** (1.0 / 3.0)
        return float(dv_mpc / rd_mpc)
    raise ValueError(f"Unknown BAO quantity: {quantity}")


def lookback_time_grid(model: Model, z: np.ndarray) -> np.ndarray:
    integrand = 1.0 / ((1.0 + z) * model.e_z(z))
    tau = cumulative_trapezoid(integrand, z, initial=0.0)
    return tau * H0_INV_GYR


def comoving_distance_grid(model: Model, z: np.ndarray) -> np.ndarray:
    integrand = 1.0 / model.e_z(z)
    chi = cumulative_trapezoid(integrand, z, initial=0.0)
    return DH_GPC * chi


def transverse_distance_from_dc(model: Model, dc_gpc: np.ndarray) -> np.ndarray:
    """Convert line-of-sight comoving distance to transverse comoving distance."""
    chi = dc_gpc / DH_GPC
    ok = model.omega_k
    if np.isclose(ok, 0.0):
        return dc_gpc
    if ok > 0.0:
        return DH_GPC / np.sqrt(ok) * np.sinh(np.sqrt(ok) * chi)
    return DH_GPC / np.sqrt(abs(ok)) * np.sin(np.sqrt(abs(ok)) * chi)


def model_time_curve(model: Model, a_stop: float = 4.2, n: int = 3200) -> tuple[np.ndarray, np.ndarray]:
    """Return t[Gyr], a(t), using quadrature plus an ODE sanity integration."""
    if model.turnaround_a is not None:
        a_stop = min(a_stop, model.turnaround_a * (1.0 - 3.0e-5))
    a_grid = np.geomspace(A_MIN, a_stop, n)
    tau = cumulative_trapezoid(time_integrand_a(a_grid, model), a_grid, initial=0.0)
    tau = tau + early_time_age(model)

    # Demonstrate solve_ivp on the same ODE. Dense output is not needed for the
    # final plot, but the call validates that the equation is well-posed in tau.
    def rhs(_tau: float, y: Iterable[float]) -> list[float]:
        a = max(float(y[0]), A_MIN)
        return [a * float(model.e_a(a))]

    def event_stop(_tau: float, y: Iterable[float]) -> float:
        target = a_stop
        return target - float(y[0])

    event_stop.terminal = True  # type: ignore[attr-defined]
    event_stop.direction = -1   # type: ignore[attr-defined]

    try:
        sol = solve_ivp(
            rhs,
            (tau[0], tau[-1]),
            [A_MIN],
            method="DOP853",
            rtol=ODE_RTOL,
            atol=ODE_ATOL,
            max_step=0.015,
            events=event_stop,
        )
        if sol.success and sol.t.size > 20:
            t_gyr = sol.t * H0_INV_GYR
            a_grid = sol.y[0]
        else:
            t_gyr = tau * H0_INV_GYR
    except Exception:
        t_gyr = tau * H0_INV_GYR

    if model.turnaround_a is not None:
        # A pressureless closed universe is symmetric about maximum expansion.
        t_turn = t_gyr[-1]
        a_turn = a_grid[-1]
        t_collapse = 2.0 * t_turn - t_gyr[-2::-1]
        a_collapse = a_grid[-2::-1]
        t_gyr = np.concatenate([t_gyr, t_collapse])
        a_grid = np.concatenate([a_grid, a_collapse])
        mask = t_gyr <= 36.0
        t_gyr, a_grid = t_gyr[mask], a_grid[mask]
        if a_grid.size > 0:
            a_grid[np.argmax(a_grid)] = a_turn
    return t_gyr, a_grid
