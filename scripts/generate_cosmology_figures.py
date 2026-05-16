#!/usr/bin/env python3
"""
Generate all numerical figures and LaTeX tables for
"Numerical Experiments in Cosmological Expansion".

The script is intentionally explicit rather than compressed. It uses SI units
internally, reports cosmological distances in Gpc, reports times in Gyr, and
saves publication-ready vector PDF graphics for inclusion in LaTeX.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.integrate import cumulative_trapezoid, quad, solve_ivp
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize_scalar

# -----------------------------------------------------------------------------
# Constants and unit conversions
# -----------------------------------------------------------------------------
C = 299_792_458.0                         # speed of light [m s^-1]
MPC = 3.0856775814913673e22               # one megaparsec [m]
GPC = 1.0e3 * MPC                         # one gigaparsec [m]
JULIAN_YEAR = 365.25 * 24 * 3600.0        # one Julian year [s]
GYR = 1.0e9 * JULIAN_YEAR                 # one gigayear [s]
H0_KM_S_MPC = 70.0                        # reference Hubble constant [km s^-1 Mpc^-1]
H0 = H0_KM_S_MPC * 1000.0 / MPC           # reference Hubble constant [s^-1]
H0_INV_GYR = 1.0 / H0 / GYR               # Hubble time [Gyr]
DH_GPC = C / H0 / GPC                     # Hubble distance [Gpc]
DH_MPC = C / H0 / MPC                     # Hubble distance [Mpc]

# Baseline concordance-like model. The small radiation term is included here and
# in the era/timeline figures so that the manuscript has one consistent baseline.
BASELINE_OMEGA_R = 9.0e-5
BASELINE_OMEGA_M = 0.3
BASELINE_OMEGA_L = 1.0 - BASELINE_OMEGA_M - BASELINE_OMEGA_R

# Numerical controls. A_MIN is used only to regularize plotting grids near the
# Big Bang; quadrature functions add the leading asymptotic contribution below it.
A_MIN = 1.0e-7
Z_MAX_DISTANCE = 6.0
Z_MAX_HUBBLE = 50.0
PANTHEON_Z_MIN = 0.01
PANTHEON_Z_MAX = 1.40
ODE_RTOL = 1.0e-10
ODE_ATOL = 1.0e-12
QUAD_EPSABS = 1.0e-11
QUAD_EPSREL = 1.0e-11

# A restrained academic palette. Colors are specified here so the LaTeX document
# can remain visually consistent across all generated figures.
PALETTE = {
    "EdS": "#24476b",
    "Radiation": "#6a4c93",
    "Flat_LCDM": "#cc5a1a",
    "Open_matter": "#2a8f86",
    "Closed_matter": "#7d2d2d",
    "Grid": "#aebfd0",
    "Ink": "#1d2733",
}

LINE_STYLES = {
    "EdS": "-",
    "Radiation": "--",
    "Flat_LCDM": "-",
    "Open_matter": "-.",
    "Closed_matter": ":",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10.8,
    "axes.titlesize": 12.8,
    "axes.labelsize": 11.4,
    "axes.titleweight": "semibold",
    "legend.fontsize": 8.8,
    "xtick.labelsize": 9.6,
    "ytick.labelsize": 9.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.24,
    "grid.linewidth": 0.55,
    "figure.dpi": 160,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


@dataclass(frozen=True)
class Model:
    """A minimal FLRW model with constant Omega_Lambda and optional curvature."""

    name: str
    label: str
    omega_r: float = 0.0
    omega_m: float = 0.0
    omega_l: float = 0.0
    omega_k: float = 0.0
    color: str = "#333333"

    def e2_a(self, a: np.ndarray | float) -> np.ndarray | float:
        a = np.asarray(a)
        return (
            self.omega_r * a ** -4
            + self.omega_m * a ** -3
            + self.omega_k * a ** -2
            + self.omega_l
        )

    def e_a(self, a: np.ndarray | float) -> np.ndarray | float:
        return np.sqrt(np.maximum(self.e2_a(a), 0.0))

    def e_z(self, z: np.ndarray | float) -> np.ndarray | float:
        zp1 = 1.0 + np.asarray(z)
        e2 = (
            self.omega_r * zp1**4
            + self.omega_m * zp1**3
            + self.omega_k * zp1**2
            + self.omega_l
        )
        return np.sqrt(np.maximum(e2, 0.0))

    @property
    def has_turnaround(self) -> bool:
        # For a dust-only closed model, E(a)^2 = Omega_m a^-3 + Omega_k a^-2.
        return self.omega_k < 0.0 and self.omega_l == 0.0 and self.omega_r == 0.0

    @property
    def turnaround_a(self) -> float | None:
        if self.has_turnaround:
            return -self.omega_m / self.omega_k
        return None


MODELS: list[Model] = [
    Model("EdS", r"Einstein--de Sitter", omega_m=1.0, color=PALETTE["EdS"]),
    Model("Radiation", r"Radiation dominated", omega_r=1.0, color=PALETTE["Radiation"]),
    Model(
        "Flat_LCDM",
        r"Flat $\Lambda$CDM-like",
        omega_r=BASELINE_OMEGA_R,
        omega_m=BASELINE_OMEGA_M,
        omega_l=BASELINE_OMEGA_L,
        color=PALETTE["Flat_LCDM"],
    ),
    Model("Open_matter", r"Open matter dominated", omega_m=0.3, omega_k=0.7, color=PALETTE["Open_matter"]),
    Model("Closed_matter", r"Closed matter dominated", omega_m=1.3, omega_k=-0.3, color=PALETTE["Closed_matter"]),
]

MODEL_BY_NAME = {m.name: m for m in MODELS}

ASTROPY_CHECK_MODEL = Model(
    "Flat_LCDM_no_radiation",
    r"Flat $\Lambda$CDM, radiation omitted",
    omega_m=BASELINE_OMEGA_M,
    omega_l=1.0 - BASELINE_OMEGA_M,
    color=PALETTE["Flat_LCDM"],
)


def model_line_style(model: Model) -> str:
    """Return the model line style used for color-independent plot reading."""
    return LINE_STYLES.get(model.name, "-")


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


def load_pantheon_binned(root: Path) -> dict[str, np.ndarray]:
    path = root / "data" / "pantheon_plus_binned.csv"
    records: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(row for row in handle if not row.startswith("#"))
        for row in reader:
            records.append({
                "z_mean": float(row["z_mean"]),
                "mu_mean": float(row["mu_mean"]),
                "mu_err": float(row["mu_err"]),
                "n_supernovae": float(row["n_supernovae"]),
            })
    return {key: np.array([record[key] for record in records]) for key in records[0]}


def flat_lcdm_shape_model(omega_m: float, omega_r: float = BASELINE_OMEGA_R) -> Model:
    """Flat radiation-inclusive LambdaCDM model for supernova shape fitting."""
    omega_l = 1.0 - omega_m - omega_r
    if omega_m <= 0.0 or omega_l <= 0.0:
        raise ValueError("flat_lcdm_shape_model requires positive matter and Lambda densities")
    return Model(
        "PantheonFlatLCDM",
        rf"flat $\Lambda$CDM, $\Omega_m={omega_m:.3f}$",
        omega_r=omega_r,
        omega_m=omega_m,
        omega_l=omega_l,
        omega_k=0.0,
        color=PALETTE["Flat_LCDM"],
    )


@lru_cache(maxsize=2)
def load_pantheon_full_likelihood(root_str: str) -> dict[str, object]:
    """Load Pantheon+SH0ES distances and the matching STAT+SYS covariance."""
    root = Path(root_str)
    data_path = root / "data" / "Pantheon+SH0ES.dat"
    cov_path = root / "data" / "Pantheon+SH0ES_STAT+SYS.cov"

    table = np.genfromtxt(data_path, names=True, dtype=None, encoding=None)
    with cov_path.open("r", encoding="utf-8") as handle:
        n_cov = int(handle.readline().strip())
    cov_flat = np.loadtxt(cov_path, skiprows=1)
    cov = cov_flat.reshape((n_cov, n_cov))

    if len(table) != n_cov:
        raise ValueError(f"Pantheon table length {len(table)} does not match covariance size {n_cov}")

    mask = (
        (table["IS_CALIBRATOR"].astype(int) == 0)
        & (table["zHD"].astype(float) > PANTHEON_Z_MIN)
        & (table["zHD"].astype(float) < PANTHEON_Z_MAX)
    )
    idx = np.where(mask)[0]
    z = table["zHD"][mask].astype(float)
    mu = table["MU_SH0ES"][mask].astype(float)
    cov_subset = cov[np.ix_(idx, idx)]

    cfac = cho_factor(cov_subset, lower=True, check_finite=False)
    ones = np.ones_like(z)
    c_inv_ones = cho_solve(cfac, ones, check_finite=False)
    denom = float(ones @ c_inv_ones)

    return {
        "z": z,
        "mu": mu,
        "cov": cov_subset,
        "cfac": cfac,
        "ones": ones,
        "c_inv_ones": c_inv_ones,
        "denom": denom,
        "n_total": int(len(table)),
        "n_used": int(len(z)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
    }


def pantheon_full_fit_chi2(root: Path, omega_m: float) -> tuple[float, float, np.ndarray]:
    """Return chi-square, best additive offset, and residuals for Omega_m."""
    data = load_pantheon_full_likelihood(str(root.resolve()))
    z = np.asarray(data["z"], dtype=float)
    mu = np.asarray(data["mu"], dtype=float)
    ones = np.asarray(data["ones"], dtype=float)
    denom = float(data["denom"])
    cfac = data["cfac"]

    model = flat_lcdm_shape_model(omega_m)
    delta = mu - distance_modulus_fast(model, z)
    c_inv_delta = cho_solve(cfac, delta, check_finite=False)
    offset = float((ones @ c_inv_delta) / denom)
    chi2 = float(delta @ c_inv_delta - (ones @ c_inv_delta) ** 2 / denom)
    residual = delta - offset
    return chi2, offset, residual


@lru_cache(maxsize=2)
def _pantheon_full_fit_summary_cached(root_str: str) -> dict[str, object]:
    root = Path(root_str)

    def objective(omega_m: float) -> float:
        return pantheon_full_fit_chi2(root, float(omega_m))[0]

    result = minimize_scalar(
        objective,
        bounds=(0.05, 0.60),
        method="bounded",
        options={"xatol": 1.0e-5},
    )
    if not result.success:
        raise RuntimeError(f"Pantheon likelihood minimization failed: {result.message}")

    omega_best = float(result.x)
    chi2_min, offset, residual = pantheon_full_fit_chi2(root, omega_best)
    left = float(brentq(lambda om: objective(om) - chi2_min - 1.0, 0.05, omega_best, xtol=1.0e-4))
    right = float(brentq(lambda om: objective(om) - chi2_min - 1.0, omega_best, 0.60, xtol=1.0e-4))

    grid = np.linspace(max(0.08, omega_best - 0.12), min(0.60, omega_best + 0.12), 121)
    chi2_grid = np.array([objective(float(omega_m)) for omega_m in grid])
    chi2_baseline, offset_baseline, _ = pantheon_full_fit_chi2(root, BASELINE_OMEGA_M)
    data = load_pantheon_full_likelihood(root_str)
    dof = int(data["n_used"]) - 2

    return {
        "omega_m_best": omega_best,
        "omega_m_left_1sigma": left,
        "omega_m_right_1sigma": right,
        "omega_m_err_minus": omega_best - left,
        "omega_m_err_plus": right - omega_best,
        "chi2_min": chi2_min,
        "offset": offset,
        "residual_rms": float(np.sqrt(np.mean(np.asarray(residual) ** 2))),
        "dof": dof,
        "reduced_chi2": chi2_min / dof,
        "omega_m_grid": grid,
        "delta_chi2_grid": chi2_grid - chi2_min,
        "chi2_baseline": chi2_baseline,
        "offset_baseline": offset_baseline,
        "delta_chi2_baseline": chi2_baseline - chi2_min,
        "n_total": int(data["n_total"]),
        "n_used": int(data["n_used"]),
        "z_min": float(data["z_min"]),
        "z_max": float(data["z_max"]),
        "cov_dim": int(data["n_used"]),
    }


def pantheon_full_fit_summary(root: Path) -> dict[str, object]:
    return _pantheon_full_fit_summary_cached(str(root.resolve()))


def pantheon_residual_summary(root: Path, model: Model) -> dict[str, float | np.ndarray]:
    data = load_pantheon_binned(root)
    z = data["z_mean"]
    mu_obs = data["mu_mean"]
    mu_err = data["mu_err"]
    mu_model = np.asarray(distance_modulus(model, z))
    weights = 1.0 / mu_err**2
    offset = float(np.average(mu_obs - mu_model, weights=weights))
    residual = mu_obs - (mu_model + offset)
    chi2 = float(np.sum((residual / mu_err) ** 2))
    dof = max(len(z) - 1, 1)
    weighted_rms = float(np.sqrt(np.average(residual**2, weights=weights)))
    return {
        "z": z,
        "mu_obs": mu_obs,
        "mu_err": mu_err,
        "mu_model": mu_model,
        "offset": offset,
        "residual": residual,
        "chi2": chi2,
        "reduced_chi2": chi2 / dof,
        "weighted_rms": weighted_rms,
        "n_bins": float(len(z)),
        "n_supernovae": float(np.sum(data["n_supernovae"])),
    }


def fate_text(model: Model) -> str:
    if model.name == "Flat_LCDM":
        return "accelerated expansion"
    if model.name == "Closed_matter":
        return "finite turnaround and recollapse"
    if model.name == "Open_matter":
        return "eternal expansion; curvature-coasting limit"
    if model.name == "Radiation":
        return "eternal decelerating expansion"
    return "eternal decelerating expansion"


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


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def finish_axis(ax: plt.Axes, legend: bool = True) -> None:
    ax.tick_params(direction="out", length=4, width=0.8)
    if legend:
        leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.75")
        if leg is not None:
            leg.get_frame().set_linewidth(0.6)
            leg.get_frame().set_alpha(0.95)


def fig_flrw_diagram(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("FLRW expansion: fixed comoving coordinates, growing physical scale", pad=14)

    centers = [(2.0, 3.1), (5.0, 3.1), (8.0, 3.1)]
    scales = [0.72, 1.06, 1.48]
    labels = [r"early: $a(t)<1$", r"today: $a_0=1$", r"later: $a(t)>1$"]
    grid_color = PALETTE["Grid"]

    for (cx, cy), s, label in zip(centers, scales, labels):
        for u in np.linspace(-1, 1, 5):
            ax.plot([cx - s, cx + s], [cy + u*s, cy + u*s], color=grid_color, lw=0.75, alpha=0.72)
            ax.plot([cx + u*s, cx + u*s], [cy - s, cy + s], color=grid_color, lw=0.75, alpha=0.72)
        ax.add_patch(Circle((cx, cy), s, ec=PALETTE["EdS"], fc="#f3f7fb", lw=1.25, alpha=0.96))
        for dx, dy in [(-0.42, -0.12), (0.32, 0.26), (-0.04, 0.48), (0.42, -0.42)]:
            ax.add_patch(Circle((cx + dx*s, cy + dy*s), 0.050, ec="none", fc=PALETTE["Flat_LCDM"]))
        ax.text(cx, 1.10, label, ha="center", va="center", fontsize=10.6)

    for x0, x1 in [(3.0, 4.05), (6.20, 6.95)]:
        arrow = FancyArrowPatch((x0, 3.1), (x1, 3.1), arrowstyle="-|>", mutation_scale=16,
                                lw=1.15, color=PALETTE["Ink"])
        ax.add_patch(arrow)
    ax.text(5.0, 5.34, r"physical separations obey $D_{\rm phys}(t)=a(t)\,D_{\rm com}$",
            ha="center", fontsize=11.1, color=PALETTE["Ink"])
    ax.text(5.0, 0.45, r"photon wavelengths stretch with the same factor: $1+z=a_0/a(t_{\rm em})$",
            ha="center", fontsize=10.9)
    savefig(out / "flrw_expansion_diagram.pdf")


def fig_scale_factor(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.65))
    for model in MODELS:
        t, a = model_time_curve(model)
        mask = t <= 34.0
        ax.plot(t[mask], a[mask], lw=2.0, ls=model_line_style(model), label=model.label, color=model.color)
    ax.axhline(1.0, ls="--", lw=0.95, color="0.35", alpha=0.74)
    ax.text(33.6, 1.04, r"$a=1$", ha="right", va="bottom", fontsize=9.4)
    ax.set_xlim(0, 34)
    ax.set_ylim(0, 4.7)
    ax.set_xlabel("cosmic time since Big Bang [Gyr]")
    ax.set_ylabel(r"scale factor $a(t)$")
    ax.set_title(r"Expansion histories from $\mathrm{d}a/\mathrm{d}t=aH_0E(a)$")
    finish_axis(ax)
    savefig(out / "scale_factor_evolution.pdf")


def fig_analytic_validation(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.65))
    tau = np.geomspace(1.0e-4, 1.3, 500)
    a_eds = (1.5 * tau) ** (2.0 / 3.0)
    a_rad = np.sqrt(2.0 * tau)
    ax.loglog(tau, a_eds, color=PALETTE["EdS"], lw=2.0, label=r"EdS analytic: $a=(3\tau/2)^{2/3}$")
    ax.loglog(tau, a_rad, color=PALETTE["Radiation"], lw=2.0, label=r"radiation analytic: $a=(2\tau)^{1/2}$")
    for model in [MODEL_BY_NAME["EdS"], MODEL_BY_NAME["Radiation"]]:
        t_gyr, a = model_time_curve(model, a_stop=2.0, n=2200)
        tau_num = t_gyr / H0_INV_GYR
        mask = (tau_num >= tau.min()) & (tau_num <= tau.max())
        ax.loglog(tau_num[mask], a[mask], ls="--", lw=1.2, color=model.color,
                  label=rf"{model.label} numerical")
    ax.set_xlabel(r"dimensionless time $\tau=H_0t$")
    ax.set_ylabel(r"scale factor $a(\tau)$")
    ax.set_title("Analytic scale-factor benchmarks for two exactly soluble limits")
    finish_axis(ax)
    savefig(out / "analytic_scale_factor_validation.pdf")


def fig_hubble_parameter(out: Path) -> None:
    z = np.geomspace(1.0e-3, Z_MAX_HUBBLE, 1000)
    fig, ax = plt.subplots(figsize=(7.3, 4.55))
    for model in MODELS:
        ax.loglog(z, model.e_z(z), lw=2.0, ls=model_line_style(model), label=model.label, color=model.color)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"$H(z)/H_0 = E(z)$")
    ax.set_title("Expansion rate as a function of redshift")
    ax.grid(True, which="both", alpha=0.23)
    finish_axis(ax)
    savefig(out / "hubble_parameter.pdf")


def fig_density_fractions(out: Path) -> None:
    omega_r = BASELINE_OMEGA_R
    omega_m = BASELINE_OMEGA_M
    omega_l = BASELINE_OMEGA_L
    a = np.geomspace(1.0e-5, 2.0, 1200)
    e2 = omega_r * a**-4 + omega_m * a**-3 + omega_l
    frac_r = omega_r * a**-4 / e2
    frac_m = omega_m * a**-3 / e2
    frac_l = omega_l / e2
    z_eq = omega_m / omega_r - 1.0
    a_eq = 1.0 / (1.0 + z_eq)
    a_lam = (omega_m / omega_l) ** (1.0 / 3.0)

    fig, ax = plt.subplots(figsize=(7.3, 4.55))
    ax.semilogx(a, frac_r, lw=2.0, color=PALETTE["Radiation"], label=r"$\Omega_r(a)$")
    ax.semilogx(a, frac_m, lw=2.0, color=PALETTE["EdS"], label=r"$\Omega_m(a)$")
    ax.semilogx(a, frac_l, lw=2.0, color=PALETTE["Flat_LCDM"], label=r"$\Omega_\Lambda(a)$")
    ax.axvline(a_eq, color="0.45", ls=":", lw=0.95)
    ax.axvline(a_lam, color="0.45", ls=":", lw=0.95)
    ax.text(a_eq * 1.12, 0.08, r"matter-radiation equality", rotation=90, va="bottom", fontsize=8.8)
    ax.text(a_lam * 1.08, 0.08, r"matter-$\Lambda$ equality", rotation=90, va="bottom", fontsize=8.8)
    ax.set_xlabel(r"scale factor $a$")
    ax.set_ylabel(r"fractional contribution to $E(a)^2$")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(r"Energy-component dominance in a standard-like flat model")
    finish_axis(ax)
    savefig(out / "density_fractions_lcdm.pdf")


def fig_lookback_time(out: Path) -> None:
    z = np.linspace(0, Z_MAX_DISTANCE, 2200)
    fig, ax = plt.subplots(figsize=(7.3, 4.55))
    for model in MODELS:
        ax.plot(z, lookback_time_grid(model, z), lw=2.0, ls=model_line_style(model), label=model.label, color=model.color)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel("lookback time [Gyr]")
    ax.set_title("Redshift as a clock: lookback time in each model")
    finish_axis(ax)
    savefig(out / "lookback_time.pdf")


def fig_comoving_distance(out: Path) -> None:
    z = np.linspace(0, Z_MAX_DISTANCE, 2200)
    fig, ax = plt.subplots(figsize=(7.3, 4.55))
    for model in MODELS:
        ax.plot(z, comoving_distance_grid(model, z), lw=2.0, ls=model_line_style(model), label=model.label, color=model.color)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"line-of-sight comoving distance $D_C$ [Gpc]")
    ax.set_title("Comoving distance-redshift relation")
    finish_axis(ax)
    savefig(out / "comoving_distance.pdf")


def fig_luminosity_distance(out: Path) -> None:
    z = np.linspace(0, Z_MAX_DISTANCE, 2200)
    fig, ax = plt.subplots(figsize=(7.3, 4.55))
    for model in MODELS:
        dc = comoving_distance_grid(model, z)
        dm = transverse_distance_from_dc(model, dc)
        dl = (1.0 + z) * dm
        ax.plot(z, dl, lw=2.0, ls=model_line_style(model), label=model.label, color=model.color)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"luminosity distance $D_L$ [Gpc]")
    ax.set_title("Flux distances grow rapidly with redshift")
    finish_axis(ax)
    savefig(out / "luminosity_distance.pdf")


def fig_angular_distance(out: Path) -> None:
    z = np.linspace(0, Z_MAX_DISTANCE, 2200)
    fig, ax = plt.subplots(figsize=(7.3, 4.55))
    for model in MODELS:
        dc = comoving_distance_grid(model, z)
        dm = transverse_distance_from_dc(model, dc)
        da = dm / (1.0 + z)
        ax.plot(z, da, lw=2.0, ls=model_line_style(model), label=model.label, color=model.color)
        if model.name == "Flat_LCDM":
            i = int(np.argmax(da))
            ax.scatter([z[i]], [da[i]], s=34, color=model.color, zorder=4)
            ax.annotate(r"maximum $D_A$", (z[i], da[i]), xytext=(z[i] + 0.55, da[i] - 0.08),
                        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": model.color},
                        fontsize=9.2, color=model.color)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"angular diameter distance $D_A$ [Gpc]")
    ax.set_title("Angular diameter distance is not monotonic")
    finish_axis(ax)
    savefig(out / "angular_diameter_distance.pdf")


def fig_particle_horizon(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.55))
    for model in MODELS:
        a_stop = 3.8
        if model.turnaround_a is not None:
            a_stop = min(a_stop, model.turnaround_a * (1.0 - 3.0e-5))
        a = np.geomspace(A_MIN, a_stop, 3000)
        tau = cumulative_trapezoid(time_integrand_a(a, model), a, initial=0.0) + early_time_age(model)
        eta = cumulative_trapezoid(conformal_integrand_a(a, model), a, initial=0.0) + early_time_conformal(model)
        t_gyr = tau * H0_INV_GYR
        horizon_gpc = a * eta * DH_GPC
        mask = t_gyr <= 34.0
        ax.plot(t_gyr[mask], horizon_gpc[mask], lw=2.0, ls=model_line_style(model), label=model.label, color=model.color)
    ax.set_xlabel("cosmic time since Big Bang [Gyr]")
    ax.set_ylabel(r"proper particle horizon $D_{\rm hor}$ [Gpc]")
    ax.set_title("Growth of the causally connected region")
    finish_axis(ax)
    savefig(out / "particle_horizon.pdf")


def fig_cosmic_timeline(out: Path) -> None:
    omega_r = BASELINE_OMEGA_R
    omega_m = BASELINE_OMEGA_M
    omega_l = BASELINE_OMEGA_L
    z_eq = omega_m / omega_r - 1.0
    z_l = (omega_l / omega_m) ** (1.0 / 3.0) - 1.0
    x_eq = np.log10(1.0 / (1.0 + z_eq))
    x_l = np.log10(1.0 / (1.0 + z_l))

    fig, ax = plt.subplots(figsize=(7.55, 3.15))
    ax.set_xlim(-4.35, 0.55)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Schematic cosmic history by dominant energy component", pad=10)

    bands = [
        (-4.0, x_eq, "", PALETTE["Radiation"]),
        (x_eq, x_l, "matter era", PALETTE["EdS"]),
        (x_l, 0.0, "", PALETTE["Flat_LCDM"]),
    ]
    for x0, x1, label, color in bands:
        ax.add_patch(Rectangle((x0, 0.36), x1 - x0, 0.28, fc=color, ec="none", alpha=0.82))
        if label:
            ax.text((x0 + x1) / 2, 0.50, label, ha="center", va="center", color="white", fontsize=10.2)
    ax.text(-4.0, 0.77, "radiation era", ha="left", va="center", color=PALETTE["Radiation"], fontsize=9.2)
    ax.text(0.04, 0.61, r"$\Lambda$ domination", ha="left", va="center", color=PALETTE["Flat_LCDM"], fontsize=9.2)

    ax.annotate("", xy=(0.24, 0.5), xytext=(-4.0, 0.5),
                arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": PALETTE["Ink"]})
    ticks = [
        (-4.0, r"$a=10^{-4}$", 0.14),
        (x_eq, rf"$z_{{eq}}\approx {z_eq:.0f}$", 0.14),
        (x_l, rf"$z_\Lambda\approx {z_l:.2f}$", 0.82),
        (0.0, r"today: $a=1$", 0.14),
    ]
    for x, label, y_text in ticks:
        ax.plot([x, x], [0.25, 0.72], color=PALETTE["Ink"], lw=0.85)
        ax.text(x, y_text, label, ha="center", va="center", fontsize=8.8)
    ax.text(-1.86, 0.86, r"horizontal coordinate: $\log_{10}a$", ha="center", fontsize=9.2)
    savefig(out / "cosmic_history_timeline.pdf")


def fig_parameter_sensitivity(out: Path) -> None:
    z = np.linspace(0, 3.0, 1200)
    variations = [
        (Model("Baseline", r"baseline: $\Omega_m=0.30$, $\Omega_\Lambda=0.70$, $H_0=70$", omega_m=0.30, omega_l=0.70), 70.0, PALETTE["EdS"], "-"),
        (Model("Om025", r"lower matter: $\Omega_m=0.25$, $\Omega_\Lambda=0.75$", omega_m=0.25, omega_l=0.75), 70.0, PALETTE["Open_matter"], "-"),
        (Model("Om035", r"higher matter: $\Omega_m=0.35$, $\Omega_\Lambda=0.65$", omega_m=0.35, omega_l=0.65), 70.0, PALETTE["Closed_matter"], "-"),
        (Model("H065", r"lower Hubble scale: $H_0=65$", omega_m=0.30, omega_l=0.70), 65.0, PALETTE["Radiation"], "--"),
        (Model("H075", r"higher Hubble scale: $H_0=75$", omega_m=0.30, omega_l=0.70), 75.0, PALETTE["Flat_LCDM"], "--"),
    ]
    fig, ax = plt.subplots(figsize=(7.3, 4.55))
    for model, h0_value, color, ls in variations:
        dh_gpc = C / (h0_value * 1000.0 / MPC) / GPC
        chi = cumulative_trapezoid(1.0 / model.e_z(z), z, initial=0.0)
        dl = (1.0 + z) * dh_gpc * chi
        ax.plot(z, dl, lw=2.0, label=model.label, color=color, ls=ls)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"luminosity distance $D_L$ [Gpc]")
    ax.set_title("Parameter sensitivity in the distance-redshift relation")
    finish_axis(ax)
    savefig(out / "parameter_sensitivity.pdf")


def fig_parameter_heatmap(out: Path) -> None:
    z0 = 2.0
    h0_values = np.linspace(60.0, 80.0, 60)
    om_values = np.linspace(0.15, 0.45, 70)
    dl = np.zeros((len(om_values), len(h0_values)))
    z = np.linspace(0.0, z0, 1000)
    for i, om in enumerate(om_values):
        model = Model("flat", "flat", omega_m=float(om), omega_l=float(1.0 - om))
        chi = cumulative_trapezoid(1.0 / model.e_z(z), z, initial=0.0)[-1]
        for j, h0 in enumerate(h0_values):
            dh = C / (h0 * 1000.0 / MPC) / GPC
            dl[i, j] = (1.0 + z0) * dh * chi
    fig, ax = plt.subplots(figsize=(7.15, 4.7))
    image = ax.imshow(
        dl,
        origin="lower",
        aspect="auto",
        extent=[h0_values.min(), h0_values.max(), om_values.min(), om_values.max()],
    )
    contours = ax.contour(h0_values, om_values, dl, colors="white", linewidths=0.65, levels=7)
    ax.clabel(contours, inline=True, fontsize=7.8, fmt="%.1f")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(r"$D_L(z=2)$ [Gpc]")
    ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$\Omega_m$ in flat $\Lambda$CDM")
    ax.set_title(r"Two-parameter sensitivity of $D_L$ at fixed redshift")
    savefig(out / "parameter_heatmap.pdf")


def fig_pantheon_anchor(root: Path, out: Path) -> None:
    baseline = MODEL_BY_NAME["Flat_LCDM"]
    summary = pantheon_residual_summary(root, baseline)
    z = np.asarray(summary["z"])
    mu_obs = np.asarray(summary["mu_obs"])
    mu_err = np.asarray(summary["mu_err"])
    mu_model = np.asarray(summary["mu_model"])
    offset = float(summary["offset"])
    residual = np.asarray(summary["residual"])
    z_line = np.linspace(0.01, 1.4, 500)
    mu_line = np.asarray(distance_modulus(baseline, z_line)) + offset

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.3, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.08},
    )
    ax_top.errorbar(
        z,
        mu_obs,
        yerr=mu_err,
        fmt="o",
        ms=4.2,
        color=PALETTE["Flat_LCDM"],
        ecolor=PALETTE["Ink"],
        elinewidth=0.8,
        capsize=2.2,
        label="binned Pantheon+SH0ES distances",
    )
    ax_top.plot(z_line, mu_line, color=PALETTE["EdS"], lw=2.0, label=r"flat $\Lambda$CDM-like model + offset")
    ax_top.set_ylabel(r"distance modulus $\mu$")
    ax_top.set_title("Observational anchor: supernova distance modulus")
    finish_axis(ax_top)

    ax_bottom.axhline(0.0, color="0.35", lw=0.9)
    ax_bottom.errorbar(
        z,
        residual,
        yerr=mu_err,
        fmt="o",
        ms=4.2,
        color=PALETTE["Flat_LCDM"],
        ecolor=PALETTE["Ink"],
        elinewidth=0.8,
        capsize=2.2,
    )
    ax_bottom.set_xlabel("redshift $z$")
    ax_bottom.set_ylabel(r"$\Delta\mu$ [mag]")
    ax_bottom.text(
        0.02,
        0.93,
        rf"weighted RMS = {float(summary['weighted_rms']):.3f} mag; reduced $\chi^2$ = {float(summary['reduced_chi2']):.2f}",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
    )
    ax_bottom.tick_params(direction="out", length=4, width=0.8)
    savefig(out / "pantheon_distance_anchor.pdf")


def fig_pantheon_likelihood(root: Path, out: Path) -> None:
    summary = pantheon_full_fit_summary(root)
    omega = np.asarray(summary["omega_m_grid"], dtype=float)
    delta = np.asarray(summary["delta_chi2_grid"], dtype=float)
    omega_best = float(summary["omega_m_best"])
    err_minus = float(summary["omega_m_err_minus"])
    err_plus = float(summary["omega_m_err_plus"])

    fig, ax = plt.subplots(figsize=(7.25, 4.6))
    ax.plot(omega, delta, color=PALETTE["Flat_LCDM"], lw=2.15, label=r"Pantheon+SH0ES full covariance")
    ax.fill_between(omega, 0.0, delta, where=delta <= 1.0, color=PALETTE["Flat_LCDM"], alpha=0.18, interpolate=True)
    ax.axhline(1.0, color="0.35", lw=0.9, ls="--", label=r"$\Delta\chi^2=1$")
    ax.axhline(4.0, color="0.55", lw=0.8, ls=":", label=r"$\Delta\chi^2=4$")
    ax.axvline(omega_best, color=PALETTE["Ink"], lw=1.05)
    ax.axvline(BASELINE_OMEGA_M, color=PALETTE["EdS"], lw=1.05, ls="--", label=r"baseline $\Omega_m=0.300$")
    ax.scatter([omega_best], [0.0], s=34, color=PALETTE["Ink"], zorder=4)
    ax.text(
        0.04,
        0.93,
        rf"$\Omega_m={omega_best:.3f}^{{+{err_plus:.3f}}}_{{-{err_minus:.3f}}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.1,
        bbox={"facecolor": "white", "edgecolor": "0.78", "linewidth": 0.55, "boxstyle": "square,pad=0.25"},
    )
    ax.set_xlabel(r"matter density $\Omega_m$ in flat $\Lambda$CDM")
    ax.set_ylabel(r"$\Delta\chi^2$ after analytic offset marginalization")
    ax.set_ylim(0.0, min(9.0, max(4.5, float(np.nanmax(delta)) * 1.02)))
    ax.set_title("Pantheon+SH0ES shape likelihood profile")
    finish_axis(ax)
    savefig(out / "pantheon_likelihood_profile.pdf")


def convergence_summary(model: Model) -> list[tuple[str, int, float]]:
    z_eval = np.array([0.5, 1.0, 2.0, 3.0, 6.0])
    rows: list[tuple[str, int, float]] = []
    for n in [1100, 2200, 4400]:
        z_grid = np.linspace(0.0, Z_MAX_DISTANCE, n)
        dc_grid = comoving_distance_grid(model, z_grid)
        tl_grid = lookback_time_grid(model, z_grid)
        dl_grid = (1.0 + z_grid) * transverse_distance_from_dc(model, dc_grid)
        dc_interp = np.interp(z_eval, z_grid, dc_grid)
        tl_interp = np.interp(z_eval, z_grid, tl_grid)
        dl_interp = np.interp(z_eval, z_grid, dl_grid)
        dc_exact = np.array([dimensionless_comoving_distance(model, float(z)) * DH_GPC for z in z_eval])
        tl_exact = np.array([dimensionless_lookback_time(model, float(z)) * H0_INV_GYR for z in z_eval])
        dl_exact = (1.0 + z_eval) * np.array([transverse_distance_gpc(model, float(z)) for z in z_eval])
        rows.extend([
            ("$D_C(z)$", n, float(np.max(np.abs((dc_interp - dc_exact) / dc_exact)))),
            ("$t_L(z)$", n, float(np.max(np.abs((tl_interp - tl_exact) / tl_exact)))),
            ("$D_L(z)$", n, float(np.max(np.abs((dl_interp - dl_exact) / dl_exact)))),
        ])
    return rows


def astropy_crosscheck_summary() -> list[dict[str, float]]:
    """Compare the custom distance functions with Astropy's FlatLambdaCDM.

    Astropy's ``FlatLambdaCDM`` includes photons/neutrinos when ``Tcmb0`` is
    nonzero. The cross-check deliberately sets ``Tcmb0=0`` and uses a matching
    no-radiation model so that the comparison isolates the distance equations
    rather than small differences in radiation conventions.
    """
    try:
        import astropy.units as u
        from astropy.cosmology import FlatLambdaCDM
    except ImportError as exc:  # pragma: no cover - exercised by environment setup
        raise RuntimeError("Astropy is required for the external cross-check; run pip install -r requirements.txt") from exc

    model = ASTROPY_CHECK_MODEL
    astropy_model = FlatLambdaCDM(
        H0=H0_KM_S_MPC * u.km / u.s / u.Mpc,
        Om0=BASELINE_OMEGA_M,
        Tcmb0=0.0 * u.K,
    )
    rows: list[dict[str, float]] = []
    for z in [0.5, 1.0, 2.0, 3.0]:
        ours_t = dimensionless_lookback_time(model, z) * H0_INV_GYR
        ours_dc = dimensionless_comoving_distance(model, z) * DH_GPC
        ours_dl = (1.0 + z) * ours_dc
        ours_da = ours_dc / (1.0 + z)

        astropy_t = astropy_model.lookback_time(z).to_value(u.Gyr)
        astropy_dc = astropy_model.comoving_distance(z).to_value(u.Gpc)
        astropy_dl = astropy_model.luminosity_distance(z).to_value(u.Gpc)
        astropy_da = astropy_model.angular_diameter_distance(z).to_value(u.Gpc)

        rows.append({
            "z": z,
            "lookback_rel": abs((ours_t - astropy_t) / astropy_t),
            "dc_rel": abs((ours_dc - astropy_dc) / astropy_dc),
            "dl_rel": abs((ours_dl - astropy_dl) / astropy_dl),
            "da_rel": abs((ours_da - astropy_da) / astropy_da),
            "max_rel": max(
                abs((ours_t - astropy_t) / astropy_t),
                abs((ours_dc - astropy_dc) / astropy_dc),
                abs((ours_dl - astropy_dl) / astropy_dl),
                abs((ours_da - astropy_da) / astropy_da),
            ),
        })
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KiB"
    return f"{num_bytes} B"


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_data_manifest_table(root: Path, out: Path) -> None:
    entries = [
        (
            "Pantheon+ release README",
            root / "data" / "Pantheon+README",
            "Source-data documentation retained with the release files",
        ),
        (
            "Pantheon+SH0ES distance table",
            root / "data" / "Pantheon+SH0ES.dat",
            "Full source table used for the covariance-aware likelihood",
        ),
        (
            "Pantheon+SH0ES STAT+SYS covariance",
            root / "data" / "Pantheon+SH0ES_STAT+SYS.cov",
            "Full covariance matrix sliced to the likelihood subset",
        ),
        (
            "Binned Pantheon+ anchor",
            root / "data" / "pantheon_plus_binned.csv",
            "Compact derived visual anchor; not used for the covariance fit",
        ),
    ]
    tex = [
        r"\begingroup",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.29\textwidth}>{\raggedright\arraybackslash}X>{\raggedleft\arraybackslash}p{0.10\textwidth}>{\raggedright\arraybackslash}p{0.11\textwidth}@{}}",
        r"\toprule",
        r"Retained file & Role & Size & SHA-256 \\",
        r"\midrule",
    ]
    for label, path, role in entries:
        tex.append(
            rf"{tex_escape(label)} & {tex_escape(role)} & "
            rf"{format_size(path.stat().st_size)} & \texttt{{{file_sha256(path)[:10]}}} \\"
        )
    tex.extend([r"\bottomrule", r"\end{tabularx}", r"\endgroup"])
    (out / "data_manifest_table.tex").write_text("\n".join(tex), encoding="utf-8")


def write_results_table(root: Path, out: Path) -> None:
    rows = []
    for model in MODELS:
        age = dimensionless_age(model, 1.0) * H0_INV_GYR
        horizon = dimensionless_horizon(model, 1.0) * DH_GPC
        turn = "--"
        if model.turnaround_a is not None:
            t_turn = dimensionless_age(model, model.turnaround_a * (1.0 - 1.0e-8)) * H0_INV_GYR
            turn = rf"$a_{{\max}}\simeq {model.turnaround_a:.2f}$, $t_{{\max}}\simeq {t_turn:.1f}$ Gyr"
        rows.append((model.label, age, horizon, turn, fate_text(model)))

    tex = [
        r"\begin{tabularx}{\textwidth}{lcc>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Model & Age at $a=1$ [Gyr] & $D_{\rm hor}(a=1)$ [Gpc] & Future marker & Qualitative fate \\",
        r"\midrule",
    ]
    for label, age, horizon, turn, fate in rows:
        tex.append(rf"{label} & {age:.2f} & {horizon:.2f} & {turn} & {fate} \\")
    tex.extend([r"\bottomrule", r"\end{tabularx}"])
    (out / "results_summary_table.tex").write_text("\n".join(tex), encoding="utf-8")

    constants = [
        r"\begin{tabularx}{\textwidth}{llX}",
        r"\toprule",
        r"Quantity & Value & Notes \\",
        r"\midrule",
        rf"$H_0$ & {H0_KM_S_MPC:.1f} km s$^{{-1}}$ Mpc$^{{-1}}$ & reference value used in all baseline runs \\",
        rf"Baseline $(\Omega_r,\Omega_m,\Omega_\Lambda)$ & ({BASELINE_OMEGA_R:.1e}, {BASELINE_OMEGA_M:.3f}, {BASELINE_OMEGA_L:.5f}) & radiation-inclusive flat $\Lambda$CDM-like model \\",
        rf"$H_0^{{-1}}$ & {H0_INV_GYR:.3f} Gyr & Hubble time associated with the reference $H_0$ \\",
        rf"$c/H_0$ & {DH_GPC:.3f} Gpc & Hubble distance associated with the reference $H_0$ \\",
        rf"ODE tolerances & rtol $=10^{{-10}}$, atol $=10^{{-12}}$ & DOP853 controls in solve\_ivp \\",
        rf"Quadrature tolerances & epsrel $=10^{{-11}}$ & adaptive scalar-integral checks \\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    (out / "generated_constants_table.tex").write_text("\n".join(constants), encoding="utf-8")

    # Analytic checks in dimensionless units.
    z_check = 3.0
    eds = MODEL_BY_NAME["EdS"]
    rad = MODEL_BY_NAME["Radiation"]
    checks = [
        ("EdS age", dimensionless_age(eds, 1.0), 2.0 / 3.0),
        ("radiation age", dimensionless_age(rad, 1.0), 1.0 / 2.0),
        (r"EdS $D_C/D_H$ at $z=3$", dimensionless_comoving_distance(eds, z_check), 2.0 * (1.0 - 1.0 / np.sqrt(1.0 + z_check))),
        (r"radiation $D_C/D_H$ at $z=3$", dimensionless_comoving_distance(rad, z_check), z_check / (1.0 + z_check)),
    ]
    tex = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Benchmark & Numerical value & Analytic value & Absolute error \\",
        r"\midrule",
    ]
    for label, num, exact in checks:
        tex.append(rf"{label} & {num:.12f} & {exact:.12f} & {abs(num-exact):.2e} \\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "validation_table.tex").write_text("\n".join(tex), encoding="utf-8")

    # A compact table of distances in the baseline flat LCDM-like model.
    baseline = MODEL_BY_NAME["Flat_LCDM"]
    tex = [
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$z$ & $t_L$ [Gyr] & $D_C$ [Gpc] & $D_L$ [Gpc] \\",
        r"\midrule",
    ]
    for z in [0.5, 1.0, 2.0, 3.0, 6.0]:
        t_l = dimensionless_lookback_time(baseline, z) * H0_INV_GYR
        dc = dimensionless_comoving_distance(baseline, z) * DH_GPC
        dl = (1.0 + z) * dc
        tex.append(rf"{z:.1f} & {t_l:.2f} & {dc:.2f} & {dl:.2f} \\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "distance_sample_table.tex").write_text("\n".join(tex), encoding="utf-8")

    # Derived diagnostics for the radiation-inclusive baseline.
    z_da, da_max = angular_diameter_maximum(baseline)
    z_acc = transition_redshift(baseline)
    z_eq = BASELINE_OMEGA_M / BASELINE_OMEGA_R - 1.0
    z_lambda = (BASELINE_OMEGA_L / BASELINE_OMEGA_M) ** (1.0 / 3.0) - 1.0
    dl70 = float(distance_modulus(baseline, 2.0))
    h65 = C / (65.0 * 1000.0 / MPC) / GPC
    h75 = C / (75.0 * 1000.0 / MPC) / GPC
    chi_z2 = dimensionless_comoving_distance(baseline, 2.0)
    dl_gpc_65 = (1.0 + 2.0) * h65 * chi_z2
    dl_gpc_75 = (1.0 + 2.0) * h75 * chi_z2
    dl_gpc_70 = (1.0 + 2.0) * DH_GPC * chi_z2
    tex = [
        r"\begin{tabularx}{\textwidth}{l>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Diagnostic & Baseline value \\",
        r"\midrule",
        rf"Age at $a=1$ & {dimensionless_age(baseline, 1.0) * H0_INV_GYR:.2f} Gyr \\",
        rf"Matter-radiation equality & $z_{{\rm eq}}\simeq {z_eq:.0f}$ \\",
        rf"Matter-$\Lambda$ equality & $z_\Lambda\simeq {z_lambda:.2f}$ \\",
        rf"Acceleration transition & $q(z)=0$ at $z\simeq {z_acc:.2f}$ \\",
        rf"Maximum angular-diameter distance & $D_A$ peaks at $z\simeq {z_da:.2f}$ with $D_A\simeq {da_max:.2f}$ Gpc \\",
        rf"Particle horizon today & {dimensionless_horizon(baseline, 1.0) * DH_GPC:.2f} Gpc \\",
        rf"Distance modulus at $z=2$ & $\mu\simeq {dl70:.2f}$ mag for the baseline distance scale \\",
        rf"$H_0$ sensitivity at $z=2$ & $D_L$ changes by {100.0 * (dl_gpc_65 / dl_gpc_70 - 1.0):+.1f}\% for $H_0=65$ and {100.0 * (dl_gpc_75 / dl_gpc_70 - 1.0):+.1f}\% for $H_0=75$ \\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    (out / "baseline_diagnostics_table.tex").write_text("\n".join(tex), encoding="utf-8")

    tex = [
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"Quantity & Grid points & Max relative error at sampled redshifts \\",
        r"\midrule",
    ]
    for label, n, relerr in convergence_summary(baseline):
        tex.append(rf"{label} & {n} & {relerr:.2e} \\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "convergence_table.tex").write_text("\n".join(tex), encoding="utf-8")

    tex = [
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$z$ & $t_L$ rel. diff. & $D_C$ rel. diff. & $D_L$ rel. diff. & $D_A$ rel. diff. \\",
        r"\midrule",
    ]
    for row in astropy_crosscheck_summary():
        tex.append(
            rf"{row['z']:.1f} & {row['lookback_rel']:.2e} & {row['dc_rel']:.2e} & "
            rf"{row['dl_rel']:.2e} & {row['da_rel']:.2e} \\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "astropy_crosscheck_table.tex").write_text("\n".join(tex), encoding="utf-8")

    summary = pantheon_residual_summary(root, baseline)
    tex = [
        r"\begin{tabularx}{\textwidth}{l>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        rf"Binned sample & {int(summary['n_bins'])} redshift bins from {int(summary['n_supernovae'])} non-calibrator supernova entries \\",
        rf"Offset applied to model & $\Delta\mu_0 = {float(summary['offset']):+.3f}$ mag, absorbing absolute-calibration differences \\",
        rf"Weighted residual RMS & {float(summary['weighted_rms']):.3f} mag \\",
        rf"Reduced $\chi^2$ & {float(summary['reduced_chi2']):.2f} for diagonal binned errors only \\",
        r"Scope & Visual distance-anchor check only; the covariance-aware shape likelihood is reported separately. \\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    (out / "pantheon_anchor_table.tex").write_text("\n".join(tex), encoding="utf-8")

    full_fit = pantheon_full_fit_summary(root)
    omega_best = float(full_fit["omega_m_best"])
    err_minus = float(full_fit["omega_m_err_minus"])
    err_plus = float(full_fit["omega_m_err_plus"])
    tex = [
        r"\begin{tabularx}{\textwidth}{l>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        rf"Dataset selection & {int(full_fit['n_used'])} non-calibrator Pantheon+SH0ES supernovae, "
        rf"${float(full_fit['z_min']):.3f}<z_{{\rm HD}}<{float(full_fit['z_max']):.3f}$ \\",
        rf"Covariance treatment & Full STAT+SYS covariance subset, ${int(full_fit['cov_dim'])}\times {int(full_fit['cov_dim'])}$ \\",
        rf"Fitted parameter & $\Omega_m={omega_best:.3f}^{{+{err_plus:.3f}}}_{{-{err_minus:.3f}}}$ "
        r"for flat radiation-inclusive $\Lambda$CDM shape \\",
        rf"Marginalized offset & $\Delta\mu_0={float(full_fit['offset']):+.3f}$ mag \\",
        rf"Minimum $\chi^2$ & {float(full_fit['chi2_min']):.2f} for {int(full_fit['dof'])} degrees of freedom \\",
        rf"Reduced $\chi^2$ & {float(full_fit['reduced_chi2']):.3f} \\",
        rf"Baseline comparison & Fixed $\Omega_m=0.300$ gives $\Delta\chi^2={float(full_fit['delta_chi2_baseline']):.2f}$ after its own offset fit \\",
        r"Scope & Supernova-only shape fit with fixed radiation term and reference distance scale; not a calibrated $H_0$ or multi-probe cosmological constraint. \\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    (out / "pantheon_full_fit_table.tex").write_text("\n".join(tex), encoding="utf-8")
    write_data_manifest_table(root, out)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig_flrw_diagram(out)
    fig_scale_factor(out)
    fig_analytic_validation(out)
    fig_hubble_parameter(out)
    fig_density_fractions(out)
    fig_lookback_time(out)
    fig_comoving_distance(out)
    fig_luminosity_distance(out)
    fig_angular_distance(out)
    fig_particle_horizon(out)
    fig_cosmic_timeline(out)
    fig_parameter_sensitivity(out)
    fig_parameter_heatmap(out)
    fig_pantheon_anchor(root, out)
    fig_pantheon_likelihood(root, out)
    write_results_table(root, out)

    print(f"Generated figures and tables in: {out}")
    print(f"Hubble time: {H0_INV_GYR:.3f} Gyr")
    print(f"Hubble distance: {DH_GPC:.3f} Gpc")


if __name__ == "__main__":
    main()
