"""Supernova and BAO likelihood utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import brentq, minimize_scalar

from .datasets import load_desi_bao_dr2, load_pantheon_binned, load_pantheon_full_likelihood
from .distances import bao_prediction, distance_modulus, distance_modulus_fast
from .models import BASELINE_OMEGA_M, BASELINE_OMEGA_R, BAO_RD_MPC, MODEL_BY_NAME, Model, flat_lcdm_shape_model

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


def desi_bao_dr2_summary(root: Path, model: Model | None = None) -> dict[str, object]:
    """Compare a model to DESI DR2 compressed BAO measurements."""
    if model is None:
        model = MODEL_BY_NAME["Flat_LCDM"]
    data = load_desi_bao_dr2(root)
    z = np.asarray(data["z"], dtype=float)
    obs = np.asarray(data["value"], dtype=float)
    quantity = np.asarray(data["quantity"], dtype=object)
    covariance = np.asarray(data["covariance"], dtype=float)
    prediction = np.asarray([bao_prediction(model, float(zi), str(qi)) for zi, qi in zip(z, quantity)], dtype=float)
    inv_cov = np.linalg.inv(covariance)
    delta_fixed = prediction - obs
    chi2_fixed = float(delta_fixed @ inv_cov @ delta_fixed)
    alpha = float((prediction @ inv_cov @ obs) / (prediction @ inv_cov @ prediction))
    scaled_prediction = alpha * prediction
    delta_scaled = scaled_prediction - obs
    chi2_scaled = float(delta_scaled @ inv_cov @ delta_scaled)
    sigma = np.sqrt(np.diag(covariance))
    return {
        "z": z,
        "quantity": quantity,
        "observed": obs,
        "sigma": sigma,
        "prediction_fixed": prediction,
        "prediction_scaled": scaled_prediction,
        "residual_fixed": delta_fixed,
        "residual_scaled": delta_scaled,
        "normalized_residual_scaled": delta_scaled / sigma,
        "covariance": covariance,
        "alpha": alpha,
        "rd_mpc": BAO_RD_MPC,
        "chi2_fixed": chi2_fixed,
        "chi2_scaled": chi2_scaled,
        "dof_fixed": int(len(obs)),
        "dof_scaled": int(len(obs) - 1),
        "n_measurements": int(len(obs)),
    }
