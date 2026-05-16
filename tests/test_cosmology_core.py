import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import flrw_paper as cosmo  # noqa: E402


def test_analytic_ages_match_known_limits():
    eds = cosmo.MODEL_BY_NAME["EdS"]
    rad = cosmo.MODEL_BY_NAME["Radiation"]

    assert math.isclose(cosmo.dimensionless_age(eds, 1.0), 2.0 / 3.0, rel_tol=0.0, abs_tol=5e-10)
    assert math.isclose(cosmo.dimensionless_age(rad, 1.0), 0.5, rel_tol=0.0, abs_tol=5e-12)


def test_analytic_comoving_distances_match_known_limits():
    eds = cosmo.MODEL_BY_NAME["EdS"]
    rad = cosmo.MODEL_BY_NAME["Radiation"]
    z = 3.0

    eds_exact = 2.0 * (1.0 - 1.0 / math.sqrt(1.0 + z))
    rad_exact = z / (1.0 + z)

    assert math.isclose(cosmo.dimensionless_comoving_distance(eds, z), eds_exact, rel_tol=0.0, abs_tol=5e-12)
    assert math.isclose(cosmo.dimensionless_comoving_distance(rad, z), rad_exact, rel_tol=0.0, abs_tol=5e-12)


def test_distance_duality_for_curved_and_flat_models():
    for model in cosmo.MODELS:
        z = np.array([0.1, 0.7, 1.5, 3.0])
        dc = cosmo.comoving_distance_grid(model, np.linspace(0.0, 3.0, 1000))
        grid_z = np.linspace(0.0, 3.0, 1000)
        dm = cosmo.transverse_distance_from_dc(model, np.interp(z, grid_z, dc))
        dl = (1.0 + z) * dm
        da = dm / (1.0 + z)

        np.testing.assert_allclose(dl, (1.0 + z) ** 2 * da, rtol=1e-12, atol=1e-12)


def test_flat_branch_keeps_transverse_distance_equal_to_line_of_sight():
    flat = cosmo.MODEL_BY_NAME["Flat_LCDM"]
    dc = np.array([0.5, 1.0, 2.5, 4.0])

    np.testing.assert_allclose(cosmo.transverse_distance_from_dc(flat, dc), dc, rtol=0.0, atol=0.0)


def test_closed_turnaround_and_acceleration_transition():
    closed = cosmo.MODEL_BY_NAME["Closed_matter"]
    baseline = cosmo.MODEL_BY_NAME["Flat_LCDM"]

    assert math.isclose(closed.turnaround_a, -closed.omega_m / closed.omega_k, rel_tol=0.0, abs_tol=1e-12)
    z_acc = cosmo.transition_redshift(baseline)
    assert z_acc is not None
    assert 0.6 < z_acc < 0.8


def test_pantheon_anchor_is_loaded_and_offset_marginalized():
    summary = cosmo.pantheon_residual_summary(ROOT, cosmo.MODEL_BY_NAME["Flat_LCDM"])

    assert int(summary["n_bins"]) == 11
    assert int(summary["n_supernovae"]) == 1572
    assert abs(float(summary["offset"])) < 0.25
    assert float(summary["weighted_rms"]) < 0.05


def test_fast_distance_modulus_matches_scalar_quadrature():
    model = cosmo.MODEL_BY_NAME["Flat_LCDM"]
    z = np.linspace(0.02, 1.4, 80)

    fast = cosmo.distance_modulus_fast(model, z)
    scalar = np.asarray(cosmo.distance_modulus(model, z), dtype=float)

    np.testing.assert_allclose(fast, scalar, rtol=0.0, atol=2e-5)


def test_pantheon_full_covariance_likelihood_matches_regression():
    summary = cosmo.pantheon_full_fit_summary(ROOT)

    assert int(summary["n_used"]) == 1572
    assert 0.31 < float(summary["omega_m_best"]) < 0.35
    assert 0.015 < float(summary["omega_m_err_minus"]) < 0.025
    assert 0.015 < float(summary["omega_m_err_plus"]) < 0.025
    assert 1300.0 < float(summary["chi2_min"]) < 1450.0
    assert float(summary["reduced_chi2"]) < 1.0
    assert float(summary["delta_chi2_baseline"]) > 2.0


def test_desi_bao_dr2_profiled_scale_likelihood_matches_regression():
    summary = cosmo.desi_bao_dr2_summary(ROOT)

    assert int(summary["n_measurements"]) == 13
    assert 1.0 < float(summary["alpha"]) < 1.04
    assert 9.0 < float(summary["chi2_scaled"]) < 12.0
    assert int(summary["dof_scaled"]) == 12
    assert float(summary["chi2_scaled"]) < float(summary["chi2_fixed"])


def test_bao_observable_predictions_are_positive_for_all_desi_points():
    data = cosmo.load_desi_bao_dr2(ROOT)
    model = cosmo.MODEL_BY_NAME["Flat_LCDM"]
    predictions = [
        cosmo.bao_prediction(model, float(z), str(quantity))
        for z, quantity in zip(data["z"], data["quantity"])
    ]

    assert len(predictions) == 13
    assert min(predictions) > 0.0


def test_dense_grid_convergence_stays_below_reported_tolerance_scale():
    rows = cosmo.convergence_summary(cosmo.MODEL_BY_NAME["Flat_LCDM"])
    high_resolution_errors = [relerr for _label, n, relerr in rows if n == 4400]

    assert high_resolution_errors
    assert max(high_resolution_errors) < 1e-6


def test_external_astropy_crosscheck_agrees_with_custom_distances():
    rows = cosmo.astropy_crosscheck_summary()

    assert len(rows) == 4
    assert max(float(row["max_rel"]) for row in rows) < 1e-8
