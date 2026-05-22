"""Generated LaTeX table helpers for the public FLRW release."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .distances import (
    angular_diameter_maximum,
    comoving_distance_grid,
    dimensionless_age,
    dimensionless_comoving_distance,
    dimensionless_horizon,
    dimensionless_lookback_time,
    distance_modulus,
    lookback_time_grid,
    transverse_distance_from_dc,
    transverse_distance_gpc,
    transition_redshift,
)
from .likelihoods import desi_bao_dr2_summary, pantheon_full_fit_summary, pantheon_residual_summary
from .models import (
    ASTROPY_CHECK_MODEL,
    BASELINE_OMEGA_L,
    BASELINE_OMEGA_M,
    BASELINE_OMEGA_R,
    C,
    DH_GPC,
    GPC,
    H0_INV_GYR,
    H0_KM_S_MPC,
    MODEL_BY_NAME,
    MODELS,
    MPC,
    Model,
    Z_MAX_DISTANCE,
    fate_text,
)

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


def format_rel_diff(value: float) -> str:
    if value < 1.0e-14:
        return r"$<10^{-14}$"
    return f"{value:.2e}"


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
        (
            "DESI DR2 BAO compressed means",
            root / "data" / "desi_dr2_bao_all_gccomb_mean.txt",
            "Compressed BAO observables used for the DESI-style distance check",
        ),
        (
            "DESI DR2 BAO covariance",
            root / "data" / "desi_dr2_bao_all_gccomb_cov.txt",
            "Covariance matrix for the compressed BAO data vector",
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
        r"\begingroup",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.25\textwidth}cc>{\raggedright\arraybackslash}X@{}}",
        r"\toprule",
        r"Model & Age [Gyr] & $D_{\rm hor}$ [Gpc] & Fate or future marker \\",
        r"\midrule",
    ]
    for label, age, horizon, turn, fate in rows:
        future = fate if turn == "--" else rf"{fate}; {turn}"
        tex.append(rf"{label} & {age:.2f} & {horizon:.2f} & {future} \\")
    tex.extend([r"\bottomrule", r"\end{tabularx}", r"\endgroup"])
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
            rf"{row['z']:.1f} & {format_rel_diff(row['lookback_rel'])} & {format_rel_diff(row['dc_rel'])} & "
            rf"{format_rel_diff(row['dl_rel'])} & {format_rel_diff(row['da_rel'])} \\"
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

    bao = desi_bao_dr2_summary(root, baseline)
    tex = [
        r"\begin{tabularx}{\textwidth}{l>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        rf"Data vector & {int(bao['n_measurements'])} DESI DR2 compressed BAO measurements: $D_V/r_d$, $D_M/r_d$, and $D_H/r_d$ \\",
        rf"Reference sound horizon & $r_d={float(bao['rd_mpc']):.2f}$ Mpc, used only to form the baseline prediction before scale profiling \\",
        rf"Scale nuisance & $\alpha_{{\rm BAO}}={float(bao['alpha']):.3f}$ multiplying all baseline BAO ratios \\",
        rf"Fixed-scale $\chi^2$ & {float(bao['chi2_fixed']):.2f} for {int(bao['dof_fixed'])} data-vector entries \\",
        rf"Profiled-scale $\chi^2$ & {float(bao['chi2_scaled']):.2f} for {int(bao['dof_scaled'])} degrees of freedom \\",
        r"Scope & DESI-style compressed-distance consistency check for the baseline FLRW shape; not a joint BAO cosmological parameter fit. \\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    (out / "desi_bao_dr2_fit_table.tex").write_text("\n".join(tex), encoding="utf-8")

    z = np.asarray(bao["z"], dtype=float)
    quantity = np.asarray(bao["quantity"], dtype=object)
    observed = np.asarray(bao["observed"], dtype=float)
    predicted = np.asarray(bao["prediction_scaled"], dtype=float)
    sigma = np.asarray(bao["sigma"], dtype=float)
    tex = [
        r"\begin{tabular}{cclrrr}",
        r"\toprule",
        r"$z$ & Observable & Source & Observed & Model & Residual/$\sigma$ \\",
        r"\midrule",
    ]
    q_label = {"DV_over_rs": r"$D_V/r_d$", "DM_over_rs": r"$D_M/r_d$", "DH_over_rs": r"$D_H/r_d$"}
    for zi, qi, obs_i, pred_i, sig_i in zip(z, quantity, observed, predicted, sigma):
        tex.append(
            rf"{zi:.3f} & {q_label[str(qi)]} & DESI DR2 & {obs_i:.3f} & {pred_i:.3f} & {(pred_i - obs_i) / sig_i:+.2f} \\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "desi_bao_dr2_measurements_table.tex").write_text("\n".join(tex), encoding="utf-8")
    write_data_manifest_table(root, out)
