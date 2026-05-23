"""Publication-figure generation for the FLRW paper release."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.integrate import cumulative_trapezoid

from .distances import (
    comoving_distance_grid,
    conformal_integrand_a,
    distance_modulus,
    early_time_age,
    early_time_conformal,
    lookback_time_grid,
    model_time_curve,
    time_integrand_a,
    transverse_distance_from_dc,
)
from .likelihoods import desi_bao_dr2_summary, pantheon_full_fit_summary, pantheon_residual_summary
from .models import (
    A_MIN,
    BASELINE_OMEGA_L,
    BASELINE_OMEGA_M,
    BASELINE_OMEGA_R,
    C,
    DH_GPC,
    GPC,
    H0_INV_GYR,
    MODEL_BY_NAME,
    MODELS,
    MPC,
    PALETTE,
    Z_MAX_DISTANCE,
    Z_MAX_HUBBLE,
    Model,
    model_line_style,
)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10.8,
    "axes.titlesize": 12.4,
    "axes.labelsize": 11.4,
    "axes.titleweight": "semibold",
    "legend.fontsize": 8.8,
    "xtick.labelsize": 9.6,
    "ytick.labelsize": 9.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": PALETTE["Ink"],
    "axes.labelcolor": PALETTE["Ink"],
    "axes.grid": True,
    "grid.color": PALETTE["Grid"],
    "grid.alpha": 0.38,
    "grid.linewidth": 0.45,
    "legend.framealpha": 0.92,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#c9d0d8",
    "figure.facecolor": "#ffffff",
    "axes.facecolor": PALETTE["Paper"],
    "figure.dpi": 160,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PDF_METADATA = {
    "Creator": "FLRW numerical simulation figure generator",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2026, 5, 22, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 5, 22, tzinfo=timezone.utc),
}

def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    for ax in fig.axes:
        ax.set_title("")
        ax.tick_params(colors=PALETTE["Ink"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["Ink"])
    fig.savefig(path, metadata=PDF_METADATA)
    plt.close(fig)


def finish_axis(ax: plt.Axes, legend: bool = True) -> None:
    ax.tick_params(direction="out", length=4, width=0.8)
    ax.grid(True, color=PALETTE["Grid"], alpha=0.38, linewidth=0.45)
    ax.set_axisbelow(True)
    if legend:
        leg = ax.legend(frameon=True, fancybox=False, edgecolor="#c9d0d8", facecolor="#ffffff")
        if leg is not None:
            leg.get_frame().set_linewidth(0.6)
            leg.get_frame().set_alpha(0.92)


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


def fig_desi_bao_comparison(root: Path, out: Path) -> None:
    summary = desi_bao_dr2_summary(root)
    z = np.asarray(summary["z"], dtype=float)
    observed = np.asarray(summary["observed"], dtype=float)
    sigma = np.asarray(summary["sigma"], dtype=float)
    prediction = np.asarray(summary["prediction_scaled"], dtype=float)
    residual = np.asarray(summary["normalized_residual_scaled"], dtype=float)
    quantity = np.asarray(summary["quantity"], dtype=object)
    labels = {
        "DV_over_rs": r"$D_V/r_d$",
        "DM_over_rs": r"$D_M/r_d$",
        "DH_over_rs": r"$D_H/r_d$",
    }
    colors = {
        "DV_over_rs": PALETTE["Radiation"],
        "DM_over_rs": PALETTE["Flat_LCDM"],
        "DH_over_rs": PALETTE["Open_matter"],
    }

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.3, 6.05),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )
    for q in ["DV_over_rs", "DM_over_rs", "DH_over_rs"]:
        mask = quantity == q
        ax_top.errorbar(
            z[mask],
            observed[mask],
            yerr=sigma[mask],
            fmt="o",
            ms=4.4,
            color=colors[q],
            ecolor=colors[q],
            elinewidth=0.85,
            capsize=2.2,
            label=f"DESI DR2 {labels[q]}",
        )
        ax_top.scatter(
            z[mask],
            prediction[mask],
            marker="x",
            s=42,
            color=PALETTE["Ink"],
            linewidths=1.1,
            label=f"baseline shape {labels[q]}" if q == "DV_over_rs" else None,
        )
        ax_bottom.scatter(z[mask], residual[mask], s=28, color=colors[q], label=labels[q])

    ax_top.set_ylabel(r"BAO distance ratio")
    ax_top.set_title("DESI DR2 BAO compressed distances versus baseline FLRW shape")
    ax_top.text(
        0.02,
        0.94,
        rf"$\alpha_{{\rm BAO}}={float(summary['alpha']):.3f}$; "
        rf"$\chi^2={float(summary['chi2_scaled']):.2f}$ for {int(summary['dof_scaled'])} dof",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        bbox={"facecolor": "white", "edgecolor": "0.78", "linewidth": 0.55, "boxstyle": "square,pad=0.25"},
    )
    finish_axis(ax_top)

    ax_bottom.axhline(0.0, color="0.35", lw=0.9)
    ax_bottom.axhspan(-1.0, 1.0, color=PALETTE["Grid"], alpha=0.16, zorder=0)
    ax_bottom.set_xlabel("redshift $z$")
    ax_bottom.set_ylabel(r"scaled residual [$\sigma$]")
    ax_bottom.set_ylim(-3.0, 3.0)
    finish_axis(ax_bottom)
    savefig(out / "desi_bao_dr2_comparison.pdf")
