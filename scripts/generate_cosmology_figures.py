#!/usr/bin/env python3
"""Regenerate all public FLRW figures and generated tables."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flrw_paper.plotting import (  # noqa: E402
    fig_analytic_validation,
    fig_angular_distance,
    fig_comoving_distance,
    fig_cosmic_timeline,
    fig_density_fractions,
    fig_desi_bao_comparison,
    fig_flrw_diagram,
    fig_hubble_parameter,
    fig_lookback_time,
    fig_luminosity_distance,
    fig_pantheon_anchor,
    fig_pantheon_likelihood,
    fig_parameter_heatmap,
    fig_parameter_sensitivity,
    fig_particle_horizon,
    fig_scale_factor,
)
from flrw_paper.tables import DH_GPC, H0_INV_GYR, write_results_table  # noqa: E402


def main() -> None:
    out = ROOT / "figures"
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
    fig_pantheon_anchor(ROOT, out)
    fig_pantheon_likelihood(ROOT, out)
    fig_desi_bao_comparison(ROOT, out)
    write_results_table(ROOT, out)

    print(f"Generated figures and tables in: {out}")
    print(f"Hubble time: {H0_INV_GYR:.3f} Gyr")
    print(f"Hubble distance: {DH_GPC:.3f} Gpc")


if __name__ == "__main__":
    main()
