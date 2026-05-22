# Figure index

The figures in this repository are generated outputs, not hand-edited
illustrations. Run `python scripts/generate_cosmology_figures.py` to regenerate
the vector PDFs in `figures/`.

README preview PNGs live in `docs/assets/` and can be regenerated with
`python scripts/render_readme_assets.py` if Poppler's `pdftoppm` is available.

## Theory and validation

| File | What it shows |
| --- | --- |
| `figures/flrw_expansion_diagram.pdf` | Conceptual relationship between scale factor, physical separation, and redshift. |
| `figures/analytic_scale_factor_validation.pdf` | Numerical solutions checked against analytic scale-factor limits. |
| `figures/convergence_table.tex` | Grid convergence of the main numerical integrations. |
| `figures/astropy_crosscheck_table.tex` | Distance cross-check against Astropy under matching assumptions. |

## Expansion history and distances

| File | What it shows |
| --- | --- |
| `figures/scale_factor_evolution.pdf` | Scale-factor histories for representative FLRW models. |
| `figures/hubble_parameter.pdf` | Expansion-rate evolution. |
| `figures/lookback_time.pdf` | Lookback time as a function of redshift. |
| `figures/comoving_distance.pdf` | Line-of-sight comoving distance. |
| `figures/luminosity_distance.pdf` | Luminosity distance, including curvature and model comparisons. |
| `figures/angular_diameter_distance.pdf` | Angular-diameter distance and its turnover. |
| `figures/particle_horizon.pdf` | Particle-horizon evolution. |
| `figures/density_fractions_lcdm.pdf` | Radiation, matter, and dark-energy density fractions. |
| `figures/cosmic_history_timeline.pdf` | Key redshift and time markers in the baseline model. |

## Data-facing checks

| File | What it shows |
| --- | --- |
| `figures/pantheon_distance_anchor.pdf` | Binned Pantheon+SH0ES distance-modulus anchor. |
| `figures/pantheon_likelihood_profile.pdf` | Full-covariance Pantheon+SH0ES shape likelihood profile. |
| `figures/desi_bao_dr2_comparison.pdf` | DESI DR2 compressed BAO comparison and scaled residuals. |
| `figures/parameter_sensitivity.pdf` | Distance sensitivity to selected background parameters. |
| `figures/parameter_heatmap.pdf` | Two-parameter distance response over `H0` and `Omega_m`. |

## Generated tables

The `.tex` files in `figures/` are generated numerical tables used by the
manuscript-side project and retained here so the reported numbers can be
audited without reading the paper source.

