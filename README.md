# FLRW Numerical Simulation Code Release

This repository contains the reproducible simulation code, data files, tests,
and generated figures for the computational cosmology project:

> Numerical Experiments in Cosmological Expansion: Scale Factors, Redshift, and
> Horizon Distances in FLRW Universes

The paper source files are intentionally not included here. This repository is
the public technical release: it is meant to let a reader inspect the numerical
pipeline, rerun the simulations, verify the Pantheon+SH0ES data inputs, rebuild
the figures, and reproduce the main numerical claims.

## What Is Included

- `src/flrw_paper/` contains the reusable simulation package, organized into
  models, distances, datasets, likelihoods, plotting, and table utilities.
- `scripts/generate_cosmology_figures.py` is a small orchestration script that
  regenerates all numerical figures and LaTeX-ready result tables.
- `scripts/download_pantheon_data.py` verifies or downloads the official
  Pantheon+SH0ES source files by size and SHA-256 hash.
- `tests/test_cosmology_core.py` checks analytic limits, distance identities,
  Astropy agreement, convergence, and the Pantheon+ covariance likelihood.
- `data/` contains the retained Pantheon+SH0ES and DESI DR2 BAO data inputs used
  by the simulations.
- `figures/` contains regenerated vector figures and generated result tables.
- `docs/` explains provenance, reproducibility, and result interpretation.

## What Is Not Included

This repository does not include the manuscript source, compiled paper PDF,
private drafting notes, audit documents, or bibliography source files. It is a
clean code and data release for reproducibility.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python scripts\download_pantheon_data.py --verify-only
pytest -q
python scripts\generate_cosmology_figures.py
```

On macOS/Linux, use forward slashes:

```bash
python -m pip install -r requirements.txt
python scripts/download_pantheon_data.py --verify-only
pytest -q
python scripts/generate_cosmology_figures.py
```

## Main Reproducible Result

The strongest observational calculation is a one-parameter flat-LambdaCDM
shape likelihood using the official Pantheon+SH0ES distance table and STAT+SYS
covariance matrix. It profiles over one additive distance-modulus offset and
finds

```text
Omega_m = 0.330 +0.019 -0.018
chi2_min = 1383.98
dof = 1570
```

This is not a new cosmological measurement and not a calibrated H0 analysis. It
is a reproducible demonstration that the FLRW distance pipeline can be coupled
to modern public supernova covariance data.

## DESI DR2 BAO Check

The repository also includes a DESI-style compressed BAO comparison using
`D_M/r_d`, `D_H/r_d`, and `D_V/r_d` with the retained DESI DR2 covariance data.
For the baseline shape, profiling one global BAO scale nuisance gives

```text
alpha_BAO = 1.016
chi2 = 10.39
dof = 12
```

This is a standard-ruler consistency check, not a full BAO cosmological fit.

## Repository Layout

```text
.
|-- data/
|-- docs/
|-- figures/
|-- scripts/
|-- src/flrw_paper/
|-- tests/
|-- requirements.txt
|-- requirements-lock.txt
|-- pyproject.toml
|-- CITATION.cff
`-- LICENSE
```

## Reproducibility Standard

The repository should be considered healthy when this command sequence passes:

```bash
python scripts/download_pantheon_data.py --verify-only
pytest -q
python scripts/generate_cosmology_figures.py
```

The GitHub Actions workflow runs the same core checks on each push and pull
request.

## License

The code in this release is distributed under the MIT License; see `LICENSE`.
The retained Pantheon+SH0ES and DESI DR2-related BAO data files remain governed
by their upstream public-release terms and should be cited through the original
data products as well as this reproducibility release. Generated figures and
tables may be reused with attribution to this repository and the underlying
public data sources. See `docs/DATA_LICENSES.md` for the data-source boundary.
