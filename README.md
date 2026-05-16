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

- `scripts/generate_cosmology_figures.py` generates all numerical figures and
  LaTeX-ready result tables.
- `scripts/download_pantheon_data.py` verifies or downloads the official
  Pantheon+SH0ES source files by size and SHA-256 hash.
- `tests/test_cosmology_core.py` checks analytic limits, distance identities,
  Astropy agreement, convergence, and the Pantheon+ covariance likelihood.
- `data/` contains the retained data inputs used by the simulations.
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

## Repository Layout

```text
.
|-- data/
|-- docs/
|-- figures/
|-- scripts/
|-- tests/
|-- requirements.txt
|-- requirements-lock.txt
|-- pyproject.toml
`-- CITATION.cff
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

No open-source license has been selected yet. Until a license is added, reuse is
limited to what GitHub's terms and applicable law allow.
