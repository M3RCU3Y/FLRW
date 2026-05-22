# Reproducibility Guide

This repository is designed so the numerical results can be regenerated without
the manuscript source.

## Environment

Use Python 3.11 or newer. Install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements-lock.txt` records the exact package versions used in the local
verification environment.

## Verification Commands

Run the full local check:

```bash
python scripts/download_pantheon_data.py --verify-only
pytest -q
python scripts/generate_cosmology_figures.py
```

Expected test result:

```text
12 passed
```

## Generated Outputs

The generation script writes vector PDF figures and generated LaTeX tables into
`figures/`. The files are committed so the repository is useful immediately,
but they are not hand-edited artifacts; they can be regenerated from the Python
code.

## Numerical Scope

The code computes background FLRW quantities:

- scale-factor histories,
- lookback time,
- line-of-sight and transverse comoving distance,
- luminosity distance,
- angular-diameter distance,
- particle horizon,
- density-era diagnostics,
- Pantheon+SH0ES binned distance-modulus anchor,
- Pantheon+SH0ES covariance-aware shape likelihood.
- DESI DR2 compressed BAO consistency check.

It does not compute perturbations, CMB spectra, galaxy clustering, weak
lensing, or a full multi-probe cosmological parameter fit.

## Data Scope

The Pantheon+SH0ES source table/covariance and DESI DR2 compressed BAO
mean/covariance files are retained in `data/` for reproducibility. Their
provenance and hashes are documented in `docs/DATA_PROVENANCE.md`.
Upstream data-source citation and redistribution notes are summarized in
`docs/DATA_LICENSES.md`.
