# Results Summary

This repository reproduces the numerical backbone of a background FLRW
cosmology paper.

## Baseline Model

The baseline model is a radiation-inclusive flat LambdaCDM-like model with:

```text
Omega_r = 9.0e-5
Omega_m = 0.300
Omega_Lambda = 0.69991
H0 = 70 km s^-1 Mpc^-1
```

Selected generated diagnostics:

```text
Hubble time = 13.968 Gyr
Hubble distance = 4.283 Gpc
baseline age = 13.46 Gyr
matter-radiation equality = z ~ 3332
acceleration transition = z ~ 0.67
maximum angular-diameter distance = z ~ 1.61
```

## Pantheon+SH0ES Shape Likelihood

The strongest observational reproducibility check uses:

- official `Pantheon+SH0ES.dat`,
- official `Pantheon+SH0ES_STAT+SYS.cov`,
- 1572 non-calibrator Hubble-flow supernovae,
- full covariance subset,
- flat radiation-inclusive LambdaCDM shape,
- one analytically profiled additive distance-modulus offset.

Generated result:

```text
Omega_m = 0.330 +0.019 -0.018
Delta_mu_0 = -0.099 mag
chi2_min = 1383.98
dof = 1570
reduced_chi2 = 0.882
```

This is an internal reproducibility result, not a new calibrated cosmological
measurement.

## DESI DR2 BAO Compressed-Distance Check

The BAO check uses retained DESI DR2 compressed measurements and covariance for:

- `D_V / r_d`,
- `D_M / r_d`,
- `D_H / r_d`.

The baseline prediction uses `r_d = 147.09 Mpc` as a reference scale and then
profiles one global multiplicative distance-scale nuisance.

Generated result:

```text
alpha_BAO = 1.016
fixed-scale chi2 = 43.40 for 13 data-vector entries
profiled-scale chi2 = 10.39 for 12 degrees of freedom
```

This is a standard-ruler consistency check, not a joint BAO cosmological
parameter fit.

## Why The Code Matters

The repository demonstrates that the same compact expansion function, `E(a)`,
can be used consistently to generate ages, distances, horizons, validation
checks, a covariance-weighted supernova shape likelihood, and a DESI-style BAO
standard-ruler comparison. That consistency is the core scientific point.
