# Data Provenance

This repository retains the data needed to reproduce the numerical figures,
tables, and Pantheon+SH0ES shape-likelihood check.

## Data Boundary

The project uses Pantheon+SH0ES in two distinct ways:

- `data/pantheon_plus_binned.csv` is a compact visual anchor for the distance
  modulus plot.
- `data/Pantheon+SH0ES.dat` and `data/Pantheon+SH0ES_STAT+SYS.cov` are the
  official source distance table and covariance matrix used by the
  covariance-aware likelihood.

The binned file is not used for the covariance likelihood.

## Official Sources

Official Pantheon+SH0ES release page:

https://pantheonplussh0es.github.io/

Official data repository directory:

https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR

Raw source files:

- https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/README
- https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat
- https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov

## Local Manifest

| File | Size | SHA-256 |
| --- | ---: | --- |
| `data/Pantheon+README` | 4,056 bytes | `e2b0d262757f01c1794a938c78d32600a21e289b2a0320e5c660c4c6fc9aa87e` |
| `data/Pantheon+SH0ES.dat` | 579,283 bytes | `1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8` |
| `data/Pantheon+SH0ES_STAT+SYS.cov` | 33,284,960 bytes | `abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc` |
| `data/pantheon_plus_binned.csv` | 968 bytes | `c2b6b8f600ede326e4f2606ebd2ee3407cdabc9d9ca5d693cf53f733542650e5` |

## Verification

```bash
python scripts/download_pantheon_data.py --verify-only
```

Without `--verify-only`, the script downloads missing or changed official
source files and verifies them again.
