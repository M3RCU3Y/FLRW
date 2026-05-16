# Data Provenance

This project keeps the observational data boundary explicit. The paper uses the
public Pantheon+SH0ES release in two different ways:

- `data/pantheon_plus_binned.csv` is a compact derived visual anchor.
- `data/Pantheon+SH0ES.dat` and `data/Pantheon+SH0ES_STAT+SYS.cov` are the full
  source table and covariance matrix used by the covariance-aware likelihood.

The binned file is not used for the likelihood. The likelihood uses the source
distance table and the matching STAT+SYS covariance file.

The paper also uses DESI DR2 compressed BAO means and covariance from the public
CobayaSampler BAO data repository. These files are used for the DESI-style
standard-ruler consistency check, not for a full BAO cosmological fit.

## Sources

Official Pantheon+SH0ES release page:

https://pantheonplussh0es.github.io/

Official data repository directory:

https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR

Raw source files used here:

- https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/README
- https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat
- https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov
- https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt
- https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt

## Local Manifest

| File | Size | SHA-256 |
| --- | ---: | --- |
| `data/Pantheon+README` | 4,056 bytes | `e2b0d262757f01c1794a938c78d32600a21e289b2a0320e5c660c4c6fc9aa87e` |
| `data/Pantheon+SH0ES.dat` | 579,283 bytes | `1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8` |
| `data/Pantheon+SH0ES_STAT+SYS.cov` | 33,284,960 bytes | `abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc` |
| `data/pantheon_plus_binned.csv` | 968 bytes | `c2b6b8f600ede326e4f2606ebd2ee3407cdabc9d9ca5d693cf53f733542650e5` |
| `data/desi_dr2_bao_all_gccomb_mean.txt` | 472 bytes | `9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585` |
| `data/desi_dr2_bao_all_gccomb_cov.txt` | 2,547 bytes | `252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509` |

## Verification

Run:

```powershell
python scripts\download_pantheon_data.py --verify-only
```

The script verifies the retained source files by size and SHA-256 hash. Without
`--verify-only`, it downloads any missing or changed official source files and
then verifies them.
