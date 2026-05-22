# Data Source And Redistribution Notes

This repository retains small public data products so that the numerical checks
can be reproduced without relying on hand-created intermediate files.

## Pantheon+SH0ES

Retained files:

- `data/Pantheon+README`
- `data/Pantheon+SH0ES.dat`
- `data/Pantheon+SH0ES_STAT+SYS.cov`
- `data/pantheon_plus_binned.csv`

Source:

- https://pantheonplussh0es.github.io/
- https://github.com/PantheonPlusSH0ES/DataRelease

Use in this repository:

- The full table and covariance are used for a covariance-aware, shape-only
  flat-LambdaCDM validation exercise.
- The compact binned file is a derived visual anchor and is not used for the
  full-covariance likelihood.

Citation expectation:

- Cite the Pantheon+SH0ES publications and public release in any reuse of these
  data products or generated outputs.

## DESI DR2-Related Compressed BAO Data

Retained files:

- `data/desi_dr2_bao_all_gccomb_mean.txt`
- `data/desi_dr2_bao_all_gccomb_cov.txt`

Source:

- https://github.com/CobayaSampler/bao_data

Use in this repository:

- These files are used for a compressed-distance BAO consistency check with one
  profiled global scale nuisance. They are not used for a full DESI BAO
  cosmological analysis.

Citation expectation:

- Cite the DESI BAO publications and the CobayaSampler BAO data repository when
  reusing these files or generated BAO outputs.

## Repository License Boundary

The Python code and repository-specific documentation are distributed under the
MIT License in `LICENSE`. The retained third-party data files remain governed
by their upstream public-release terms and citation requirements.
