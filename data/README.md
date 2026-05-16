# Data Directory

This directory contains the small binned visual anchor, the official
Pantheon+SH0ES source files needed for the covariance-aware supernova
likelihood, and the DESI DR2 compressed BAO means/covariance used for the
standard-ruler consistency check.

Run:

```bash
python scripts/download_pantheon_data.py --verify-only
```

to verify the retained source files by size and SHA-256 hash.

See `docs/DATA_PROVENANCE.md` for source URLs, hashes, and the data boundary.
