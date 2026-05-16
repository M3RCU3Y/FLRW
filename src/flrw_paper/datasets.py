"""Data-loading utilities for the public FLRW release."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .models import PANTHEON_Z_MAX, PANTHEON_Z_MIN

def load_pantheon_binned(root: Path) -> dict[str, np.ndarray]:
    path = root / "data" / "pantheon_plus_binned.csv"
    records: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(row for row in handle if not row.startswith("#"))
        for row in reader:
            records.append({
                "z_mean": float(row["z_mean"]),
                "mu_mean": float(row["mu_mean"]),
                "mu_err": float(row["mu_err"]),
                "n_supernovae": float(row["n_supernovae"]),
            })
    return {key: np.array([record[key] for record in records]) for key in records[0]}


def load_desi_bao_dr2(root: Path) -> dict[str, np.ndarray]:
    """Load DESI DR2 compressed BAO means and covariance."""
    mean_path = root / "data" / "desi_dr2_bao_all_gccomb_mean.txt"
    cov_path = root / "data" / "desi_dr2_bao_all_gccomb_cov.txt"
    redshift: list[float] = []
    value: list[float] = []
    quantity: list[str] = []
    with mean_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            z_text, value_text, quantity_text = stripped.split()
            redshift.append(float(z_text))
            value.append(float(value_text))
            quantity.append(quantity_text)
    covariance = np.loadtxt(cov_path)
    if covariance.shape != (len(value), len(value)):
        raise ValueError(f"DESI BAO covariance shape {covariance.shape} does not match {len(value)} measurements")
    return {
        "z": np.asarray(redshift, dtype=float),
        "value": np.asarray(value, dtype=float),
        "quantity": np.asarray(quantity, dtype=object),
        "covariance": np.asarray(covariance, dtype=float),
    }


def load_pantheon_full_likelihood(root_str: str) -> dict[str, object]:
    """Load Pantheon+SH0ES distances and the matching STAT+SYS covariance."""
    root = Path(root_str)
    data_path = root / "data" / "Pantheon+SH0ES.dat"
    cov_path = root / "data" / "Pantheon+SH0ES_STAT+SYS.cov"

    table = np.genfromtxt(data_path, names=True, dtype=None, encoding=None)
    with cov_path.open("r", encoding="utf-8") as handle:
        n_cov = int(handle.readline().strip())
    cov_flat = np.loadtxt(cov_path, skiprows=1)
    cov = cov_flat.reshape((n_cov, n_cov))

    if len(table) != n_cov:
        raise ValueError(f"Pantheon table length {len(table)} does not match covariance size {n_cov}")

    mask = (
        (table["IS_CALIBRATOR"].astype(int) == 0)
        & (table["zHD"].astype(float) > PANTHEON_Z_MIN)
        & (table["zHD"].astype(float) < PANTHEON_Z_MAX)
    )
    idx = np.where(mask)[0]
    z = table["zHD"][mask].astype(float)
    mu = table["MU_SH0ES"][mask].astype(float)
    cov_subset = cov[np.ix_(idx, idx)]

    cfac = cho_factor(cov_subset, lower=True, check_finite=False)
    ones = np.ones_like(z)
    c_inv_ones = cho_solve(cfac, ones, check_finite=False)
    denom = float(ones @ c_inv_ones)

    return {
        "z": z,
        "mu": mu,
        "cov": cov_subset,
        "cfac": cfac,
        "ones": ones,
        "c_inv_ones": c_inv_ones,
        "denom": denom,
        "n_total": int(len(table)),
        "n_used": int(len(z)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
    }
