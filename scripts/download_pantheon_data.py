#!/usr/bin/env python3
"""Download or verify the observational source files used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve


@dataclass(frozen=True)
class DataFile:
    label: str
    local_name: str
    url: str
    sha256: str
    size: int


FILES = [
    DataFile(
        "Pantheon+ release README",
        "Pantheon+README",
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/README",
        "e2b0d262757f01c1794a938c78d32600a21e289b2a0320e5c660c4c6fc9aa87e",
        4056,
    ),
    DataFile(
        "Pantheon+SH0ES distance table",
        "Pantheon+SH0ES.dat",
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",
        "1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8",
        579283,
    ),
    DataFile(
        "Pantheon+SH0ES STAT+SYS covariance",
        "Pantheon+SH0ES_STAT+SYS.cov",
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov",
        "abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc",
        33284960,
    ),
    DataFile(
        "DESI DR2 BAO compressed means",
        "desi_dr2_bao_all_gccomb_mean.txt",
        "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt",
        "9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585",
        472,
    ),
    DataFile(
        "DESI DR2 BAO covariance",
        "desi_dr2_bao_all_gccomb_cov.txt",
        "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt",
        "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509",
        2547,
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, spec: DataFile) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        return False, f"size mismatch: expected {spec.size}, got {actual_size}"
    actual_hash = sha256(path)
    if actual_hash != spec.sha256:
        return False, f"sha256 mismatch: expected {spec.sha256}, got {actual_hash}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check local files and fail if any are missing or changed.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for spec in FILES:
        target = data_dir / spec.local_name
        ok, message = verify(target, spec)
        if ok:
            print(f"OK: {spec.label} ({spec.local_name})")
            continue
        if args.verify_only:
            failures.append(f"{spec.local_name}: {message}")
            print(f"FAIL: {spec.local_name}: {message}")
            continue

        print(f"Downloading {spec.label}...")
        urlretrieve(spec.url, target)
        ok, message = verify(target, spec)
        if ok:
            print(f"OK: {spec.local_name}")
        else:
            failures.append(f"{spec.local_name}: {message}")
            print(f"FAIL: {spec.local_name}: {message}")

    if failures:
        print("\nData verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
