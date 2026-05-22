"""Render selected figure PDFs into PNG previews for the GitHub README."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"

FIGURES = {
    "flrw_expansion_diagram": "expansion-diagram",
    "scale_factor_evolution": "scale-factor-evolution",
    "luminosity_distance": "luminosity-distance",
    "pantheon_likelihood_profile": "pantheon-likelihood-profile",
    "desi_bao_dr2_comparison": "desi-bao-comparison",
    "parameter_heatmap": "parameter-heatmap",
    "cosmic_history_timeline": "cosmic-history-timeline",
}


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for source_stem, asset_stem in FIGURES.items():
        source = ROOT / "figures" / f"{source_stem}.pdf"
        target_prefix = ASSETS / asset_stem
        if not source.exists():
            raise FileNotFoundError(source)
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-r",
                "180",
                str(source),
                str(target_prefix),
            ],
            check=True,
        )
        print(f"wrote {target_prefix.with_suffix('.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

