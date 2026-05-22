"""Model definitions, constants, and baseline cosmologies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Constants and unit conversions
C = 299_792_458.0                         # speed of light [m s^-1]
MPC = 3.0856775814913673e22               # one megaparsec [m]
GPC = 1.0e3 * MPC                         # one gigaparsec [m]
JULIAN_YEAR = 365.25 * 24 * 3600.0        # one Julian year [s]
GYR = 1.0e9 * JULIAN_YEAR                 # one gigayear [s]
H0_KM_S_MPC = 70.0                        # reference Hubble constant [km s^-1 Mpc^-1]
H0 = H0_KM_S_MPC * 1000.0 / MPC           # reference Hubble constant [s^-1]
H0_INV_GYR = 1.0 / H0 / GYR               # Hubble time [Gyr]
DH_GPC = C / H0 / GPC                     # Hubble distance [Gpc]
DH_MPC = C / H0 / MPC                     # Hubble distance [Mpc]

# Baseline concordance-like model. The small radiation term is included here and
# in the era/timeline figures so that the manuscript has one consistent baseline.
BASELINE_OMEGA_R = 9.0e-5
BASELINE_OMEGA_M = 0.3
BASELINE_OMEGA_L = 1.0 - BASELINE_OMEGA_M - BASELINE_OMEGA_R

# Numerical controls. A_MIN is used only to regularize plotting grids near the
# Big Bang; quadrature functions add the leading asymptotic contribution below it.
A_MIN = 1.0e-7
Z_MAX_DISTANCE = 6.0
Z_MAX_HUBBLE = 50.0
PANTHEON_Z_MIN = 0.01
PANTHEON_Z_MAX = 1.40
BAO_RD_MPC = 147.09
ODE_RTOL = 1.0e-10
ODE_ATOL = 1.0e-12
QUAD_EPSABS = 1.0e-11
QUAD_EPSREL = 1.0e-11

# A restrained academic palette. Colors are specified here so the LaTeX document
# can remain visually consistent across all generated figures.
PALETTE = {
    "EdS": "#24476b",
    "Radiation": "#6a4c93",
    "Flat_LCDM": "#cc5a1a",
    "Open_matter": "#2a8f86",
    "Closed_matter": "#7d2d2d",
    "Grid": "#aebfd0",
    "Ink": "#1d2733",
}

LINE_STYLES = {
    "EdS": "-",
    "Radiation": "--",
    "Flat_LCDM": "-",
    "Open_matter": "-.",
    "Closed_matter": ":",
}


@dataclass(frozen=True)
class Model:
    """A minimal FLRW model with constant Omega_Lambda and optional curvature."""

    name: str
    label: str
    omega_r: float = 0.0
    omega_m: float = 0.0
    omega_l: float = 0.0
    omega_k: float = 0.0
    color: str = "#333333"

    def e2_a(self, a: np.ndarray | float) -> np.ndarray | float:
        a = np.asarray(a)
        return (
            self.omega_r * a ** -4
            + self.omega_m * a ** -3
            + self.omega_k * a ** -2
            + self.omega_l
        )

    def e_a(self, a: np.ndarray | float) -> np.ndarray | float:
        return np.sqrt(np.maximum(self.e2_a(a), 0.0))

    def e_z(self, z: np.ndarray | float) -> np.ndarray | float:
        zp1 = 1.0 + np.asarray(z)
        e2 = (
            self.omega_r * zp1**4
            + self.omega_m * zp1**3
            + self.omega_k * zp1**2
            + self.omega_l
        )
        return np.sqrt(np.maximum(e2, 0.0))

    @property
    def has_turnaround(self) -> bool:
        # For a dust-only closed model, E(a)^2 = Omega_m a^-3 + Omega_k a^-2.
        return self.omega_k < 0.0 and self.omega_l == 0.0 and self.omega_r == 0.0

    @property
    def turnaround_a(self) -> float | None:
        if self.has_turnaround:
            return -self.omega_m / self.omega_k
        return None


MODELS: list[Model] = [
    Model("EdS", r"Einstein--de Sitter", omega_m=1.0, color=PALETTE["EdS"]),
    Model("Radiation", r"Radiation dominated", omega_r=1.0, color=PALETTE["Radiation"]),
    Model(
        "Flat_LCDM",
        r"Flat $\Lambda$CDM-like",
        omega_r=BASELINE_OMEGA_R,
        omega_m=BASELINE_OMEGA_M,
        omega_l=BASELINE_OMEGA_L,
        color=PALETTE["Flat_LCDM"],
    ),
    Model("Open_matter", r"Open matter dominated", omega_m=0.3, omega_k=0.7, color=PALETTE["Open_matter"]),
    Model("Closed_matter", r"Closed matter dominated", omega_m=1.3, omega_k=-0.3, color=PALETTE["Closed_matter"]),
]

MODEL_BY_NAME = {m.name: m for m in MODELS}

ASTROPY_CHECK_MODEL = Model(
    "Flat_LCDM_no_radiation",
    r"Flat $\Lambda$CDM, radiation omitted",
    omega_m=BASELINE_OMEGA_M,
    omega_l=1.0 - BASELINE_OMEGA_M,
    color=PALETTE["Flat_LCDM"],
)


def model_line_style(model: Model) -> str:
    """Return the model line style used for color-independent plot reading."""
    return LINE_STYLES.get(model.name, "-")


def flat_lcdm_shape_model(omega_m: float, omega_r: float = BASELINE_OMEGA_R) -> Model:
    """Flat radiation-inclusive LambdaCDM model for supernova shape fitting."""
    omega_l = 1.0 - omega_m - omega_r
    if omega_m <= 0.0 or omega_l <= 0.0:
        raise ValueError("flat_lcdm_shape_model requires positive matter and Lambda densities")
    return Model(
        "PantheonFlatLCDM",
        rf"flat $\Lambda$CDM, $\Omega_m={omega_m:.3f}$",
        omega_r=omega_r,
        omega_m=omega_m,
        omega_l=omega_l,
        omega_k=0.0,
        color=PALETTE["Flat_LCDM"],
    )


def fate_text(model: Model) -> str:
    if model.name == "Flat_LCDM":
        return "accelerating"
    if model.name == "Closed_matter":
        return "turnaround and recollapse"
    if model.name == "Open_matter":
        return "curvature-coasting"
    if model.name == "Radiation":
        return "decelerating"
    return "decelerating"
