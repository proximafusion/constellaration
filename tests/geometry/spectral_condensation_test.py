import json
import pathlib

import numpy as np
import pytest

from constellaration.geometry import surface_rz_fourier, surface_utils
from constellaration.geometry.spectral_condensation import (
    SpectralCondensationSettings,
    _create_normal_distance_constraint,
    spectrally_condense_surface,
)

TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent / "test_data"

# Reference shapes from Hirshman 1985:
# "Optimized Fourier representations for three-dimensional magnetic surfaces."
# The Physics of fluids 28.5 (1985): 1387-1391.
HIRSHMAN_1985: list[surface_rz_fourier.SurfaceRZFourier] = [
    # Square
    surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array([3.0, 0.427, 0.0, 0.07322]).reshape(-1, 1),
        z_sin=np.array([0.0, 0.427, 0.0, -0.07322]).reshape(-1, 1),
        n_field_periods=1,
        is_stellarator_symmetric=True,
    ),
    # Belt pinch
    surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array([3.0, 0.453, 0.0, 0.0]).reshape(-1, 1),
        z_sin=np.array([0.0, 0.60, 0.0, 0.196]).reshape(-1, 1),
        n_field_periods=1,
        is_stellarator_symmetric=True,
    ),
    # D-Shape
    surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array([3.0, 0.991, 0.136]).reshape(-1, 1),
        z_sin=np.array([0.0, 1.409, -0.118]).reshape(-1, 1),
        n_field_periods=1,
        is_stellarator_symmetric=True,
    ),
    # Heliac
    surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array(
            [
                [0.0, 0.0, 0.0, 4.115, 0.475, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.323, -0.0621, 0.136, -0.0415],
                [0.0, 0.0, 0.0, 0.0136, 0.0806, -0.0205, 0.0445],
            ]
        ),
        z_sin=np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -0.505, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.337, 0.0836, 0.133, -0.0409],
                [0.0, 0.0, 0.0, 0.0155, 0.0619, -0.0382, -0.0237],
            ]
        ),
        n_field_periods=1,
        is_stellarator_symmetric=True,
    ),
]

rng = np.random.default_rng(42)

RANDOMLY_GENERATED_SURFACES: list[surface_rz_fourier.SurfaceRZFourier] = []
for _ in range(3):
    _nfp = int(rng.integers(1, 5))
    _mpol = int(rng.integers(2, 8))
    _ntor = 0
    _scale = np.arange(0, _mpol + 1)[:, None] ** 3
    _scale[0] = 1.0
    _r_cos = rng.standard_normal((_mpol + 1, 2 * _ntor + 1)) / _scale
    _r_cos[0, :_ntor] = 0.0
    _r_cos[0, _ntor] = 10.0
    _z_sin = rng.standard_normal((_mpol + 1, 2 * _ntor + 1)) / _scale
    _z_sin[0, : _ntor + 1] = 0.0
    RANDOMLY_GENERATED_SURFACES.append(
        surface_rz_fourier.SurfaceRZFourier(
            r_cos=_r_cos,
            z_sin=_z_sin,
            n_field_periods=_nfp,
            is_stellarator_symmetric=True,
        )
    )

SURFACES = HIRSHMAN_1985 + RANDOMLY_GENERATED_SURFACES


@pytest.mark.parametrize("surface", SURFACES)
def test_spectrally_condense_surface_lowers_spectral_width(
    surface: surface_rz_fourier.SurfaceRZFourier,
) -> None:
    settings = SpectralCondensationSettings(
        p=1,
        q=1,
        normalize=False,
        maximum_normal_displacement=1e-3,
    )
    condensed = spectrally_condense_surface(surface, settings)

    init_width = float(
        surface_rz_fourier.spectral_width(
            [surface.r_cos, surface.z_sin], p=1, q=1, normalize=False
        )
    )
    final_width = float(
        surface_rz_fourier.spectral_width(
            [condensed.r_cos, condensed.z_sin], p=1, q=1, normalize=False
        )
    )
    rtol = 1e-3
    assert init_width > final_width * (1 + rtol)


def _load_boundary(filename: str) -> surface_rz_fourier.SurfaceRZFourier:
    with (TEST_DATA_DIR / filename).open() as f:
        data = json.load(f)
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array(data["r_cos"]),
        z_sin=np.array(data["z_sin"]),
        n_field_periods=data["n_field_periods"],
        is_stellarator_symmetric=data["is_stellarator_symmetric"],
    )


@pytest.mark.parametrize("filename", ["test_boundary_1.json", "test_boundary_2.json"])
def test_spectrally_condense_3d_stellarator_boundary(filename: str) -> None:
    """Spectral condensation on realistic 3D stellarator boundaries.

    Checks that the spectral width is reduced by at least 10% and that the
    normal displacement constraint is satisfied.
    """
    surface = _load_boundary(filename)
    max_disp = 1e-3
    settings = SpectralCondensationSettings(
        p=4,
        q=1,
        normalize=False,
        maximum_normal_displacement=max_disp,
        n_restarts=0,
    )

    condensed = spectrally_condense_surface(surface, settings)

    init_width = float(
        surface_rz_fourier.spectral_width(
            [surface.r_cos, surface.z_sin], p=4, q=1, normalize=False
        )
    )
    final_width = float(
        surface_rz_fourier.spectral_width(
            [condensed.r_cos, condensed.z_sin], p=4, q=1, normalize=False
        )
    )
    assert (
        final_width < init_width * 0.9
    ), f"Expected >= 10% reduction, got {(1 - final_width / init_width) * 100:.1f}%"

    (
        n_pol,
        n_tor,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        surface.n_poloidal_modes, surface.max_toroidal_mode
    )
    constraint_fn = _create_normal_distance_constraint(surface, n_pol, n_tor)
    violation = np.max(np.abs(constraint_fn(condensed)))
    assert (
        violation <= max_disp * 1.01
    ), f"Constraint violated: {violation:.3e} > {max_disp:.3e}"
