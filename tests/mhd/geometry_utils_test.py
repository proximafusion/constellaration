import numpy as np
from constellaration import initial_guess
from constellaration.mhd import geometry_utils


def test_surface_zero_triangularity() -> None:
    surface = initial_guess.generate_rotating_ellipse(
        aspect_ratio=10.0,
        elongation=3.0,
        rotational_transform=0.1,
        n_field_periods=2,
    )
    average_triangularity = geometry_utils.average_triangularity(
        surface=surface,
        n_poloidal_points=201,
    )
    np.testing.assert_allclose(
        average_triangularity,
        0.0,
        atol=5e-3,
    )
