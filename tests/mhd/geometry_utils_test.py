import jax
import numpy as np
import pytest
import vmecpp

from constellaration import initial_guess
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import geometry_utils
from tests.equilibrium_reference_test import NAMES, reference_inputs


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


@pytest.mark.parametrize("name", NAMES)
def test_boundary_geometry_gradients(name):
    surface, mhd, _ = reference_inputs(name)
    with vmecpp.enable_x64(True):
        area = surface_rz_fourier.compute_mean_cross_sectional_area
        np.testing.assert_allclose(
            area(surface), mhd.boundary_toroidal_flux, rtol=1e-12
        )
        for metric in (area, geometry_utils.average_triangularity):
            value = jax.jit(metric)
            derivative = jax.jit(jax.grad(metric))(surface)
            assert derivative.n_field_periods == surface.n_field_periods
            for step in (1e-4, 5e-5):
                r = surface.r_cos
                direction = np.zeros_like(r)
                direction[1, surface.max_toroidal_mode] = 1
                plus = surface.model_copy(update={"r_cos": r + step * direction})
                minus = surface.model_copy(update={"r_cos": r - step * direction})
                fd = (value(plus) - value(minus)) / (2 * step)
                np.testing.assert_allclose(
                    derivative.r_cos[1, surface.max_toroidal_mode],
                    fd,
                    rtol=1e-3,
                    atol=2e-7,
                )
