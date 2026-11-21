import json

import jax
import numpy as np
import pytest
import vmecpp

from constellaration import initial_guess
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import geometry_utils, vmec_utils
from tests.equilibrium_reference_test import NAMES, REFERENCES, reference_inputs
from tests.utils.vmec_utils_test import (
    assert_directional_derivative,
    grid,
    pinned_case,
    resolved_case,
)


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


@pytest.mark.parametrize("name", NAMES)
def test_elongation_value(name):
    with vmecpp.enable_x64(True):
        wout = vmec_utils.VmecppWOut.model_construct(**pinned_case(name)[2])
        theta, _ = grid(wout.mpol, wout.ntor, wout.nfp)
        actual = geometry_utils.max_elongation(wout, *theta.shape)
    expected = json.loads((REFERENCES / name / "metrics.json").read_text())
    # Continuous perimeter and area quadrature replaces the legacy polygon.
    np.testing.assert_allclose(actual, expected["max_elongation"], rtol=0.045)
    np.testing.assert_allclose(
        actual,
        {"boundary_1": 6.76411628, "boundary_2": 6.71931483, "finite_beta": 2.86350131}[
            name
        ],
        rtol=1e-7,
    )


def test_elongation_gradient():
    with resolved_case() as (template, family, modes):
        theta, _ = grid(template.mpol, template.ntor, template.nfp)

        def metric(t):
            _, out = family(t)
            return geometry_utils.max_elongation(
                vmec_utils.VmecppWOut.model_construct(
                    _fields_set=None, **out, **modes, nfp=template.nfp, lasym=False
                ),
                *theta.shape,
            )

        assert_directional_derivative(metric)
