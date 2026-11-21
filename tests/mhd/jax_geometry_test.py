import jax.numpy as jnp
import numpy as np
import pytest
import vmecpp

from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import jax_geometry
from tests.equilibrium_reference_test import NAMES
from tests.mhd.jax_metric_checks import (
    assert_directional_derivative,
    pinned_case,
    resolved_case,
)


@pytest.mark.parametrize("name", NAMES)
def test_geometrical_metrics(name):
    with vmecpp.enable_x64(True):
        boundary, _, _, _ = pinned_case(name)
        b = jnp.stack((boundary.r_cos, boundary.z_sin))
        actual = jax_geometry.mean_cross_sectional_area(b, boundary.n_field_periods)
        expected = surface_rz_fourier.compute_mean_cross_sectional_area(boundary)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        # Refining the grid maximum of Z changes the triangle value slightly.
        actual = jax_geometry.average_triangularity(b, boundary.n_field_periods)
        expected = {
            "boundary_1": -0.5209025223,
            "boundary_2": -0.4459885210,
            "finite_beta": 0.00497512438,
        }
        np.testing.assert_allclose(actual, expected[name], rtol=1e-8)


def test_geometrical_gradients():
    with resolved_case() as (template, family, _):

        def metric(t):
            b, _ = family(t)
            return jnp.stack(
                (
                    jax_geometry.mean_cross_sectional_area(b, template.nfp),
                    jax_geometry.average_triangularity(b, template.nfp),
                )
            )

        assert_directional_derivative(metric)
