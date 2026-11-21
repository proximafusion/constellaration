import json

import jax
import numpy as np
import pytest
import vmecpp

from constellaration.mhd import jax_turbulent_transport
from tests.equilibrium_reference_test import NAMES, REFERENCES
from tests.mhd.jax_metric_checks import (
    assert_directional_derivative,
    pinned_case,
    resolved_case,
)


@pytest.mark.parametrize("name", NAMES)
def test_flux_compression_value(name):
    with vmecpp.enable_x64(True):
        _, settings, out, arrays = pinned_case(name)
        s = settings.turbulent_settings
        assert s is not None
        args = (
            arrays,
            arrays,
            s.normalized_toroidal_flux,
            out["nfp"],
            s.n_field_lines,
            s.n_toroidal_points,
        )
        kernel = jax.jit(
            jax_turbulent_transport.flux_compression,
            static_argnums=(3, 4, 5),
            static_argnames="smoothing",
        )
        hard = kernel(*args, smoothing=0.0)
        smooth = kernel(*args)
    expected = json.loads((REFERENCES / name / "metrics.json").read_text())
    np.testing.assert_allclose(
        hard, [expected["flux_compression_in_regions_of_bad_curvature"]], atol=5e-4
    )
    # The sigmoid slightly redistributes weight near the zero-drift contour.
    np.testing.assert_allclose(smooth, hard, atol=0.003)


def test_flux_compression_gradient():
    with resolved_case() as (template, family, modes):

        def metric(t):
            _, out = family(t)
            return jax_turbulent_transport.flux_compression(
                out, modes, [0.49], template.nfp, 29, 32
            )[0]

        # At low iota the long traces amplify changes in toroidal angle.
        # These steps resolve the narrow curvature mask's transition region.
        assert_directional_derivative(metric, steps=(2e-5, 1e-5))
