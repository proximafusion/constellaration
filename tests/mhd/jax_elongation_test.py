import json

import numpy as np
import pytest
import vmecpp

from constellaration.mhd import jax_elongation
from tests.equilibrium_reference_test import NAMES, REFERENCES
from tests.mhd.jax_metric_checks import (
    assert_directional_derivative,
    grid,
    pinned_case,
    resolved_case,
)


@pytest.mark.parametrize("name", NAMES)
def test_elongation_value(name):
    with vmecpp.enable_x64(True):
        _, _, wout, arrays = pinned_case(name)
        theta, _ = grid(wout["mpol"], wout["ntor"], wout["nfp"])
        actual = jax_elongation.max_elongation(
            arrays, arrays["xm"], arrays["xn"], wout["nfp"], *theta.shape
        )
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
            return jax_elongation.max_elongation(
                out, modes["xm"], modes["xn"], template.nfp, *theta.shape
            )

        assert_directional_derivative(metric)
