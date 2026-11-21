import json

import numpy as np
import pytest
import vmecpp

from constellaration.mhd import jax_gradient_length
from tests.equilibrium_reference_test import NAMES, REFERENCES
from tests.mhd.jax_metric_checks import (
    assert_directional_derivative,
    grid,
    pinned_case,
    resolved_case,
)


@pytest.mark.parametrize("name", NAMES)
def test_gradient_length_value(name):
    with vmecpp.enable_x64(True):
        _, _, wout, arrays = pinned_case(name)
        theta, phi = grid(wout["mpol"], wout["ntor"], wout["nfp"])
        actual = (
            jax_gradient_length.normalized_scale_length(
                arrays, arrays, theta, phi
            ).min()
            * wout["nfp"]
        )
    expected = json.loads((REFERENCES / name / "metrics.json").read_text())
    np.testing.assert_allclose(
        actual, expected["minimum_normalized_magnetic_gradient_scale_length"], rtol=1e-9
    )


def test_gradient_length_derivative():
    with resolved_case() as (template, family, modes):
        theta, phi = grid(template.mpol, template.ntor, template.nfp)

        def metric(t):
            _, out = family(t)
            return (
                jax_gradient_length.normalized_scale_length(
                    out, modes, theta, phi
                ).min()
                * template.nfp
            )

        assert_directional_derivative(metric)
