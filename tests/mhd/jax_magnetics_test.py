import json

import jax.numpy as jnp
import numpy as np
import pytest
import vmecpp

from constellaration.mhd import jax_magnetics
from tests.equilibrium_reference_test import NAMES, REFERENCES
from tests.mhd.jax_metric_checks import (
    assert_directional_derivative,
    grid,
    pinned_case,
    resolved_case,
)


def values(out, modes, theta, phi):
    axis, edge = jax_magnetics.mirror_ratios(out, modes, theta, phi)
    return jnp.stack(
        (
            axis,
            edge,
            jax_magnetics.vacuum_well(out),
            jax_magnetics.mercier_softmin(out, 1e-3),
        )
    )


@pytest.mark.parametrize("name", NAMES)
def test_magnetic_values(name):
    with vmecpp.enable_x64(True):
        _, _, wout, arrays = pinned_case(name)
        theta, phi = grid(wout["mpol"], wout["ntor"], wout["nfp"])
        actual = values(arrays, arrays, theta, phi)
    expected = json.loads((REFERENCES / name / "metrics.json").read_text())
    np.testing.assert_allclose(
        actual,
        [
            expected[key]
            for key in (
                "axis_magnetic_mirror_ratio",
                "edge_magnetic_mirror_ratio",
                "vacuum_well",
                "minimum_normalized_mercier_criterion",
            )
        ],
        rtol=1e-10,
        atol=1e-11,
    )


def test_magnetic_gradients():
    with resolved_case() as (template, family, modes):
        theta, phi = grid(template.mpol, template.ntor, template.nfp)

        def metric(t):
            _, out = family(t)
            return values(out, modes, theta, phi)

        assert_directional_derivative(metric)
