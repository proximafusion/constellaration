import json

import jax
import numpy as np
import pytest
import vmecpp

from constellaration.boozer import jax_transform
from constellaration.omnigeneity import jax_qi, qi
from tests.equilibrium_reference_test import NAMES, REFERENCES
from tests.mhd.jax_metric_checks import (
    assert_directional_derivative,
    pinned_case,
    resolved_case,
)


@pytest.mark.parametrize("name", NAMES)
def test_qi_value(name):
    with vmecpp.enable_x64(True):
        _, settings, wout, arrays = pinned_case(name)
        transform = jax_transform.make_transform(
            arrays,
            nfp=wout["nfp"],
            ns=wout["ns"],
            mboz=6 * wout["mpol"],
            nboz=2 * wout["ntor"] - 1,
            surfaces=[1.0],
            magnetic_only=True,
        )
        spectrum, _ = jax.jit(transform)(arrays)
        actual = jax.jit(
            lambda sp: jax_qi.residual(
                sp, wout["nfp"], settings.qi_settings, arrays["bmnc"][1, 1]
            )
        )(spectrum)
    expected = json.loads((REFERENCES / name / "metrics.json").read_text())
    # Smooth contour crossings, inverse weight floor and nested shuffling replace
    # the historical hard crossings and monotonicity repair.
    np.testing.assert_allclose(actual, expected["qi"], rtol=0.004)


def test_qi_gradient_through_equilibrium_and_boozer():
    with resolved_case() as (template, family, modes):
        transform = jax_transform.make_transform(
            modes,
            nfp=template.nfp,
            ns=int(template.ns_array[-1]),
            mboz=12,
            nboz=5,
            surfaces=[1.0],
            magnetic_only=True,
        )
        settings = qi.QISettings(
            n_field_lines=19,
            n_toroidal_points=161,
            n_field_contours=101,
            n_output_toroidal_points=191,
        )

        def metric(t):
            _, out = family(t)
            spectrum, _ = transform(out)
            return jax_qi.residual(spectrum, template.nfp, settings, out["bmnc"][1, 1])

        # Resolve the derivative locally without stepping across contour knots.
        assert_directional_derivative(metric, steps=(5e-5, 2e-5))
