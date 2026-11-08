import numpy as np
import pytest

from constellaration.mhd import flux_power_series, pressure_profile


@pytest.mark.parametrize(
    "logits", [(), (2,), (1, -1), (0, 0, 0), (3, -2, 1), (-5, 4, 1), (1, -2, 3, 0, 2)]
)
def test_pressure_profile(logits):
    profile = pressure_profile.IntegratedBernsteinProfile(
        axis_pressure=10000, gradient_logits=logits
    )
    power = pressure_profile.to_power_series(profile)
    s = np.linspace(0, 1, 1001)
    p = flux_power_series.evaluate_at_normalized_toroidal_flux(power, s)
    assert p[0] == pytest.approx(10000)
    assert p[-1] == pytest.approx(0, abs=1e-9)
    assert np.min(p) >= -1e-9
    assert np.max(np.diff(p)) <= 1e-9
    assert profile == profile.model_validate_json(profile.model_dump_json())
    assert len(power.coefficients) == len(logits) + 2


def test_power_series_matches_bernstein_gradient():
    from math import comb

    logits = (1, -2, 0.5, 2)
    profile = pressure_profile.IntegratedBernsteinProfile(
        axis_pressure=7, gradient_logits=logits
    )
    x = np.linspace(0, 1, 101)
    weights = np.exp([*logits, 0])
    weights /= weights.sum()
    n = len(logits)
    expected = (
        -7
        * (n + 1)
        * sum(weights[k] * comb(n, k) * x**k * (1 - x) ** (n - k) for k in range(n + 1))
    )
    derivative = flux_power_series.evaluate_derivative(
        pressure_profile.to_power_series(profile)
    )
    np.testing.assert_allclose(
        flux_power_series.evaluate_at_normalized_toroidal_flux(derivative, x),
        expected,
        atol=1e-11,
    )


def test_shape_and_scale_are_independent():
    base = pressure_profile.IntegratedBernsteinProfile(axis_pressure=1)
    peaked = pressure_profile.IntegratedBernsteinProfile(
        axis_pressure=1, gradient_logits=(8, -8, -8)
    )
    assert (
        flux_power_series.evaluate_volume_average(
            pressure_profile.to_power_series(peaked)
        )
        < 1 / 3
    )
    assert pressure_profile.to_power_series(base) != pressure_profile.to_power_series(
        peaked
    )
    scaled = pressure_profile.IntegratedBernsteinProfile(axis_pressure=2)
    assert pressure_profile.to_power_series(scaled).coefficients[0] == 2
