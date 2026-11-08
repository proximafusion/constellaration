from types import SimpleNamespace

import numpy as np
import pytest

from constellaration import forward_model, initial_guess
from constellaration.mhd import ideal_mhd_parameters, magnetics_utils


def test_mercier_softmin():
    equilibrium = SimpleNamespace(DMerc=np.array([-100.0, 2, 3, -100]), phi=[0, 2])
    value = magnetics_utils.minimum_normalized_mercier_criterion(equilibrium, 0.1)
    assert value <= 8
    assert value == pytest.approx(8, abs=1e-10)
    equilibrium.DMerc[1] = -1
    assert magnetics_utils.minimum_normalized_mercier_criterion(equilibrium, 1e-3) < 0
    equilibrium.DMerc[1] = np.nan
    assert (
        magnetics_utils.minimum_normalized_mercier_criterion(equilibrium, 1e-3) is None
    )
    with pytest.raises(ValueError, match="temperature"):
        magnetics_utils.minimum_normalized_mercier_criterion(equilibrium, 0)
    with pytest.raises(TypeError):
        magnetics_utils.minimum_normalized_mercier_criterion(equilibrium)


def test_mirror_and_beta():
    assert magnetics_utils.mirror_ratio_well(0.1, 0.11) == pytest.approx(2 / 21)
    assert magnetics_utils.mirror_ratio_well(0, 0) is None
    equilibrium = SimpleNamespace(betatotal=0.03, beta_vol=[0, 0.09, 0.07])
    assert magnetics_utils.volume_averaged_beta(equilibrium) == pytest.approx(0.03)
    assert magnetics_utils.beta_peaking_factor(equilibrium) == pytest.approx(10 / 3)
    equilibrium.betatotal = 0
    assert magnetics_utils.volume_averaged_beta(equilibrium) == 0
    assert magnetics_utils.beta_peaking_factor(equilibrium) is None
    equilibrium.betatotal = 0.03
    equilibrium.beta_vol = [0, 0.09]
    assert magnetics_utils.beta_peaking_factor(equilibrium) is None


def test_metrics_from_real_vmec():
    boundary = initial_guess.generate_nae(
        aspect_ratio=10,
        max_elongation=5,
        rotational_transform=0.75,
        mirror_ratio=0.25,
        n_field_periods=3,
        max_poloidal_mode=1,
        max_toroidal_mode=1,
    )
    mhd = ideal_mhd_parameters.boundary_to_ideal_mhd_parameters(boundary)
    mhd.pressure.coefficients = [3000, -3000]
    settings = forward_model.ConstellarationSettings(
        qi_settings=None, boozer_preset_settings=None, turbulent_settings=None
    )
    metrics, wout = forward_model.forward_model(boundary, mhd, settings)
    assert metrics.volume_averaged_beta == pytest.approx(wout.betatotal)
    assert metrics.beta_peaking_factor == pytest.approx(
        (1.5 * wout.beta_vol[1] - 0.5 * wout.beta_vol[2]) / wout.betatotal
    )
    interior = wout.DMerc[1:-1] * wout.phi[-1] ** 2
    assert metrics.minimum_normalized_mercier_criterion <= np.min(interior)
