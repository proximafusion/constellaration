"""Pinned legacy VMEC++, BOOZ_XFORM and benchmark metrics.

The legacy files were generated on the fixture PR (#148) with VMEC++ 0.5.4
and booz-xform 0.0.9: python -m tests.equilibrium_reference_test
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from constellaration import forward_model, initial_guess
from constellaration.boozer import boozer
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import ideal_mhd_parameters, vmec_settings, vmec_utils

REFERENCES = Path(__file__).parent / "metrics_references"
NAMES = ("boundary_1", "boundary_2", "finite_beta")


def reference_inputs(name):
    root = REFERENCES / name
    settings = json.loads((root / "settings.json").read_text())
    settings["turbulent_settings"]["normalized_toroidal_flux"] = np.array(
        settings["turbulent_settings"]["normalized_toroidal_flux"]
    )
    return (
        surface_rz_fourier.SurfaceRZFourier.model_validate_json(
            (root / "boundary.json").read_text()
        ),
        ideal_mhd_parameters.IdealMHDParameters.model_validate_json(
            (root / "mhd.json").read_text()
        ),
        forward_model.ConstellarationSettings.model_validate(settings),
    )


def regenerate():
    source = Path(__file__).parent / "geometry" / "test_data"
    for name in NAMES:
        root = REFERENCES / name
        root.mkdir(parents=True, exist_ok=True)
        if name == "finite_beta":
            # Seed from tests/mhd/finite_beta_metrics_test.py.
            boundary = initial_guess.generate_nae(
                aspect_ratio=10,
                max_elongation=5,
                rotational_transform=0.75,
                mirror_ratio=0.25,
                n_field_periods=3,
                max_poloidal_mode=1,
                max_toroidal_mode=1,
            )
        else:
            boundary = surface_rz_fourier.SurfaceRZFourier.model_validate_json(
                (source / f"test_{name}.json").read_text()
            )
        mhd = ideal_mhd_parameters.boundary_to_ideal_mhd_parameters(boundary)
        if name == "finite_beta":
            mhd.pressure.coefficients = [3000, -3000]
        settings = forward_model.ConstellarationSettings(
            vmec_preset_settings=vmec_settings.VmecPresetSettings(
                fidelity="from_boundary_resolution"
            ),
        )
        metrics, wout = forward_model.forward_model(boundary, mhd, settings)
        assert settings.boozer_preset_settings is not None
        transform = boozer.run_boozer(
            wout,
            boozer.create_boozer_settings_from_equilibrium_resolution(
                wout, settings.boozer_preset_settings
            ),
        )
        for key, value in (
            ("boundary", boundary),
            ("mhd", mhd),
            ("settings", settings),
            ("metrics", metrics),
        ):
            (root / f"{key}.json").write_text(
                json.dumps(value.model_dump(), indent=4, default=lambda x: x.tolist())
                + "\n"
            )
        wout.save(root / "wout.nc")
        spectra: dict[str, Any] = {
            key: value
            for key, value in transform
            if isinstance(value, np.ndarray) and key.endswith("_b")
        }
        np.savez_compressed(root / "boozer.npz", **spectra)


@pytest.mark.parametrize("name", NAMES)
def test_reference_metrics(name):
    root = REFERENCES / name
    expected = forward_model.ConstellarationMetrics.model_validate_json(
        (root / "metrics.json").read_text()
    )
    _, _, settings = reference_inputs(name)
    wout = vmec_utils.vmecppwout_from_wout(
        vmec_utils.VmecppWOut.from_wout_file(root / "wout.nc")
    )
    actual = forward_model.metrics_from_equilibrium(wout, settings)
    actual_values, expected_values = actual.model_dump(), expected.model_dump()
    # The outer-surface poloidal origin can shift, and the height maximum is
    # now refined between quadrature points rather than rounded to the grid.
    key = "average_triangularity"
    assert actual_values.pop(key) == pytest.approx(expected_values.pop(key), abs=0.03)
    # Continuous section quadrature replaces the legacy polygon fit.
    key = "max_elongation"
    assert actual_values.pop(key) == pytest.approx(expected_values.pop(key), rel=0.045)
    assert actual_values == pytest.approx(expected_values, rel=3e-5, abs=1e-8)


if __name__ == "__main__":
    regenerate()
