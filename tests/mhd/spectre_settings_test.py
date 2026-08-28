import pydantic
import pytest

from constellaration.mhd import spectre_settings


def test_defaults_are_the_commissioned_rule() -> None:
    s = spectre_settings.spectre_settings_metrics()
    assert s.tolerance_pct == 1.0
    assert s.max_poloidal_order == 40
    assert s.poloidal_floor == 14
    assert s.toroidal_ladder == (14, 18, 22, 26)
    assert s.stop_residual == pytest.approx(0.02)
    assert s.poincare is None
    assert s.max_threads == 1


@pytest.mark.parametrize(
    ("m", "mpol", "lrad"),
    [(2, 14, 18), (5, 14, 18), (9, 14, 18), (10, 15, 19), (13, 20, 24), (27, 41, 45)],
)
def test_resolution_rule(m: int, mpol: int, lrad: int) -> None:
    s = spectre_settings.SpectreSettings()
    assert s.poloidal_modes(m) == mpol
    assert s.radial_modes(m) == lrad


def test_radial_resolution_override() -> None:
    s = spectre_settings.SpectreSettings(radial_resolution=30)
    assert s.radial_modes(5) == 30


def test_stop_residual_follows_tolerance() -> None:
    assert spectre_settings.SpectreSettings(tolerance_pct=5.0).stop_residual == 0.1


@pytest.mark.parametrize("bad", [(18, 14), (14, 14, 18), ()])
def test_ladder_must_increase(bad: tuple[int, ...]) -> None:
    with pytest.raises(pydantic.ValidationError):
        spectre_settings.SpectreSettings(toroidal_ladder=bad)


@pytest.mark.parametrize("kwargs", [{"tolerance_pct": 0.0}, {"max_poloidal_order": 0}])
def test_validation_floors(kwargs: dict) -> None:
    with pytest.raises(pydantic.ValidationError):
        spectre_settings.SpectreSettings(**kwargs)


def test_round_trip() -> None:
    s = spectre_settings.SpectreSettings(
        tolerance_pct=2.5,
        poincare=spectre_settings.PoincareSettings(n_trajectories=10),
    )
    assert spectre_settings.SpectreSettings.model_validate(s.model_dump()) == s
