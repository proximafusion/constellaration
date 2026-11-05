import pydantic
import pytest

from constellaration.mhd import spectre_settings


def test_defaults_are_the_commissioned_rule() -> None:
    s = spectre_settings.spectre_settings_metrics()
    assert s.tolerance_pct == 1.0
    assert s.max_poloidal_order == 20  # m <= 20 solved
    assert s.max_rationals == 3
    assert s.poloidal_floor == 14
    assert s.toroidal_ladder == (14, 18, 22, 26)
    assert s.stop_residual == pytest.approx(0.02)
    assert s.max_threads == 1


@pytest.mark.parametrize(
    ("m", "mpol"),
    [(2, 14), (5, 14), (9, 14), (10, 15), (13, 20), (27, 41)],
)
def test_poloidal_resolution_rule(m: int, mpol: int) -> None:
    assert spectre_settings.SpectreSettings().poloidal_modes(m) == mpol


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
    s = spectre_settings.SpectreSettings(tolerance_pct=2.5, max_rationals=2)
    assert spectre_settings.SpectreSettings.model_validate(s.model_dump()) == s


def test_max_rationals_is_at_least_one_or_unbounded() -> None:
    assert spectre_settings.SpectreSettings(max_rationals=None).max_rationals is None
    with pytest.raises(pydantic.ValidationError):
        spectre_settings.SpectreSettings(max_rationals=0)


def test_only_one_volume_is_supported() -> None:
    with pytest.raises(pydantic.ValidationError):
        spectre_settings.SpectreSettings(n_volumes=2)  # type: ignore[arg-type]
