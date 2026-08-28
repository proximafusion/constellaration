import sys

import numpy as np
import pytest

from constellaration.mhd import spectre, spectre_settings


def _profile(lo: float, hi: float, n: int = 99) -> tuple[np.ndarray, np.ndarray]:
    psi = np.linspace(0.0, 1.0, n)
    return lo + (hi - lo) * psi, psi


def test_imports_without_spectre() -> None:
    assert "spectre" not in sys.modules


@pytest.mark.parametrize(
    ("n", "m", "nfp", "ok"),
    [
        (2, 4, 2, True),  # iota = 1/2 in a two-period field: four islands
        (1, 2, 2, False),  # n must be a multiple of nfp
        (4, 8, 2, False),  # reducible to (2, 4) within the symmetry
        (3, 3, 3, True),  # iota = 1 in a three-period field: three islands
        (6, 5, 3, True),
        (12, 10, 3, False),
        (28, 36, 4, True),  # iota = 7/9 in a four-period field
        (4, 6, 4, True),
        (8, 12, 4, False),
    ],
)
def test_admissible(n: int, m: int, nfp: int, ok: bool) -> None:
    assert spectre.admissible(n, m, nfp) is ok


@pytest.mark.parametrize(
    ("lo", "hi", "nfp", "expected"),
    [
        (0.46, 0.51, 2, (2, 4)),
        (0.95, 1.05, 3, (3, 3)),
        (1.15, 1.25, 3, (6, 5)),
        (0.777, 0.788, 4, (28, 36)),
    ],
)
def test_lowest_order_resonance(
    lo: float, hi: float, nfp: int, expected: tuple[int, int]
) -> None:
    iota, psi = _profile(lo, hi)
    res = spectre.lowest_order_resonance(iota, psi, nfp, max_poloidal_order=40)
    assert res is not None
    assert (res.n, res.m) == expected
    assert res.n_crossings == 1
    assert res.shear == pytest.approx(hi - lo, rel=1e-6)


def test_screen_no_rational() -> None:
    iota, psi = _profile(0.2199, 0.2217)  # no j/m with m <= 40 at nfp = 1
    refusal, res = spectre.screen(iota, psi, 1, spectre_settings.SpectreSettings())
    assert refusal is spectre.RefusalClass.NO_RATIONAL
    assert res is None


def test_screen_sign_indefinite() -> None:
    iota, psi = _profile(-0.003, 0.002)
    refusal, res = spectre.screen(iota, psi, 4, spectre_settings.SpectreSettings())
    assert refusal is spectre.RefusalClass.SIGN_INDEFINITE
    assert res is None


def test_screen_horizon_is_a_setting() -> None:
    iota, psi = _profile(0.777, 0.788)
    s = spectre_settings.SpectreSettings(max_poloidal_order=20)
    assert spectre.screen(iota, psi, 4, s)[0] is spectre.RefusalClass.NO_RATIONAL


def _ladder(residuals: list[float]) -> tuple[list[float], bool]:
    it = iter(residuals)

    def solve(_ntor: int) -> float:
        return next(it)

    return spectre.ladder(solve, spectre_settings.SpectreSettings())


def test_ladder_stops_at_first_converged_rung() -> None:
    assert _ladder([2.5e-4]) == ([2.5e-4], False)


def test_ladder_climbs_then_stops() -> None:
    assert _ladder([0.05, 0.01, 1e-3]) == ([0.05, 0.01], False)


def test_ladder_exhausted_is_not_a_refusal() -> None:
    assert _ladder([0.5, 0.3, 0.2, 0.1]) == ([0.5, 0.3, 0.2, 0.1], False)


def test_ladder_diverged_when_not_falling() -> None:
    assert _ladder([26.0, 26.0]) == ([26.0, 26.0], True)


def test_ladder_first_rung_above_five_gets_one_more_rung() -> None:
    assert _ladder([30.0, 7.5, 1.9, 0.47]) == ([30.0, 7.5, 1.9, 0.47], False)


def test_ladder_diverged_at_the_top_rung() -> None:
    assert _ladder([30.0, 20.0, 10.0, 6.0]) == ([30.0, 20.0, 10.0, 6.0], True)


def _output(**kwargs) -> spectre.SpectreOutput:
    base = dict(
        settings=spectre_settings.SpectreSettings(),
        n_field_periods=3,
        resonance=spectre.Resonance(n=6, m=5, n_crossings=1, shear=0.2752),
        beltrami_residual=2.48e-4,
    )
    base.update(kwargs)
    return spectre.SpectreOutput(**base)


def test_metrics_single_chain_inside_the_domain() -> None:
    chain = spectre.IslandChain(
        n=6, m=5, psi_n=0.64, residue_o=3.465e-3, residue_x=-3.465e-3, shear=0.2752
    )
    m = spectre.compute_field_integrity_metrics(_output(chains=[chain]))
    expected = 3 * (3.465e-3**2) ** 0.25 / (25 * 0.2752)
    assert m.severity == pytest.approx(expected)
    assert m.severity_poloidal_mode == 5
    assert m.pendulum_domain is True
    assert m.residue_ratio == pytest.approx(1.0)
    assert m.trust_pct == pytest.approx(0.41 + 100 * 1.97 * 2.48e-4)
    assert m.refusal_class is spectre.RefusalClass.NONE
    assert m.metrics_version == spectre.METRICS_VERSION
    assert spectre.predicted_flux_fraction(m.severity) == pytest.approx(
        0.825 * expected, rel=1e-2
    )


def test_metrics_sum_over_distinct_chains() -> None:
    a = spectre.IslandChain(
        n=3, m=8, psi_n=0.07, residue_o=1.46e-3, residue_x=-1.4e-3, shear=0.02235
    )
    b = spectre.IslandChain(
        n=3, m=8, psi_n=0.67, residue_o=1.8e-7, residue_x=-1.8e-7, shear=0.02235
    )
    out = _output(
        n_field_periods=3,
        resonance=spectre.Resonance(n=3, m=8, n_crossings=2, shear=0.02235),
        chains=[a, b],
    )
    m = spectre.compute_field_integrity_metrics(out, excursion_psi=3e-2 / 0.02235)
    single = spectre.compute_field_integrity_metrics(
        _output(n_field_periods=3, resonance=out.resonance, chains=[a])
    )
    assert m.severity > single.severity
    assert m.n_chains_found == 2
    assert m.merged is False


def test_metrics_merged_withholds_the_sum() -> None:
    a = spectre.IslandChain(
        n=3, m=4, psi_n=0.3, residue_o=3e-2, residue_x=-3e-2, shear=0.02236
    )
    b = spectre.IslandChain(
        n=3, m=4, psi_n=0.35, residue_o=4e-3, residue_x=-4e-3, shear=0.02236
    )
    out = _output(
        resonance=spectre.Resonance(n=3, m=4, n_crossings=3, shear=0.02236),
        chains=[a, b],
    )
    m = spectre.compute_field_integrity_metrics(out, excursion_psi=1e-3 / 0.02236)
    assert m.severity is None
    assert m.merged is True
    assert m.refusal_class is spectre.RefusalClass.NONTWIST_MERGED


def test_metrics_outside_the_domain_still_serves_m() -> None:
    chain = spectre.IslandChain(
        n=2, m=4, psi_n=0.7, residue_o=3.89e-2, residue_x=-1.16e-2, shear=0.0498
    )
    m = spectre.compute_field_integrity_metrics(
        _output(
            n_field_periods=2,
            resonance=spectre.Resonance(n=2, m=4, n_crossings=1, shear=0.0498),
            chains=[chain],
        )
    )
    assert m.pendulum_domain is False
    assert m.severity == pytest.approx(0.366, rel=2e-2)


@pytest.mark.parametrize(
    ("residual", "trust"),
    [
        (2.5e-4, 0.41 + 100 * 1.97 * 2.5e-4),
        (8.8e-3, 0.41 + 100 * 9.73 * 8.8e-3),
        (0.05, 0.41 + 100 * 11.5 * 0.05),
        (0.2, None),
    ],
)
def test_trust_envelope_bands(residual: float, trust: float | None) -> None:
    got = spectre.trust_envelope_pct(residual)
    assert got == (pytest.approx(trust) if trust is not None else None)


@pytest.mark.parametrize(
    "refusal",
    [
        spectre.RefusalClass.SIGN_INDEFINITE,
        spectre.RefusalClass.FIELD_DIVERGED,
        spectre.RefusalClass.SEARCH_INCOMPLETE,
    ],
)
def test_metrics_refusals_serve_none(refusal: spectre.RefusalClass) -> None:
    m = spectre.compute_field_integrity_metrics(
        _output(refusal_class=refusal, chains=[])
    )
    assert m.severity is None
    assert m.refusal_class is refusal


def test_metrics_no_rational_serves_zero() -> None:
    m = spectre.compute_field_integrity_metrics(
        _output(refusal_class=spectre.RefusalClass.NO_RATIONAL, resonance=None)
    )
    assert m.severity == 0.0
    assert m.refusal_class is spectre.RefusalClass.NO_RATIONAL


def test_metrics_round_trip() -> None:
    chain = spectre.IslandChain(
        n=6, m=5, psi_n=0.64, residue_o=3.465e-3, residue_x=-3.465e-3, shear=0.2752
    )
    m = spectre.compute_field_integrity_metrics(_output(chains=[chain]))
    assert spectre.FieldIntegrityMetrics.model_validate(m.model_dump()) == m


class _Wout:
    nfp = 2
    phi = np.linspace(0.0, 0.03, 99)
    iota_full = np.linspace(0.46, 0.51, 99)


def test_run_spectre_screens_without_spectre() -> None:
    out = spectre.run_spectre(
        _Wout(), spectre_settings.SpectreSettings(max_poloidal_order=3)
    )
    assert out.refusal_class is spectre.RefusalClass.NO_RATIONAL
    assert "spectre" not in sys.modules


def test_run_spectre_needs_spectre_past_the_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "spectre", None)  # makes `import spectre` fail
    with pytest.raises(spectre.SpectreNotAvailableError):
        spectre.run_spectre(_Wout(), spectre_settings.SpectreSettings())
