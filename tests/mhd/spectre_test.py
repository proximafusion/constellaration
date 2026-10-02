import importlib.metadata
from typing import Any, cast

import numpy as np
import pytest

from constellaration.mhd import spectre, spectre_settings, vmec_utils


def _profile(lo: float, hi: float, n: int = 99) -> tuple[np.ndarray, np.ndarray]:
    psi = np.linspace(0.0, 1.0, n)
    return lo + (hi - lo) * psi, psi


SPECTRE_COMMIT = "f96b2b541dc1aa48bd7d39e6ccaf30343c3da29e"


def test_spectre_is_a_declared_dependency() -> None:
    requirements = importlib.metadata.requires("constellaration") or []
    pinned = "spectre@git+https://gitlab.com/spectre-eq/spectre.git@" + SPECTRE_COMMIT
    assert any(r.replace(" ", "").startswith(pinned) for r in requirements)


def test_no_optional_backend_error() -> None:
    assert not hasattr(spectre, "SpectreNotAvailableError")


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
def test_crossed_resonances_lowest_order_first(
    lo: float, hi: float, nfp: int, expected: tuple[int, int]
) -> None:
    iota, psi = _profile(lo, hi)
    found = spectre.crossed_resonances(iota, psi, nfp, 40)
    assert found, "the transform crosses an admissible rational"
    assert (found[0].n, found[0].m) == expected
    assert found[0].n_crossings == 1
    assert found[0].shear == pytest.approx(hi - lo)
    assert [r.m for r in found] == sorted(r.m for r in found)
    assert all(spectre.admissible(r.n, r.m, nfp) for r in found)
    assert all(lo <= r.iota <= hi for r in found)


def test_crossed_resonances_enumerates_more_than_one() -> None:
    """m05_nfp3_D7RBpfuf crosses both 3/3 and 6/5; 3/3 is the lower order."""
    iota, psi = _profile(0.8676, 1.4284)
    found = spectre.crossed_resonances(iota, psi, 3, 40)
    labels = [(r.n, r.m) for r in found]
    assert (3, 3) in labels
    assert (6, 5) in labels
    assert labels.index((3, 3)) < labels.index((6, 5))


def test_select_resonances_serves_the_lowest_order() -> None:
    iota, psi = _profile(0.8676, 1.4284)
    found = spectre.crossed_resonances(iota, psi, 3, 40)
    chosen = spectre.select_resonances(found)
    assert len(chosen) == 1
    assert (chosen[0].n, chosen[0].m) == (found[0].n, found[0].m)


def test_crossed_resonances_empty_when_none_admissible() -> None:
    iota, psi = _profile(0.46, 0.51)
    assert spectre.crossed_resonances(iota, psi, 2, 3) == []


def test_screen_no_rational() -> None:
    iota, psi = _profile(0.2199, 0.2217)  # no j/m with m <= 40 at nfp = 1
    refusal, res = spectre.screen(iota, psi, 1, spectre_settings.SpectreSettings())
    assert refusal is spectre.RefusalClass.NO_RATIONAL
    assert res == []


def test_screen_sign_indefinite() -> None:
    iota, psi = _profile(-0.003, 0.002)
    refusal, res = spectre.screen(iota, psi, 4, spectre_settings.SpectreSettings())
    assert refusal is spectre.RefusalClass.SIGN_INDEFINITE
    assert res == []


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


def _metrics(**kwargs: Any) -> spectre.FieldIntegrityMetrics:
    """The reduction, with the commissioning row's defaults."""
    base: dict[str, Any] = dict(
        refusal_class=spectre.RefusalClass.NONE,
        resonance=spectre.ScreenedResonance(n=6, m=5, n_crossings=1, shear=0.2752),
        chains=[],
        beltrami_residual=2.48e-4,
        n_field_periods=3,
    )
    base.update(kwargs)
    return spectre._metrics_from(**base)


def test_metrics_single_chain_inside_the_domain() -> None:
    chain = spectre.SearchedChain(
        n=6, m=5, psi_n=0.64, residue_o=3.465e-3, residue_x=-3.465e-3, shear=0.2752
    )
    m = _metrics(chains=[chain])
    expected = 3 * (3.465e-3**2) ** 0.25 / (25 * 0.2752)
    assert m.severity == pytest.approx(expected)
    assert m.severity_poloidal_mode == 5
    assert m.pendulum_domain is True
    assert m.residue_ratio == pytest.approx(1.0)
    assert m.trust_pct == pytest.approx(0.41 + 100 * 1.97 * 2.48e-4)
    assert m.refusal_class is spectre.RefusalClass.NONE


def test_metrics_sum_over_distinct_chains() -> None:
    a = spectre.SearchedChain(
        n=3, m=8, psi_n=0.07, residue_o=1.46e-3, residue_x=-1.4e-3, shear=0.02235
    )
    b = spectre.SearchedChain(
        n=3, m=8, psi_n=0.67, residue_o=1.8e-7, residue_x=-1.8e-7, shear=0.02235
    )
    resonance = spectre.ScreenedResonance(n=3, m=8, n_crossings=2, shear=0.02235)
    m = _metrics(resonance=resonance, chains=[a, b])
    single = _metrics(resonance=resonance, chains=[a])
    assert m.severity is not None
    assert single.severity is not None
    assert m.severity > single.severity
    assert m.n_chains_enumerated == 2
    assert m.n_chains_found == 2


def test_metrics_outside_the_domain_still_serves_m() -> None:
    chain = spectre.SearchedChain(
        n=2, m=4, psi_n=0.7, residue_o=3.89e-2, residue_x=-1.16e-2, shear=0.0498
    )
    m = _metrics(
        n_field_periods=2,
        resonance=spectre.ScreenedResonance(n=2, m=4, n_crossings=1, shear=0.0498),
        chains=[chain],
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
    ("residue_o", "residue_x", "inside"),
    [
        (3.465e-3, -3.465e-3, True),  # the ratio is exactly 1
        (3.89e-2, -1.16e-2, False),  # ratio 0.30, outside [0.85, 1.15]
        (0.2, -0.2, False),  # |R_O| above 0.10
        (0.0, -1e-3, False),  # a zero O residue is not a pendulum
    ],
)
def test_in_pendulum_domain(residue_o: float, residue_x: float, inside: bool) -> None:
    assert spectre.in_pendulum_domain(residue_o, residue_x) is inside


def test_predicted_flux_fraction() -> None:
    assert spectre.predicted_flux_fraction(0.1) == pytest.approx(
        spectre.KAPPA * (4.0 / np.pi) * 0.1
    )


@pytest.mark.parametrize(
    "refusal",
    [
        spectre.RefusalClass.SIGN_INDEFINITE,
        spectre.RefusalClass.FIELD_DIVERGED,
        spectre.RefusalClass.SEARCH_INCOMPLETE,
    ],
)
def test_metrics_refusals_serve_none(refusal: spectre.RefusalClass) -> None:
    m = _metrics(refusal_class=refusal, chains=[])
    assert m.severity is None
    assert m.refusal_class is refusal


def test_metrics_no_rational_serves_zero() -> None:
    m = _metrics(refusal_class=spectre.RefusalClass.NO_RATIONAL, resonance=None)
    assert m.severity == 0.0
    assert m.refusal_class is spectre.RefusalClass.NO_RATIONAL


def test_metrics_round_trip() -> None:
    chain = spectre.SearchedChain(
        n=6, m=5, psi_n=0.64, residue_o=3.465e-3, residue_x=-3.465e-3, shear=0.2752
    )
    m = _metrics(chains=[chain])
    assert spectre.FieldIntegrityMetrics.model_validate(m.model_dump()) == m


class _Equilibrium:
    """The fields ``compute_spectre_metrics`` reads off a ``VmecppWOut``."""

    nfp = 2
    n_field_periods = 2
    phi = np.linspace(0.0, 0.03, 99)
    iotaf = np.linspace(0.46, 0.51, 99)


def _equilibrium() -> vmec_utils.VmecppWOut:
    return cast(vmec_utils.VmecppWOut, _Equilibrium())


def test_compute_spectre_metrics_refuses_at_the_screen() -> None:
    m = spectre.compute_spectre_metrics(
        _equilibrium(), spectre_settings.SpectreSettings(max_poloidal_order=3)
    )
    assert m.refusal_class is spectre.RefusalClass.NO_RATIONAL
    assert m.severity == 0.0
