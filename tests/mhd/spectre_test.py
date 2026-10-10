import importlib.metadata
from typing import Any, cast

import numpy as np
import pytest

from constellaration import forward_model
from constellaration.geometry import surface_rz_fourier
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
    "model",
    [
        spectre.FieldIntegrityMetrics,
        spectre.SearchedChain,
        spectre.ScreenedCrossing,
        spectre.SpectreOutput,
    ],
)
def test_every_field_is_documented(model: Any) -> None:
    undocumented = [
        name for name, field in model.model_fields.items() if not field.description
    ]
    assert undocumented == []


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
def test_is_fundamental_harmonic(n: int, m: int, nfp: int, ok: bool) -> None:
    assert spectre._is_fundamental_harmonic(n, m, nfp) is ok


@pytest.mark.parametrize(
    ("lo", "hi", "nfp", "expected"),
    [
        (0.46, 0.51, 2, (2, 4)),
        (0.95, 1.05, 3, (3, 3)),
        (1.15, 1.25, 3, (6, 5)),
        (0.777, 0.788, 4, (28, 36)),
    ],
)
def test_find_crossings_lowest_order_first(
    lo: float, hi: float, nfp: int, expected: tuple[int, int]
) -> None:
    iota, psi = _profile(lo, hi)
    found = spectre._find_crossings(iota, psi, nfp, 40)
    assert found, "the transform crosses a rational"
    assert (found[0].n, found[0].m) == expected
    assert found[0].shear == pytest.approx(hi - lo)
    assert found[0].psi_n == pytest.approx((found[0].n / found[0].m - lo) / (hi - lo))
    assert [r.m for r in found] == sorted(r.m for r in found)
    assert all(spectre._is_fundamental_harmonic(r.n, r.m, nfp) for r in found)
    assert all(lo <= r.n / r.m <= hi for r in found)


def test_find_crossings_enumerates_more_than_one() -> None:
    """m05_nfp3_D7RBpfuf crosses both 3/3 and 6/5; 3/3 is the lower order."""
    iota, psi = _profile(0.8676, 1.4284)
    found = spectre._find_crossings(iota, psi, 3, 40)
    labels = [(r.n, r.m) for r in found]
    assert (3, 3) in labels
    assert (6, 5) in labels
    assert labels.index((3, 3)) < labels.index((6, 5))


def test_find_crossings_one_entry_per_crossing() -> None:
    """A transform that dips below 3/8 and comes back crosses it twice."""
    psi = np.linspace(0.0, 1.0, 101)
    iota = 0.375 + 0.02 * ((psi - 0.4) ** 2 - 0.09)  # 3/8 at psi = 0.1 and 0.7
    found = [
        r for r in spectre._find_crossings(iota, psi, 3, 8) if (r.n, r.m) == (3, 8)
    ]
    assert len(found) == 2
    assert [r.psi_n for r in found] == pytest.approx([0.1, 0.7], abs=1e-3)
    # d iota / d psi = 0.04 (psi - 0.4): the two crossings have their own shear
    assert [r.shear for r in found] == pytest.approx([0.012, 0.012], rel=2e-2)


def test_find_crossings_empty_when_none_crossed() -> None:
    iota, psi = _profile(0.46, 0.51)
    assert spectre._find_crossings(iota, psi, 2, 3) == []


def test_screen_no_rational() -> None:
    iota, psi = _profile(0.2199, 0.2217)  # no j/m with m <= 20 at nfp = 1
    refusal, res = spectre.screen_rotational_transform(
        iota, psi, 1, spectre_settings.SpectreSettings()
    )
    assert refusal is spectre.RefusalClass.NO_RATIONAL
    assert res == []


def test_screen_sign_indefinite() -> None:
    iota, psi = _profile(-0.003, 0.002)
    refusal, res = spectre.screen_rotational_transform(
        iota, psi, 4, spectre_settings.SpectreSettings()
    )
    assert refusal is spectre.RefusalClass.SIGN_INDEFINITE
    assert res == []


def test_screen_refuses_orders_above_20_by_default() -> None:
    iota, psi = _profile(0.777, 0.788)  # its lowest-order rational has m > 20
    default = spectre.screen_rotational_transform(
        iota, psi, 4, spectre_settings.SpectreSettings()
    )
    assert default[0] is spectre.RefusalClass.NO_RATIONAL
    wide = spectre_settings.SpectreSettings(max_poloidal_order=40)
    assert (
        spectre.screen_rotational_transform(iota, psi, 4, wide)[0]
        is spectre.RefusalClass.NONE
    )


def test_select_crossings_keeps_the_lowest_orders() -> None:
    """m05_nfp3_DcPhJsLe crosses 3/3 and 6/5, and its 6/5 chain is the wider one."""
    iota, psi = _profile(0.8676, 1.4284)
    found = spectre._find_crossings(iota, psi, 3, 20)
    assert len({(r.n, r.m) for r in found}) > 3
    assert spectre._select_crossings(found, None) == found
    assert {(r.n, r.m) for r in spectre._select_crossings(found, 1)} == {(3, 3)}
    assert [(r.n, r.m) for r in spectre._select_crossings(found, 2)] == [
        (3, 3),
        (6, 5),
    ]


def test_select_crossings_counts_rationals_not_crossings() -> None:
    psi = np.linspace(0.0, 1.0, 101)
    iota = 0.375 + 0.02 * ((psi - 0.4) ** 2 - 0.09)
    found = spectre._find_crossings(iota, psi, 3, 20)
    lowest = spectre._select_crossings(found, 1)
    assert [(r.n, r.m) for r in lowest] == [(3, 8), (3, 8)]


def test_screen_selects_three_rationals_by_default() -> None:
    iota, psi = _profile(0.8676, 1.4284)
    _, selected = spectre.screen_rotational_transform(
        iota, psi, 3, spectre_settings.SpectreSettings()
    )
    assert len({(r.n, r.m) for r in selected}) == 3


def _ladder(residuals: list[float]) -> tuple[list[float], bool]:
    it = iter(residuals)

    def solve(_ntor: int) -> float:
        return next(it)

    return spectre._ladder(solve, spectre_settings.SpectreSettings())


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


class _CircularField:
    """A field whose s is the distance r to R = 1: s = 2 r - 1, boundary at r = 1."""

    def get_st_coords(
        self, _volume: int, r: float, z: float, _phi: float
    ) -> tuple[float, float]:
        return 2.0 * float(np.hypot(r - 1.0, z)) - 1.0, 0.0


class _CircularEquilibrium:
    """Circular VMEC flux surfaces of minor radius sqrt(psi_n) around R = 1."""

    xm = np.array([0, 1])
    normalized_toroidal_flux_full_grid_mesh = np.linspace(0.0, 1.0, 101)
    rmnc = np.array([np.ones(101), np.sqrt(np.linspace(0.0, 1.0, 101))])
    zmns = np.array([np.zeros(101), np.sqrt(np.linspace(0.0, 1.0, 101))])


def _search_window(psi_n: float) -> tuple[float, float]:
    return spectre._search_window(
        _CircularField(), cast(vmec_utils.VmecppWOut, _CircularEquilibrium()), psi_n
    )


def test_search_window_covers_the_flux_band_of_the_crossing() -> None:
    """On circular surfaces s = 2 sqrt(psi_n) - 1: psi_n -+ 0.05 around 0.25."""
    assert _search_window(0.25) == pytest.approx(
        (2 * 0.20**0.5 - 1, 2 * 0.30**0.5 - 1), abs=2e-3
    )
    assert _search_window(0.64) == pytest.approx(
        (2 * 0.59**0.5 - 1, 2 * 0.69**0.5 - 1), abs=2e-3
    )


def test_search_window_stops_short_of_the_axis_and_at_the_boundary() -> None:
    assert _search_window(0.02)[0] == -0.98
    assert _search_window(0.98)[1] == 0.999


def _crossing(**kwargs: Any) -> spectre.ScreenedCrossing:
    base: dict[str, Any] = dict(n=6, m=5, psi_n=0.64, shear=0.2752)
    base.update(kwargs)
    return spectre.ScreenedCrossing(**base)


def _found(psi_n: float, residue_o: float, residue_x: float) -> dict[str, Any]:
    s = 2.0 * psi_n**0.5 - 1.0
    return dict(
        residue_o=residue_o,
        residue_x=residue_x,
        o_point=(s, 0.0),
        x_point=(s, 0.6),
    )


def test_score_chains_single_chain_inside_the_domain() -> None:
    """Walkthrough design A: one crossing of 6/5, an island 3.3 % of the flux wide."""
    chains, _ = spectre.score_chains(
        [_crossing()], [_found(0.62, 3.465e-3, -3.465e-3)], n_field_periods=3
    )
    assert len(chains) == 1
    assert chains[0].score == pytest.approx(
        (4 / np.pi) * 3 * (3.465e-3**2) ** 0.25 / (25 * 0.2752)
    )
    assert chains[0].score == pytest.approx(0.0327, rel=1e-2)


def test_score_chains_scores_each_chain_with_its_own_shear() -> None:
    inner = _crossing(n=3, m=8, psi_n=0.07, shear=0.03)
    outer = _crossing(n=3, m=8, psi_n=0.67, shear=0.01)
    chains, _ = spectre.score_chains(
        [inner, outer],
        [_found(0.05, 1.46e-3, -1.4e-3), _found(0.69, 1.8e-7, -1.8e-7)],
        n_field_periods=3,
    )
    assert [c.psi_n for c in chains] == [0.07, 0.67]
    assert [c.crossings for c in chains] == [[inner], [outer]]
    assert chains[0].residue_o == 1.46e-3
    assert chains[1].residue_o == 1.8e-7


def _twin_crossings(distance: float) -> list[spectre.ScreenedCrossing]:
    """Two crossings of one rational, ``distance`` apart in normalised flux."""
    return [
        _crossing(n=3, m=8, psi_n=0.4, shear=0.05),
        _crossing(n=3, m=8, psi_n=0.4 + distance, shear=0.05),
    ]


def test_reconnection_parameter_of_two_equal_chains() -> None:
    """Lambda = (3 / 4) (M / L)^2: the chains join when M reaches 2 / sqrt(3) of L."""
    inner, outer = _twin_crossings(distance=0.2)
    assert spectre._reconnection_parameter(inner, outer, 0.1, 0.1) == pytest.approx(
        0.75 * 0.25
    )
    joined = 0.2 * 2 / 3**0.5
    assert spectre._reconnection_parameter(
        inner, outer, joined, joined
    ) == pytest.approx(1.0)


def test_reconnection_parameter_weights_each_chain_by_its_shear() -> None:
    inner, outer = _twin_crossings(distance=0.2)
    outer = outer.model_copy(update={"shear": 0.15})
    expected = 0.75 * (0.05 * 0.1**2 + 0.15 * 0.2**2) / ((0.05 + 0.15) * 0.2**2)
    assert spectre._reconnection_parameter(inner, outer, 0.1, 0.2) == pytest.approx(
        expected
    )


def _score_of(crossing: spectre.ScreenedCrossing, residue: float) -> float:
    """The score of a chain of residues +-``residue`` on one crossing, N_fp = 3."""
    chains, _ = spectre.score_chains(
        [crossing], [_found(crossing.psi_n, residue, -residue)], n_field_periods=3
    )
    return chains[0].score


def test_score_chains_keeps_separate_chains_apart() -> None:
    """Two thin chains of one rational, far apart: two entries, and their sum."""
    crossings = _twin_crossings(distance=0.3)
    chains, design_score = spectre.score_chains(
        crossings, [_found(c.psi_n, 1e-5, -1e-5) for c in crossings], n_field_periods=3
    )
    assert [len(chain.crossings) for chain in chains] == [1, 1]
    assert design_score == pytest.approx(chains[0].score + chains[1].score)


def test_score_chains_makes_one_chain_of_merged_chains() -> None:
    """Each chain is wider than the two crossings are apart: they are one chain."""
    crossings = _twin_crossings(distance=0.02)
    widths = [_score_of(crossing, 1e-3) for crossing in crossings]
    inner, outer = crossings
    assert spectre._reconnection_parameter(inner, outer, widths[0], widths[1]) > 1.0
    chains, design_score = spectre.score_chains(
        crossings, [_found(c.psi_n, 1e-3, -1e-3) for c in crossings], n_field_periods=3
    )
    assert len(chains) == 1
    (chain,) = chains
    assert chain.crossings == crossings
    # from the inner edge of the first islands to the outer edge of the second
    assert chain.score == pytest.approx(0.02 + (widths[0] + widths[1]) / 2)
    assert chain.psi_n == pytest.approx(0.41)
    assert chain.score < widths[0] + widths[1]
    assert design_score == pytest.approx(chain.score)


def test_score_chains_says_which_chains_merged() -> None:
    """Three crossings of one rational, the inner two joined: two chains."""
    crossings = [
        _crossing(n=3, m=8, psi_n=0.40, shear=0.05),
        _crossing(n=3, m=8, psi_n=0.42, shear=0.05),
        _crossing(n=3, m=8, psi_n=0.80, shear=0.05),
    ]
    chains, design_score = spectre.score_chains(
        crossings, [_found(c.psi_n, 1e-3, -1e-3) for c in crossings], n_field_periods=3
    )
    assert [chain.crossings for chain in chains] == [crossings[:2], crossings[2:]]
    assert design_score == pytest.approx(chains[0].score + chains[1].score)


def test_score_chains_keeps_the_largest_residues_of_merged_chains() -> None:
    crossings = _twin_crossings(distance=0.02)
    chains, _ = spectre.score_chains(
        crossings,
        [_found(0.40, 1e-3, -3e-3), _found(0.42, 2e-3, -1e-3)],
        n_field_periods=3,
    )
    (chain,) = chains
    assert (chain.residue_o, chain.residue_x) == (2e-3, -3e-3)
    assert chain.o_point == _found(0.42, 2e-3, -1e-3)["o_point"]
    assert chain.x_point == _found(0.40, 1e-3, -3e-3)["x_point"]


def test_score_chains_never_merges_different_rationals() -> None:
    crossings = [
        _crossing(n=3, m=8, psi_n=0.40, shear=0.05),
        _crossing(n=6, m=11, psi_n=0.42, shear=0.05),
    ]
    chains, _ = spectre.score_chains(
        crossings, [_found(c.psi_n, 1e-3, -1e-3) for c in crossings], n_field_periods=3
    )
    assert len(chains) == 2


def test_score_chains_skips_a_crossing_without_a_chain() -> None:
    inner = _crossing(n=3, m=8, psi_n=0.07, shear=0.03)
    outer = _crossing(n=3, m=8, psi_n=0.67, shear=0.01)
    chains, _ = spectre.score_chains(
        [inner, outer], [None, _found(0.66, 3e-3, -3e-3)], n_field_periods=3
    )
    assert len(chains) == 1
    assert chains[0].psi_n == 0.67
    assert chains[0].residue_o == 3e-3


def test_score_chains_gives_a_tangency_the_worst_score() -> None:
    """Without shear the formula diverges: the crossing is as bad as it gets."""
    chains, _ = spectre.score_chains(
        [_crossing(shear=0.0)], [_found(0.64, 1e-3, -1e-3)], n_field_periods=3
    )
    assert chains[0].score == 1.0


def test_score_chains_gives_a_large_residue_the_worst_score() -> None:
    """Above SATURATION_RESIDUE the formula under-states what the chain destroys."""
    chains, _ = spectre.score_chains(
        [_crossing()], [_found(0.64, 0.74, -0.54)], n_field_periods=3
    )
    assert chains[0].score == 1.0


def test_score_chains_caps_a_chain_at_the_whole_flux() -> None:
    chains, _ = spectre.score_chains(
        [_crossing(m=2, shear=0.006)], [_found(0.64, 0.05, -0.05)], n_field_periods=3
    )
    assert chains[0].score == 1.0


def _metrics(**kwargs: Any) -> spectre.FieldIntegrityMetrics:
    base: dict[str, Any] = dict(
        refusal_class=spectre.RefusalClass.NONE,
        crossings=[_crossing()],
        chains=[],
        beltrami_residual=2.48e-4,
    )
    base.update(kwargs)
    base.setdefault("score", min(1.0, sum(c.score for c in base["chains"])))
    return spectre.build_metrics(**base)


def _chains(*residues: float, n_field_periods: int = 3) -> list[spectre.SearchedChain]:
    crossings = [_crossing(psi_n=0.1 + 0.2 * i) for i in range(len(residues))]
    found = [
        _found(r.psi_n, residue, -residue) for r, residue in zip(crossings, residues)
    ]
    return spectre.score_chains(crossings, found, n_field_periods)[0]


def test_metrics_single_chain() -> None:
    chains = _chains(3.465e-3)
    m = _metrics(chains=chains)
    assert m.field_integrity_score == pytest.approx(chains[0].score)
    assert m.chains == chains
    assert m.refusal_class is spectre.RefusalClass.NONE


def test_metrics_score_is_the_sum_over_chains() -> None:
    chains = _chains(3.465e-3, 1.8e-7, 4.2e-4)
    m = _metrics(chains=chains)
    assert m.field_integrity_score is not None
    assert m.field_integrity_score == pytest.approx(sum(c.score for c in chains))
    assert m.field_integrity_score > max(c.score for c in chains)


def test_metrics_of_a_chain_far_from_a_pendulum() -> None:
    """Measured: this chain holds 0.284 of the flux, and (2 / pi) M gives 0.297."""
    crossing = _crossing(n=2, m=4, psi_n=0.7, shear=0.0498)
    chains, _ = spectre.score_chains(
        [crossing], [_found(0.7, 3.89e-2, -1.16e-2)], n_field_periods=2
    )
    m = _metrics(crossings=[crossing], chains=chains)
    assert m.field_integrity_score == pytest.approx(0.466, rel=2e-2)


def test_score_never_exceeds_the_whole_flux() -> None:
    """Two chains of different rationals, each as bad as it gets."""
    crossings = [_crossing(psi_n=0.3), _crossing(n=3, m=4, psi_n=0.7)]
    chains, design_score = spectre.score_chains(
        crossings, [_found(c.psi_n, 0.7, -0.7) for c in crossings], n_field_periods=3
    )
    assert sum(c.score for c in chains) > 1.0
    assert design_score == 1.0


def test_predicted_flux_fraction() -> None:
    # The score is the width of the band the islands span; they fill 2 / pi of it.
    assert spectre.predicted_flux_fraction(0.1) == pytest.approx((2.0 / np.pi) * 0.1)


def test_island_width_in_metres() -> None:
    """A band 0.0327 wide in flux at psi_n = 0.64, with r = a sqrt(psi_n)."""
    chain = _chains(3.465e-3)[0].model_copy(update={"psi_n": 0.64, "score": 0.0327})
    width = spectre.island_width(chain, minor_radius=0.5)
    assert width == pytest.approx(0.5 * (0.65635**0.5 - 0.62365**0.5))
    assert width == pytest.approx(0.5 * 0.0327 / (2 * 0.8), rel=1e-3)


def test_island_width_stops_at_the_axis_and_the_boundary() -> None:
    chain = _chains(3.465e-3)[0].model_copy(update={"psi_n": 0.9, "score": 1.0})
    assert spectre.island_width(chain, minor_radius=0.5) == pytest.approx(
        0.5 * (1.0 - 0.4**0.5)
    )


@pytest.mark.parametrize(
    "refusal",
    [
        spectre.RefusalClass.SIGN_INDEFINITE,
        spectre.RefusalClass.FIELD_DIVERGED,
    ],
)
def test_metrics_refusals_serve_none(refusal: spectre.RefusalClass) -> None:
    m = _metrics(refusal_class=refusal, chains=[])
    assert m.field_integrity_score is None
    assert m.refusal_class is refusal


def test_metrics_no_chain_found_scores_zero() -> None:
    """Islands the search cannot find are taken to be too thin to matter."""
    m = _metrics(chains=[])
    assert m.field_integrity_score == 0.0
    assert m.refusal_class is spectre.RefusalClass.SEARCH_INCOMPLETE
    assert len(m.crossings) == 1


def test_metrics_no_rational_serves_zero() -> None:
    m = _metrics(refusal_class=spectre.RefusalClass.NO_RATIONAL, crossings=[])
    assert m.field_integrity_score == 0.0
    assert m.refusal_class is spectre.RefusalClass.NO_RATIONAL


def test_metrics_round_trip() -> None:
    m = _metrics(chains=_chains(3.465e-3))
    assert spectre.FieldIntegrityMetrics.model_validate(m.model_dump()) == m
    assert spectre.FieldIntegrityMetrics.model_validate_json(m.model_dump_json()) == m


def test_spectre_output_round_trips_through_json() -> None:
    output = spectre.SpectreOutput(
        h5_file=bytes(range(256)),
        crossings=[_crossing()],
        refusal_class=spectre.RefusalClass.NONE,
        beltrami_residuals=[0.05, 2.48e-4],
    )
    assert spectre.SpectreOutput.model_validate_json(output.model_dump_json()) == output


class _Equilibrium:
    """The fields the screen reads off a ``VmecppWOut``."""

    nfp = 2
    n_field_periods = 2
    normalized_toroidal_flux_full_grid_mesh = np.linspace(0.0, 1.0, 99)
    iotaf = np.linspace(0.46, 0.51, 99)
    mpol = 6


def _equilibrium() -> vmec_utils.VmecppWOut:
    return cast(vmec_utils.VmecppWOut, _Equilibrium())


def test_choose_resolution_follows_the_highest_order() -> None:
    crossings = [_crossing(n=2, m=4), _crossing(n=6, m=13)]
    solver = spectre_settings.SpectreSettings()
    # mpol = ceil(1.5 * 13) = 20, lrad = mpol + 4
    assert spectre.choose_resolution(_equilibrium(), crossings, solver) == (20, 24)
    fixed = spectre_settings.SpectreSettings(radial_resolution=30)
    assert spectre.choose_resolution(_equilibrium(), crossings, fixed) == (20, 30)


def test_choose_resolution_is_never_below_the_boundary() -> None:
    """A boundary with 31 poloidal modes is solved with mpol = 30, not the floor 14."""
    equilibrium = _Equilibrium()
    equilibrium.mpol = 31
    resolution = spectre.choose_resolution(
        cast(vmec_utils.VmecppWOut, equilibrium),
        [_crossing(n=2, m=4)],
        spectre_settings.SpectreSettings(),
    )
    assert resolution == (30, 34)


def test_a_design_refused_at_the_screen_is_never_solved() -> None:
    solver = spectre_settings.SpectreSettings(max_poloidal_order=3)
    output = spectre.run_spectre(_equilibrium(), solver)
    assert output.h5_file is None
    assert output.refusal_class is spectre.RefusalClass.NO_RATIONAL
    m = spectre.compute_field_integrity_metrics(output, _equilibrium())
    assert m.refusal_class is spectre.RefusalClass.NO_RATIONAL
    assert m.field_integrity_score == 0.0


def test_a_diverged_field_is_not_searched() -> None:
    output = spectre.SpectreOutput(
        h5_file=b"not read",
        crossings=[_crossing()],
        refusal_class=spectre.RefusalClass.FIELD_DIVERGED,
        beltrami_residuals=[26.0, 26.0],
    )
    m = spectre.compute_field_integrity_metrics(output, _equilibrium())
    assert m.refusal_class is spectre.RefusalClass.FIELD_DIVERGED
    assert m.field_integrity_score is None
    assert m.beltrami_residual == 26.0


def test_forward_model_does_not_run_spectre_by_default() -> None:
    assert forward_model.ConstellarationSettings().spectre_settings is None
    fields = forward_model.ConstellarationMetrics.model_fields
    assert fields["field_integrity"].default is None


# The boundary of the dataset design with vmecpp_wout_id DcPhJsLe4Zv8aphYRMNhuRw: its
# rotational transform crosses 3/3 and 6/5, and the 6/5 chain is the larger one.
_TWO_CHAIN_BOUNDARY = dict(
    r_cos=[
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.02702702702702703, 0.0, 0.0],
        [
            0.0,
            0.0,
            0.08978104821425961,
            -0.019465376230152362,
            0.17956209642851922,
            -0.019465376230152362,
            0.0,
            0.0,
            0.0,
        ],
    ],
    z_sin=[
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.0,
            0.0,
            0.08978104821425961,
            0.0064884587433841215,
            -0.17956209642851922,
            0.0064884587433841215,
            0.0,
            0.0,
            0.0,
        ],
    ],
    n_field_periods=3,
    is_stellarator_symmetric=True,
)


@pytest.mark.spectre_solve
def test_forward_model_scores_both_chains_of_a_two_chain_design() -> None:
    """From a boundary to its metrics; a few minutes and about 3.5 GB on one core."""
    boundary = surface_rz_fourier.SurfaceRZFourier.model_validate(_TWO_CHAIN_BOUNDARY)
    settings = forward_model.ConstellarationSettings(
        boozer_preset_settings=None,
        qi_settings=None,
        turbulent_settings=None,
        spectre_settings=spectre_settings.SpectreSettings(max_rationals=2),
    )
    metrics, _ = forward_model.forward_model(boundary, settings=settings)
    field_integrity = metrics.field_integrity
    assert field_integrity is not None
    assert field_integrity.refusal_class is spectre.RefusalClass.NONE
    assert [(c.n, c.m) for c in field_integrity.chains] == [(3, 3), (6, 5)]
    lowest_order, higher_order = field_integrity.chains
    # 0.0344 and 0.0497 on the dataset's high-fidelity equilibrium; the forward
    # model's default VMEC resolution moves the shear by a few percent.
    assert lowest_order.score == pytest.approx(0.0344, rel=0.15)
    assert higher_order.score == pytest.approx(0.0497, rel=0.15)
    assert higher_order.score > lowest_order.score
    assert field_integrity.field_integrity_score == pytest.approx(
        lowest_order.score + higher_order.score
    )
