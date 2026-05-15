import numpy as np
import pytest

from constellaration import forward_model, mhd_stable_qi_scoring, problems
from constellaration.geometry import surface_rz_fourier


def test_pareto_levels_empty_input() -> None:
    levels = mhd_stable_qi_scoring.pareto_levels(np.empty((0, 2)))
    assert levels == []


def test_pareto_levels_single_point() -> None:
    levels = mhd_stable_qi_scoring.pareto_levels(np.array([[0.0, 0.0]]))
    assert len(levels) == 1
    np.testing.assert_array_equal(levels[0], [0])


def test_pareto_levels_two_objectives_clean_ordering() -> None:
    # A=(0,3) and B=(3,0) trade off (neither dominates the other).
    # C=(1,1) is dominated by neither A nor B, so it sits on L0 too.
    # D=(2,2) is dominated by C, so it falls into L1.
    objectives = np.array(
        [
            [0.0, 3.0],  # A
            [3.0, 0.0],  # B
            [1.0, 1.0],  # C
            [2.0, 2.0],  # D
        ]
    )
    levels = mhd_stable_qi_scoring.pareto_levels(objectives)
    assert len(levels) == 2
    np.testing.assert_array_equal(np.sort(levels[0]), [0, 1, 2])
    np.testing.assert_array_equal(np.sort(levels[1]), [3])


def test_pareto_levels_all_identical_points_one_level() -> None:
    objectives = np.tile([1.0, 1.0], (5, 1))
    levels = mhd_stable_qi_scoring.pareto_levels(objectives)
    # Identical points don't strictly dominate each other, so all in L0.
    assert len(levels) == 1
    np.testing.assert_array_equal(np.sort(levels[0]), [0, 1, 2, 3, 4])


def test_pareto_levels_partition_is_complete() -> None:
    rng = np.random.default_rng(0)
    objectives = rng.standard_normal((50, 2))
    levels = mhd_stable_qi_scoring.pareto_levels(objectives)
    all_indices = np.concatenate(levels)
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(50))
    # No duplicates across levels.
    assert len(set(all_indices.tolist())) == 50


def test_hypervolume_zero_for_empty_set() -> None:
    assert (
        mhd_stable_qi_scoring.hypervolume(np.empty((0, 2)), np.array([1.0, 1.0])) == 0.0
    )


def test_hypervolume_matches_known_value_for_single_point() -> None:
    # Reference (1, 1), point (0.4, 0.3) -> dominated rectangle 0.6 * 0.7 = 0.42
    value = mhd_stable_qi_scoring.hypervolume(
        np.array([[0.4, 0.3]]), np.array([1.0, 1.0])
    )
    assert value == pytest.approx(0.42)


def test_select_top_pareto_levels_drops_levels_that_collapse_hiv() -> None:
    # Three points form a true trade-off (none dominates another -> all in L0),
    # and a fourth point dominated by them (-> L1). The L1 sub-front HIV is
    # well below 0.9 * HIV(L0), so only L0 is kept.
    objectives = np.array(
        [
            [-3.0, 8.0],  # L0: best lgradB, worst AR
            [-2.0, 6.0],  # L0: mid trade-off
            [-1.0, 4.0],  # L0: worst lgradB, best AR
            [0.0, 10.0],  # L1: dominated by all three above
        ]
    )
    ref = np.array([1.0, 20.0])
    levels = mhd_stable_qi_scoring.pareto_levels(objectives)
    # Sanity check on the input setup.
    np.testing.assert_array_equal(np.sort(levels[0]), [0, 1, 2])
    np.testing.assert_array_equal(np.sort(levels[1]), [3])
    hiv_l0 = mhd_stable_qi_scoring.hypervolume(objectives[levels[0]], ref)
    hiv_l1_only = mhd_stable_qi_scoring.hypervolume(objectives[levels[1]], ref)
    assert hiv_l1_only < 0.9 * hiv_l0

    selected = mhd_stable_qi_scoring.select_top_pareto_levels_by_hiv_fraction(
        objectives, reference_point=ref, fraction=0.9
    )
    np.testing.assert_array_equal(selected, [0, 1, 2])


def test_select_top_pareto_levels_includes_l1_when_almost_as_good() -> None:
    # L0 is a trade-off front. L1 is a slightly-shifted copy of the same
    # trade-off, so its sub-front HIV is comparable to L0 -- both should be
    # kept (i.e. ratio >= 0.9).
    objectives = np.array(
        [
            [-3.0, 8.0],  # L0
            [-2.0, 6.0],  # L0
            [-1.0, 4.0],  # L0
            [-2.9, 8.5],  # L1: dominated only by (-3.0, 8.0)
            [-1.9, 6.5],  # L1: dominated only by (-2.0, 6.0)
            [-0.9, 4.5],  # L1: dominated only by (-1.0, 4.0)
        ]
    )
    ref = np.array([1.0, 20.0])
    levels = mhd_stable_qi_scoring.pareto_levels(objectives)
    assert len(levels) == 2
    hiv_l0 = mhd_stable_qi_scoring.hypervolume(objectives[levels[0]], ref)
    hiv_l1 = mhd_stable_qi_scoring.hypervolume(objectives[levels[1]], ref)
    # L1 is a small shift of L0 so it captures most of L0's HIV.
    assert hiv_l1 >= 0.9 * hiv_l0

    selected = mhd_stable_qi_scoring.select_top_pareto_levels_by_hiv_fraction(
        objectives, reference_point=ref, fraction=0.9
    )
    # All six should be included.
    np.testing.assert_array_equal(selected, [0, 1, 2, 3, 4, 5])


def test_select_top_pareto_levels_rejects_invalid_fraction() -> None:
    objectives = np.array([[-1.0, 5.0]])
    ref = np.array([1.0, 20.0])
    with pytest.raises(ValueError, match="fraction must be in"):
        mhd_stable_qi_scoring.select_top_pareto_levels_by_hiv_fraction(
            objectives, ref, fraction=0.0
        )
    with pytest.raises(ValueError, match="fraction must be in"):
        mhd_stable_qi_scoring.select_top_pareto_levels_by_hiv_fraction(
            objectives, ref, fraction=1.1
        )


def test_select_top_pareto_levels_handles_empty_input() -> None:
    out = mhd_stable_qi_scoring.select_top_pareto_levels_by_hiv_fraction(
        np.empty((0, 2)), np.array([1.0, 20.0])
    )
    assert out.size == 0


def test_select_top_pareto_levels_handles_points_outside_reference() -> None:
    # All points dominated by the reference -> L0 HIV is 0 -> just return L0.
    objectives = np.array([[5.0, 25.0], [10.0, 30.0]])
    selected = mhd_stable_qi_scoring.select_top_pareto_levels_by_hiv_fraction(
        objectives, reference_point=np.array([1.0, 20.0])
    )
    # First arg's L0 is [0] (it dominates [1]); fall-back returns L0.
    np.testing.assert_array_equal(selected, [0])


# ---------- binned_diversity_score tests ----------


def _rotating_ellipse_at_ar(
    target_ar: float, triangularity: float = 0.0
) -> surface_rz_fourier.SurfaceRZFourier:
    """Build a stellarator-symmetric NFP=3 surface then scale to target AR."""
    r_cos = np.array(
        [
            [0.0, 10.0, 0.0],
            [0.1, 1.0, triangularity],
        ]
    )
    z_sin = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 1.0, triangularity],
        ]
    )
    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=3,
        is_stellarator_symmetric=True,
    )
    return surface_rz_fourier.scale_to_aspect_ratio(
        surface, target_aspect_ratio=target_ar
    )


def test_binned_diversity_empty_input_is_zero() -> None:
    score = mhd_stable_qi_scoring.binned_diversity_score(
        boundaries=[],
        aspect_ratios=np.array([]),
        lgradB_values=np.array([]),
    )
    assert score == 0.0


def test_binned_diversity_score_validates_length_mismatch() -> None:
    b = _rotating_ellipse_at_ar(7.0)
    with pytest.raises(ValueError, match="same length"):
        mhd_stable_qi_scoring.binned_diversity_score(
            boundaries=[b, b],
            aspect_ratios=np.array([7.0]),
            lgradB_values=np.array([1.0, 2.0]),
        )


def test_binned_diversity_score_zero_when_single_point_per_bin() -> None:
    # One configuration per bin -> every bin scores 0 -> overall score 0.
    boundaries = [_rotating_ellipse_at_ar(ar) for ar in [6.3, 6.9, 7.5]]
    score = mhd_stable_qi_scoring.binned_diversity_score(
        boundaries=boundaries,
        aspect_ratios=np.array([6.3, 6.9, 7.5]),
        lgradB_values=np.array([1.0, 1.0, 1.0]),
        n_poloidal_points=8,
        n_toroidal_points=8,
    )
    assert score == 0.0


def test_binned_diversity_score_drops_boundaries_outside_range() -> None:
    # Two boundaries inside one bin, two outside the global range.
    boundaries = [
        _rotating_ellipse_at_ar(6.4, triangularity=0.0),
        _rotating_ellipse_at_ar(6.4, triangularity=0.1),
        _rotating_ellipse_at_ar(4.0),  # outside ar_min
        _rotating_ellipse_at_ar(15.0),  # outside ar_max
    ]
    score = mhd_stable_qi_scoring.binned_diversity_score(
        boundaries=boundaries,
        aspect_ratios=np.array([6.4, 6.4, 4.0, 15.0]),
        lgradB_values=np.array([1.0, 1.0, 1.0, 1.0]),
        n_poloidal_points=8,
        n_toroidal_points=8,
    )
    # One non-empty bin out of ten -> score is positive but small.
    assert 0.0 < score < 1.0
    # Score / n_bins must equal the within-bin distance / n_bins.
    # i.e. bin 0 distance = 10 * score.
    bin_distance = score * 10
    assert bin_distance > 0


def test_binned_diversity_score_returns_average_over_all_bins() -> None:
    # Populate two bins each with two slightly-different boundaries; the
    # other 8 bins are empty. Final score = (d_bin1 + d_bin2) / 10.
    boundaries = [
        _rotating_ellipse_at_ar(6.3, triangularity=0.0),
        _rotating_ellipse_at_ar(6.3, triangularity=0.1),
        _rotating_ellipse_at_ar(9.3, triangularity=0.0),
        _rotating_ellipse_at_ar(9.3, triangularity=0.15),
    ]
    score = mhd_stable_qi_scoring.binned_diversity_score(
        boundaries=boundaries,
        aspect_ratios=np.array([6.3, 6.3, 9.3, 9.3]),
        lgradB_values=np.array([1.0, 1.0, 1.0, 1.0]),
        n_poloidal_points=8,
        n_toroidal_points=8,
    )
    # Manually compute each bin's mean pairwise distance.
    d_bin1 = surface_rz_fourier.compute_rms_normal_displacement_distance(
        boundaries[0], boundaries[1], n_poloidal_points=8, n_toroidal_points=8
    )
    d_bin2 = surface_rz_fourier.compute_rms_normal_displacement_distance(
        boundaries[2], boundaries[3], n_poloidal_points=8, n_toroidal_points=8
    )
    expected = (d_bin1 + d_bin2) / 10.0
    assert score == pytest.approx(expected, rel=1e-3)


def test_binned_diversity_score_caps_per_bin_using_lgradB() -> None:
    # Three boundaries in the same bin, max_per_bin=2 -> only the two with
    # the highest lgradB are kept (the unique boundary is dropped).
    triangs = [0.0, 0.1, 0.2]
    boundaries = [_rotating_ellipse_at_ar(7.5, triangularity=t) for t in triangs]
    score_all = mhd_stable_qi_scoring.binned_diversity_score(
        boundaries=boundaries,
        aspect_ratios=np.array([7.5, 7.5, 7.5]),
        lgradB_values=np.array([1.0, 2.0, 3.0]),
        n_bins=1,
        ar_min=7.2,
        ar_max=7.8,
        max_per_bin=15,
        n_poloidal_points=8,
        n_toroidal_points=8,
    )
    score_capped = mhd_stable_qi_scoring.binned_diversity_score(
        boundaries=boundaries,
        aspect_ratios=np.array([7.5, 7.5, 7.5]),
        lgradB_values=np.array([1.0, 2.0, 3.0]),
        n_bins=1,
        ar_min=7.2,
        ar_max=7.8,
        max_per_bin=2,
        n_poloidal_points=8,
        n_toroidal_points=8,
    )
    expected_capped = surface_rz_fourier.compute_rms_normal_displacement_distance(
        boundaries[1], boundaries[2], n_poloidal_points=8, n_toroidal_points=8
    )
    assert score_capped == pytest.approx(expected_capped, rel=1e-3)
    # The all-three average should differ from the top-2-only average since
    # the dropped pair has a different distance.
    assert score_all != pytest.approx(score_capped, rel=1e-3)


# ---------- top-level scoring API ----------


def _make_metrics(
    aspect_ratio: float,
    lgradB: float,
) -> forward_model.ConstellarationMetrics:
    """Return a ConstellarationMetrics that is feasible at all tightness levels."""
    base = dict(
        aspect_ratio=aspect_ratio,
        axis_magnetic_mirror_ratio=0.1,
        aspect_ratio_over_edge_rotational_transform=8.0,
        axis_rotational_transform_over_n_field_periods=0.2,
        average_triangularity=-0.6,
        edge_rotational_transform_over_n_field_periods=0.30,
        max_elongation=4.0,
        qi=1e-5,  # log10 = -5, well below -3.5
        edge_magnetic_mirror_ratio=0.1,
        flux_compression_in_regions_of_bad_curvature=0.5,
        vacuum_well=0.1,
        minimum_normalized_magnetic_gradient_scale_length=lgradB,
    )
    return forward_model.ConstellarationMetrics(**base)


def test_problem_for_tightness_level_returns_correct_classes() -> None:
    assert isinstance(
        mhd_stable_qi_scoring.problem_for_tightness_level("tight"),
        problems.MHDStableQIStellarator,
    )
    assert isinstance(
        mhd_stable_qi_scoring.problem_for_tightness_level("medium"),
        problems.MHDStableQIStellaratorMedium,
    )
    assert isinstance(
        mhd_stable_qi_scoring.problem_for_tightness_level("loose"),
        problems.MHDStableQIStellaratorLoose,
    )
    with pytest.raises(ValueError, match="tightness_level"):
        mhd_stable_qi_scoring.problem_for_tightness_level("absurd")  # type: ignore[arg-type]


def test_score_boundaries_performance_with_precomputed_metrics() -> None:
    # Three feasible points forming a trade-off, plus one infeasible.
    boundaries = [
        _rotating_ellipse_at_ar(7.0),
        _rotating_ellipse_at_ar(9.0),
        _rotating_ellipse_at_ar(11.0),
        _rotating_ellipse_at_ar(8.0),  # will be made infeasible
    ]
    metrics = [
        _make_metrics(7.0, 3.0),
        _make_metrics(9.0, 5.0),
        _make_metrics(11.0, 7.0),
        _make_metrics(8.0, 4.0).model_copy(
            update=dict(flux_compression_in_regions_of_bad_curvature=2.0)
        ),
    ]
    score = mhd_stable_qi_scoring.score_boundaries_performance(
        boundaries, "tight", metrics=metrics
    )
    # Should equal the HIV of the 3 feasible points (the 4th is infeasible).
    feasible_objs = np.array(
        [
            (-3.0, 7.0),
            (-5.0, 9.0),
            (-7.0, 11.0),
        ]
    )
    expected = mhd_stable_qi_scoring.hypervolume(
        feasible_objs, mhd_stable_qi_scoring.REFERENCE_POINT_LGRADB_AR
    )
    assert score == pytest.approx(expected, rel=1e-9)


def test_score_boundaries_performance_returns_zero_when_no_feasible() -> None:
    boundaries = [_rotating_ellipse_at_ar(8.0)]
    metrics = [
        _make_metrics(8.0, 4.0).model_copy(
            update=dict(flux_compression_in_regions_of_bad_curvature=5.0)
        )
    ]
    score = mhd_stable_qi_scoring.score_boundaries_performance(
        boundaries, "tight", metrics=metrics
    )
    assert score == 0.0


def test_score_boundaries_performance_loose_admits_more_than_tight() -> None:
    # A configuration violating Tight constraints but satisfying Loose.
    boundaries = [_rotating_ellipse_at_ar(8.0)]
    metrics = [
        _make_metrics(8.0, 4.0).model_copy(
            update=dict(
                # vacuum_well -0.04 -> fails tight (>=0) and medium (>=-0.025)
                # but passes loose (>=-0.05).
                vacuum_well=-0.04,
            )
        )
    ]
    tight = mhd_stable_qi_scoring.score_boundaries_performance(
        boundaries, "tight", metrics=metrics
    )
    medium = mhd_stable_qi_scoring.score_boundaries_performance(
        boundaries, "medium", metrics=metrics
    )
    loose = mhd_stable_qi_scoring.score_boundaries_performance(
        boundaries, "loose", metrics=metrics
    )
    assert tight == 0.0
    assert medium == 0.0
    assert loose > 0.0


def test_score_boundaries_performance_validates_metrics_length() -> None:
    boundaries = [_rotating_ellipse_at_ar(7.0), _rotating_ellipse_at_ar(8.0)]
    metrics = [_make_metrics(7.0, 3.0)]  # mismatched length
    with pytest.raises(ValueError, match="same length"):
        mhd_stable_qi_scoring.score_boundaries_performance(
            boundaries, "tight", metrics=metrics
        )


def test_score_boundaries_diversity_returns_zero_with_too_few_feasible() -> None:
    boundaries = [_rotating_ellipse_at_ar(7.0)]
    metrics = [_make_metrics(7.0, 3.0)]
    assert (
        mhd_stable_qi_scoring.score_boundaries_diversity(
            boundaries, "tight", metrics=metrics
        )
        == 0.0
    )


def test_score_boundaries_diversity_matches_binned_score_on_included_set() -> None:
    # 4 feasible boundaries: 2 in bin 6.3, 2 in bin 9.3. All in L0 (different
    # trade-offs), so the HIV filter keeps all of them.
    boundaries = [
        _rotating_ellipse_at_ar(6.3, triangularity=0.0),
        _rotating_ellipse_at_ar(6.3, triangularity=0.1),
        _rotating_ellipse_at_ar(9.3, triangularity=0.0),
        _rotating_ellipse_at_ar(9.3, triangularity=0.15),
    ]
    metrics = [
        _make_metrics(6.3, 3.0),
        _make_metrics(6.3, 2.9),
        _make_metrics(9.3, 5.0),
        _make_metrics(9.3, 4.9),
    ]
    score = mhd_stable_qi_scoring.score_boundaries_diversity(
        boundaries,
        "tight",
        metrics=metrics,
        n_poloidal_points=8,
        n_toroidal_points=8,
    )
    # Manually compute the expected per-bin distances.
    d_bin1 = surface_rz_fourier.compute_rms_normal_displacement_distance(
        boundaries[0], boundaries[1], n_poloidal_points=8, n_toroidal_points=8
    )
    d_bin2 = surface_rz_fourier.compute_rms_normal_displacement_distance(
        boundaries[2], boundaries[3], n_poloidal_points=8, n_toroidal_points=8
    )
    expected = (d_bin1 + d_bin2) / 10.0
    assert score == pytest.approx(expected, rel=1e-3)


def test_score_boundaries_diversity_drops_dominated_levels_via_hiv_filter() -> None:
    # Add a dominated configuration in a third bin. With the default 90%
    # filter it should be dropped, so the bin score for that third bin is 0
    # and the overall score matches the two-bin case.
    boundaries = [
        _rotating_ellipse_at_ar(6.3, triangularity=0.0),
        _rotating_ellipse_at_ar(6.3, triangularity=0.1),
        _rotating_ellipse_at_ar(9.3, triangularity=0.0),
        _rotating_ellipse_at_ar(9.3, triangularity=0.15),
        # A duplicated point in another bin, but dominated by all of the
        # above (worse lgradB AND worse AR -- AR farther from 6 than 9.3
        # is 11.7, and we give it very low lgradB so it's dominated).
        _rotating_ellipse_at_ar(11.7, triangularity=0.0),
        _rotating_ellipse_at_ar(11.7, triangularity=0.2),
    ]
    metrics = [
        _make_metrics(6.3, 3.0),
        _make_metrics(6.3, 2.9),
        _make_metrics(9.3, 5.0),
        _make_metrics(9.3, 4.9),
        # Very low lgradB AND high AR -> dominated by all others above.
        _make_metrics(11.7, 0.1),
        _make_metrics(11.7, 0.05),
    ]
    score = mhd_stable_qi_scoring.score_boundaries_diversity(
        boundaries,
        "tight",
        metrics=metrics,
        n_poloidal_points=8,
        n_toroidal_points=8,
    )
    # Compare to score without the dominated bin: should be approximately
    # equal because the HIV filter drops the dominated points.
    score_no_dom = mhd_stable_qi_scoring.score_boundaries_diversity(
        boundaries[:4],
        "tight",
        metrics=metrics[:4],
        n_poloidal_points=8,
        n_toroidal_points=8,
    )
    assert score == pytest.approx(score_no_dom, rel=1e-9)
