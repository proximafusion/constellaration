import numpy as np
import pytest

from constellaration import mhd_stable_qi_scoring


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
