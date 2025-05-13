from constellaration import forward_model, problems


def test_geometrical_problem_is_feasible() -> None:
    problem = problems.GeometricalProblem()
    feasible_metrics = _get_test_metrics()
    assert problem.is_feasible(feasible_metrics)

    not_feasible_metrics = feasible_metrics.model_copy(update=dict(aspect_ratio=6.0))
    assert not problem.is_feasible(not_feasible_metrics)


def test_geometrical_problem_score() -> None:
    problem = problems.GeometricalProblem()
    feasible_metrics = _get_test_metrics()
    assert problem._score(feasible_metrics) == (
        1.0 - (feasible_metrics.max_elongation - 1.0) / (10.0 - 1.0)
    )


def test_simple_to_build_problem_is_feasible() -> None:
    problem = problems.SimpleToBuildQIStellarator()
    feasible_metrics = _get_test_metrics()
    assert problem.is_feasible(feasible_metrics)

    still_feasible_metrics = feasible_metrics.model_copy(
        update=dict(qi=1e-4 + 0.9 * 1e-6)
    )
    assert problem.is_feasible(still_feasible_metrics)

    not_feasible_metrics = feasible_metrics.model_copy(update=dict(aspect_ratio=11.0))
    assert not problem.is_feasible(not_feasible_metrics)


def test_simple_to_build_problem_score() -> None:
    problem = problems.SimpleToBuildQIStellarator()
    feasible_metrics = _get_test_metrics()
    assert problem._score(feasible_metrics) == (
        feasible_metrics.minimum_normalized_magnetic_gradient_scale_length / 20.0
    )


def test_mhd_stable_problem_is_feasible() -> None:
    problem = problems.MHDStableQIStellarator()
    feasible_metrics = _get_test_metrics()
    assert problem.is_feasible(feasible_metrics)

    not_feasible_metrics = feasible_metrics.model_copy(
        update=dict(flux_compression_in_regions_of_bad_curvature=1.0)
    )
    assert not problem.is_feasible(not_feasible_metrics)


def test_mhd_stable_problem_score_dominated_set() -> None:
    problem = problems.MHDStableQIStellarator()

    candidates: list[forward_model.ConstellarationMetrics] = []
    for aspect_ratio in [30.0, 40.0, 50.0]:
        for minimum_normalized_magnetic_gradient_scale_length in [-1.0, -2.0]:
            metrics = _get_test_metrics().model_copy(
                update=dict(
                    aspect_ratio=aspect_ratio,
                    minimum_normalized_magnetic_gradient_scale_length=minimum_normalized_magnetic_gradient_scale_length,
                )
            )
            candidates.append(metrics)

    assert problem._score(candidates) == 0.0


def test_mhd_stable_problem_score_non_dominated_set() -> None:
    problem = problems.MHDStableQIStellarator()

    candidates: list[forward_model.ConstellarationMetrics] = []
    for aspect_ratio in [5.0, 6.0, 7.0]:
        for minimum_normalized_magnetic_gradient_scale_length in [2.0, 3.0]:
            metrics = _get_test_metrics().model_copy(
                update=dict(
                    aspect_ratio=aspect_ratio,
                    minimum_normalized_magnetic_gradient_scale_length=minimum_normalized_magnetic_gradient_scale_length,
                )
            )
            candidates.append(metrics)

    assert problem._score(candidates) > 0.0


def test_mhd_stable_problem_score_non_dominated_set_infeasible() -> None:
    problem = problems.MHDStableQIStellarator()

    candidates: list[forward_model.ConstellarationMetrics] = []
    for aspect_ratio in [5.0, 6.0, 7.0]:
        for minimum_normalized_magnetic_gradient_scale_length in [2.0, 3.0]:
            metrics = _get_test_metrics().model_copy(
                update=dict(
                    aspect_ratio=aspect_ratio,
                    minimum_normalized_magnetic_gradient_scale_length=minimum_normalized_magnetic_gradient_scale_length,
                    flux_compression_in_regions_of_bad_curvature=1.0,
                )
            )
            candidates.append(metrics)

    assert problem._score(candidates) == 0.0


def _get_test_metrics() -> forward_model.ConstellarationMetrics:
    return forward_model.ConstellarationMetrics(
        aspect_ratio=4.0,
        axis_magnetic_mirror_ratio=0.1,
        aspect_ratio_over_edge_rotational_transform=8.0,
        axis_rotational_transform_over_n_field_periods=0.2,
        average_triangularity=-0.6,
        edge_rotational_transform_over_n_field_periods=0.3,
        max_elongation=4.0,
        qi=1e-5,
        edge_magnetic_mirror_ratio=0.1,
        flux_compression_in_regions_of_bad_curvature=0.8,
        vacuum_well=0.1,
        minimum_normalized_magnetic_gradient_scale_length=0.5,
    )
