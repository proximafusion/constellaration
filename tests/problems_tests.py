import numpy as np
import pytest
from scipy.constants import mu_0

from constellaration import forward_model, initial_guess, problems
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import (
    ideal_mhd_parameters_utils,
    magnetics_utils,
    pressure_profile,
)


@pytest.fixture
def metrics():
    return forward_model.ConstellarationMetrics(
        aspect_ratio=4,
        aspect_ratio_over_edge_rotational_transform=4,
        max_elongation=2,
        axis_rotational_transform_over_n_field_periods=0.3,
        edge_rotational_transform_over_n_field_periods=0.3,
        axis_magnetic_mirror_ratio=0.1,
        edge_magnetic_mirror_ratio=0.1,
        average_triangularity=0,
        vacuum_well=0,
        minimum_normalized_magnetic_gradient_scale_length=2,
        qi=1e-5,
        minimum_normalized_mercier_criterion=0.01,
        magnetic_mirror_ratio_well=0.04,
        volume_averaged_beta=0.03,
        beta_peaking_factor=3,
    )


def test_maximum_beta_problem(metrics):
    problem = problems.MaximumBetaQIStellarator()
    assert problem.is_feasible(metrics)
    assert problem.get_objective(metrics) == (0.03, False)
    for name, value in [
        ("aspect_ratio", 12),
        ("edge_magnetic_mirror_ratio", 0.201),
        ("edge_magnetic_mirror_ratio", np.nan),
        ("qi", 1.01e-4),
        ("edge_rotational_transform_over_n_field_periods", 0.224),
        ("max_elongation", 7.01),
        ("minimum_normalized_mercier_criterion", 0),
        ("magnetic_mirror_ratio_well", 0.05),
        ("qi", None),
        ("volume_averaged_beta", None),
    ]:
        assert not problem.is_feasible(metrics.model_copy(update={name: value})), name
    unconstrained = metrics.model_copy(
        update={
            "volume_averaged_beta": 0.001,
            "beta_peaking_factor": None,
        }
    )
    assert problem.is_feasible(unconstrained)
    assert problem.is_feasible(
        metrics.model_copy(update={"edge_magnetic_mirror_ratio": 0.2})
    )
    for transform in (0.225, -0.225):
        assert problem.is_feasible(
            metrics.model_copy(
                update={"edge_rotational_transform_over_n_field_periods": transform}
            )
        )
    assert problem._score(metrics) > problem._score(unconstrained)
    assert (
        problem.n_constraints
        == len(problem._normalized_constraint_violations(metrics))
        == 7
    )


def test_simple_to_build_finite_beta_problem(metrics):
    problem = problems.SimpleToBuildFiniteBetaQIStellarator()
    assert problem.is_feasible(metrics)
    assert problem.get_objective(metrics) == (2, False)
    for name, value in [
        ("aspect_ratio", 5.001),
        ("qi", 1.01e-4),
        ("edge_rotational_transform_over_n_field_periods", 0.224),
        ("edge_magnetic_mirror_ratio", 0.201),
        ("max_elongation", 7.01),
        ("minimum_normalized_mercier_criterion", 0),
        ("magnetic_mirror_ratio_well", 0.05),
        ("volume_averaged_beta", 0.02999),
        ("beta_peaking_factor", 2.899),
        ("beta_peaking_factor", 3.101),
        ("beta_peaking_factor", np.nan),
    ]:
        assert not problem.is_feasible(metrics.model_copy(update={name: value})), name
    for name, value in [
        ("aspect_ratio", 5),
        ("edge_rotational_transform_over_n_field_periods", 0.225),
        ("edge_rotational_transform_over_n_field_periods", -0.225),
        ("beta_peaking_factor", 2.9),
        ("beta_peaking_factor", 3.1),
        ("max_elongation", 7),
        ("edge_magnetic_mirror_ratio", 0.2),
        ("qi", 1e-4),
    ]:
        assert problem.is_feasible(metrics.model_copy(update={name: value})), name
    higher_beta = metrics.model_copy(update={"volume_averaged_beta": 0.04})
    assert problem.get_objective(higher_beta) == problem.get_objective(metrics)
    assert problem._score(higher_beta) == problem._score(metrics)
    assert (
        problem.n_constraints
        == len(problem._normalized_constraint_violations(metrics))
        == 10
    )


@pytest.mark.parametrize(
    ("problem_type", "count"),
    [
        (problems.MaximumBetaQIStellarator, 7),
        (problems.SimpleToBuildFiniteBetaQIStellarator, 10),
    ],
)
def test_missing_or_invalid_common_metrics_are_infeasible(metrics, problem_type, count):
    problem = problem_type()
    for name, value in [
        ("qi", None),
        ("qi", -1),
        ("qi", np.nan),
        ("volume_averaged_beta", None),
        ("minimum_normalized_mercier_criterion", np.nan),
        ("magnetic_mirror_ratio_well", None),
    ]:
        invalid = metrics.model_copy(update={name: value})
        assert np.array_equal(
            problem._normalized_constraint_violations(invalid), np.full(count, np.inf)
        )
        assert not problem.is_feasible(invalid)
    assert problem.is_feasible(metrics.model_copy(update={"qi": 0}))


def test_candidate_pressure_reaches_forward_model(monkeypatch, metrics):
    boundary = initial_guess.generate_rotating_ellipse(
        aspect_ratio=6, elongation=1.5, rotational_transform=0.4, n_field_periods=4
    )
    pressure = pressure_profile.IntegratedBernsteinProfile(
        axis_pressure=1000, gradient_logits=(1, 2, 3)
    )
    candidate = problems.Design(boundary=boundary, pressure=pressure)

    def evaluate(boundary, ideal_mhd_parameters, settings):
        assert boundary is candidate.boundary
        assert ideal_mhd_parameters.pressure == pressure_profile.to_power_series(
            pressure
        )
        assert ideal_mhd_parameters.boundary_toroidal_flux == pytest.approx(
            ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters(
                candidate
            ).boundary_toroidal_flux
        )
        assert ideal_mhd_parameters.toroidal_current.coefficients == [0.0]
        assert settings.qi_settings is not None
        return metrics, None

    monkeypatch.setattr(forward_model, "forward_model", evaluate)
    for problem in (
        problems.MaximumBetaQIStellarator(),
        problems.SimpleToBuildFiniteBetaQIStellarator(),
    ):
        assert problem.evaluate(candidate).score > 0


@pytest.mark.parametrize(
    "problem_type",
    [problems.MaximumBetaQIStellarator, problems.SimpleToBuildFiniteBetaQIStellarator],
)
def test_finite_beta_requires_explicit_pressure(problem_type, monkeypatch):
    boundary = initial_guess.generate_rotating_ellipse(
        aspect_ratio=6, elongation=1.5, rotational_transform=0.4, n_field_periods=4
    )

    def unexpected_evaluation(*_args, **_kwargs):
        pytest.fail("missing pressure should be rejected before evaluating VMEC")

    monkeypatch.setattr(forward_model, "forward_model", unexpected_evaluation)
    with pytest.raises(ValueError, match="design with pressure"):
        problem_type().evaluate(problems.Design(boundary=boundary))


def test_design_without_pressure_uses_near_vacuum_inputs():
    boundary = initial_guess.generate_rotating_ellipse(
        aspect_ratio=6, elongation=1.5, rotational_transform=0.4, n_field_periods=4
    )
    design = problems.Design(boundary=boundary)
    assert (
        problems.Design.model_validate_json(design.model_dump_json()).pressure is None
    )
    mhd = ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters(design)
    assert mhd.pressure.coefficients == [1.0, -1.0]
    assert mhd.toroidal_current.coefficients == [0.0]
    area = np.pi * surface_rz_fourier.evaluate_minor_radius(boundary) ** 2
    assert mhd.boundary_toroidal_flux == pytest.approx(area)


def test_edge_flux_estimate_tracks_geometry_and_pressure():
    boundary = initial_guess.generate_rotating_ellipse(
        aspect_ratio=6, elongation=1.5, rotational_transform=0.4, n_field_periods=4
    )
    area = np.pi * surface_rz_fourier.evaluate_minor_radius(boundary) ** 2

    def flux(p0, logits=(0.0, 0.0, 0.0)):
        candidate = problems.Design(
            boundary=boundary,
            pressure=pressure_profile.IntegratedBernsteinProfile(
                axis_pressure=p0, gradient_logits=logits
            ),
        )
        return ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters(
            candidate
        ).boundary_toroidal_flux

    # Uniform Bernstein weights produce p(s) = p0*(1-s).
    assert flux(1e-6) == pytest.approx(area, rel=1e-10)
    assert flux(10000) == pytest.approx(area * (1 + mu_0 * 5000))
    assert flux(10000, (5, -5, -5)) > flux(10000)


def test_derived_flux_gives_approximately_one_tesla_in_vmec():
    boundary = initial_guess.generate_nae(
        aspect_ratio=10,
        max_elongation=5,
        rotational_transform=0.75,
        mirror_ratio=0.25,
        n_field_periods=3,
        max_poloidal_mode=1,
        max_toroidal_mode=1,
    )
    candidate = problems.Design(
        boundary=boundary,
        pressure=pressure_profile.IntegratedBernsteinProfile(
            axis_pressure=3000, gradient_logits=(2, 0, -1)
        ),
    )
    _, wout = forward_model.forward_model(
        boundary,
        ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters(candidate),
        forward_model.ConstellarationSettings(
            qi_settings=None, boozer_preset_settings=None, turbulent_settings=None
        ),
    )
    axis_field = magnetics_utils._magnetic_field_strength_nyquist_resolution(wout)[0]
    assert np.mean(axis_field) == pytest.approx(1.0, rel=0.2)
