import functools
import pickle
from concurrent.futures import ThreadPoolExecutor

import jax.numpy as jnp
import numpy as np
import pydantic
import pytest

from constellaration import forward_model, initial_guess, problems
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import ideal_mhd_parameters_utils, pressure_profile
from constellaration.optimization import (
    augmented_lagrangian as al,
)
from constellaration.optimization import (
    augmented_lagrangian_runner as runner,
)
from constellaration.optimization import (
    settings as optimization_settings,
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
        average_triangularity=-1,
        vacuum_well=0.1,
        minimum_normalized_magnetic_gradient_scale_length=2,
        qi=1e-5,
        minimum_normalized_mercier_criterion=0.01,
        magnetic_mirror_ratio_well=0.04,
        volume_averaged_beta=0.03,
        beta_peaking_factor=3,
        flux_compression_in_regions_of_bad_curvature=0.1,
    )


@pytest.fixture
def boundary():
    return initial_guess.generate_rotating_ellipse(
        aspect_ratio=6, elongation=1.5, rotational_transform=0.4, n_field_periods=4
    )


@pytest.fixture
def settings():
    return optimization_settings.OptimizationSettings(
        max_poloidal_mode=1,
        max_toroidal_mode=1,
        infinity_norm_spectrum_scaling=1.5,
        optimizer_settings=optimization_settings.AugmentedLagrangianMethodSettings(
            maxit=1,
            penalty_parameters_initial=10,
            bounds_initial=0.05,
            augmented_lagrangian_settings=al.AugmentedLagrangianSettings(),
            oracle_settings=optimization_settings.NevergradSettings(
                budget_initial=6,
                budget_max=6,
                budget_increment=0,
                max_time=None,
                num_workers=1,
                batch_mode=True,
            ),
        ),
        forward_model_settings=forward_model.ConstellarationSettings(
            turbulent_settings=None
        ),
    )


def _thread_pool(max_workers, mp_context):
    # Keep patches visible to workers in unit tests.
    del mp_context
    return ThreadPoolExecutor(max_workers=max_workers)


@pytest.mark.parametrize(
    "logits", [(), (1.0,), (0.0, 0.0, 0.0), (1.0, -1.0, 0.0, 2.0, -2.0)]
)
@pytest.mark.parametrize(
    "problem_type",
    [problems.MaximumBetaQIStellarator, problems.SimpleToBuildFiniteBetaQIStellarator],
)
def test_joint_al_runner(
    monkeypatch, boundary, settings, metrics, logits, problem_type
):
    candidate = problems.Design(
        boundary=boundary,
        pressure=pressure_profile.IntegratedBernsteinProfile(
            axis_pressure=1000, gradient_logits=logits
        ),
    )
    budget = max(16, 12 + len(logits))
    settings.optimizer_settings.oracle_settings.budget_initial = budget
    settings.optimizer_settings.oracle_settings.budget_max = budget
    calls = []
    designs = []
    to_mhd_parameters = ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters

    def evaluate(boundary, ideal_mhd_parameters, settings):
        assert settings is not None
        assert ideal_mhd_parameters.toroidal_current.coefficients == [0.0]
        calls.append((boundary, ideal_mhd_parameters))
        if len(calls) == 2:
            raise RuntimeError("VMEC++ did not converge")
        return metrics, None

    def to_inputs(design):
        designs.append(design)
        return to_mhd_parameters(design)

    monkeypatch.setattr(forward_model, "forward_model", evaluate)
    monkeypatch.setattr(runner.futures, "ProcessPoolExecutor", _thread_pool)
    np.random.seed(123)  # noqa: NPY002
    monkeypatch.setattr(
        ideal_mhd_parameters_utils, "design_to_ideal_mhd_parameters", to_inputs
    )
    recommended, result_metrics = runner.run(candidate, settings, problem_type())
    assert isinstance(recommended, problems.Design)
    assert result_metrics == metrics
    assert len(calls) == len(designs) == budget + 2
    assert any(not np.array_equal(b.r_cos, boundary.r_cos) for b, _ in calls)
    assert any(
        c.pressure.axis_pressure != designs[0].pressure.axis_pressure for c in designs
    )
    if logits:
        assert any(
            c.pressure.gradient_logits != designs[0].pressure.gradient_logits
            for c in designs
        )
    for (b, mhd), design in zip(calls, designs):
        assert len(design.pressure.gradient_logits) == len(logits)
        assert np.array_equal(b.r_cos, design.boundary.r_cos)
        expected = to_mhd_parameters(design)
        assert mhd.pressure.coefficients == pytest.approx(
            expected.pressure.coefficients
        )
        assert mhd.boundary_toroidal_flux == pytest.approx(
            expected.boundary_toroidal_flux
        )
        assert (
            problems.Design.model_validate_json(design.model_dump_json()).pressure
            == design.pressure
        )


@pytest.mark.parametrize(
    ("problem_type", "expected_objective", "count"),
    [
        (problems.MaximumBetaQIStellarator, -0.03, 7),
        (problems.SimpleToBuildFiniteBetaQIStellarator, -2.0, 10),
    ],
)
def test_finite_beta_objective_constraints(
    monkeypatch, boundary, settings, metrics, problem_type, expected_objective, count
):
    problem = problem_type()
    candidate = problems.Design(
        boundary=boundary,
        pressure=pressure_profile.IntegratedBernsteinProfile(axis_pressure=1000),
    )
    monkeypatch.setattr(forward_model, "forward_model", lambda **_: (metrics, None))
    (objective, constraints), actual_metrics = runner.objective_constraints(
        jnp.zeros(1),
        jnp.ones(1),
        problem,
        lambda _: candidate,
        settings.forward_model_settings,
        None,
    )
    assert objective == pytest.approx(expected_objective)
    assert actual_metrics == metrics
    assert len(constraints) == count
    np.testing.assert_allclose(
        constraints,
        problem._normalized_constraint_violations(metrics),
        rtol=1e-6,
    )

    def fail(**_):
        raise RuntimeError("VMEC++ did not converge")

    monkeypatch.setattr(forward_model, "forward_model", fail)
    (objective, constraints), actual_metrics = runner.objective_constraints(
        jnp.zeros(1),
        jnp.ones(1),
        problem,
        lambda _: candidate,
        settings.forward_model_settings,
        None,
    )
    assert actual_metrics is None
    assert objective == runner.NAN_TO_HIGH_VALUE
    np.testing.assert_array_equal(constraints, np.full(count, runner.NAN_TO_HIGH_VALUE))

    incomplete = metrics.model_copy(update={"volume_averaged_beta": None})
    monkeypatch.setattr(forward_model, "forward_model", lambda **_: (incomplete, None))
    (objective, constraints), actual_metrics = runner.objective_constraints(
        jnp.zeros(1),
        jnp.ones(1),
        problem,
        lambda _: candidate,
        settings.forward_model_settings,
        None,
    )
    assert actual_metrics is None
    assert objective == runner.NAN_TO_HIGH_VALUE
    np.testing.assert_array_equal(constraints, np.full(count, runner.NAN_TO_HIGH_VALUE))


def test_joint_decoder_is_picklable(boundary):
    design = problems.Design(
        boundary=boundary,
        pressure=pressure_profile.IntegratedBernsteinProfile(
            axis_pressure=np.log(1000), gradient_logits=(1, 2)
        ),
    )
    mask = problems.Design.model_construct(
        boundary=surface_rz_fourier.build_mask(boundary, 1, 1).model_copy(
            update={"r_sin": None, "z_cos": None}
        ),
        pressure=design.pressure.model_copy(
            update={"axis_pressure": 1.0, "gradient_logits": (1.0, 1.0)}
        ),
    )
    flat, unravel = runner.pytree.mask_and_ravel(design, mask)
    decode = functools.partial(
        runner._unravel_design, unravel_log_design=unravel, optimize_pressure=True
    )
    decode = pickle.loads(pickle.dumps(decode))
    candidate = decode(flat)
    assert candidate.pressure.axis_pressure == pytest.approx(1000, rel=1e-6)
    assert candidate.pressure.gradient_logits == (1, 2)
    np.testing.assert_array_equal(candidate.boundary.r_cos, boundary.r_cos)


@pytest.mark.parametrize(
    ("problem_type", "count", "objective"),
    [
        (problems.GeometricalProblem, 3, 2),
        (problems.SimpleToBuildQIStellarator, 5, 18),
        (problems.MHDStableQIStellarator, 6, 18),
    ],
)
@pytest.mark.parametrize("fixed_pressure", [False, True])
def test_boundary_only_runner(
    monkeypatch,
    boundary,
    settings,
    metrics,
    problem_type,
    count,
    objective,
    fixed_pressure,
):
    pressure = (
        pressure_profile.IntegratedBernsteinProfile(
            axis_pressure=1000, gradient_logits=(1, 2)
        )
        if fixed_pressure
        else None
    )
    design = problems.Design(boundary=boundary, pressure=pressure)
    evaluations = []
    to_mhd_parameters = ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters

    def evaluate(boundary, settings, ideal_mhd_parameters):
        assert isinstance(boundary, surface_rz_fourier.SurfaceRZFourier)
        assert settings is not None
        if pressure is None:
            assert ideal_mhd_parameters.pressure.coefficients == [1.0, -1.0]
            assert ideal_mhd_parameters.toroidal_current.coefficients == [0.0]
            area = np.pi * surface_rz_fourier.evaluate_minor_radius(boundary) ** 2
            assert ideal_mhd_parameters.boundary_toroidal_flux == pytest.approx(area)
        else:
            expected = pressure_profile.to_power_series(pressure)
            assert ideal_mhd_parameters.pressure.coefficients == pytest.approx(
                expected.coefficients
            )
        return metrics, None

    def to_inputs(candidate):
        assert candidate.pressure == pressure
        evaluations.append(candidate)
        return to_mhd_parameters(candidate)

    monkeypatch.setattr(forward_model, "forward_model", evaluate)
    monkeypatch.setattr(runner.futures, "ProcessPoolExecutor", _thread_pool)
    problem = problem_type()
    (actual, constraints), _ = runner.objective_constraints(
        jnp.zeros(1),
        jnp.ones(1),
        problem,
        lambda _: design,
        settings.forward_model_settings,
        10,
    )
    assert actual == objective
    assert len(constraints) == count
    monkeypatch.setattr(
        ideal_mhd_parameters_utils, "design_to_ideal_mhd_parameters", to_inputs
    )
    recommended, actual_metrics = runner.run(
        design=design,
        settings=settings,
        problem=problem,
        aspect_ratio_upper_bound=10,
    )
    assert isinstance(recommended, problems.Design)
    assert recommended.pressure == pressure
    assert len(evaluations) == 8
    assert any(
        not np.array_equal(c.boundary.r_cos, boundary.r_cos) for c in evaluations
    )
    assert actual_metrics == metrics


@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.3, 0.7)])
@pytest.mark.parametrize("logits", [(), (1.0,), (1.0, -2.0, 3.0)])
def test_pressure_parameter_scaling(
    monkeypatch, boundary, settings, metrics, scales, logits
):
    assert settings.log_axis_pressure_scaling == 1.0
    assert settings.pressure_gradient_logits_scaling == 1.0
    settings.log_axis_pressure_scaling, settings.pressure_gradient_logits_scaling = (
        scales
    )
    settings.optimizer_settings.maxit = 0
    design = problems.Design(
        boundary=boundary,
        pressure=pressure_profile.IntegratedBernsteinProfile(
            axis_pressure=1000, gradient_logits=logits
        ),
    )
    observed = []

    def evaluate(x, scale, problem, unravel, _settings, _aspect_bound):
        observed.append((np.asarray(x), np.asarray(scale)))
        candidate = unravel(x * scale)
        assert candidate.pressure.axis_pressure == pytest.approx(1000)
        assert candidate.pressure.gradient_logits == pytest.approx(logits)
        return ((jnp.array(0.0), jnp.zeros(problem.n_constraints)), metrics)

    monkeypatch.setattr(runner, "objective_constraints", evaluate)
    recommended, result_metrics = runner.run(
        design, settings, problems.MaximumBetaQIStellarator()
    )
    assert result_metrics == metrics
    assert recommended.pressure.axis_pressure == pytest.approx(1000)
    x, scale = observed[0]
    count = 1 + len(logits)
    expected_scales = np.array([scales[0], *([scales[1]] * len(logits))])
    np.testing.assert_array_equal(scale[-count:], expected_scales)
    np.testing.assert_allclose(
        x[-count:], np.array([np.log(1000), *logits]) / expected_scales
    )
    restored = optimization_settings.OptimizationSettings.model_validate_json(
        settings.model_dump_json()
    )
    assert restored.log_axis_pressure_scaling == scales[0]
    assert restored.pressure_gradient_logits_scaling == scales[1]


@pytest.mark.parametrize(
    "field", ["log_axis_pressure_scaling", "pressure_gradient_logits_scaling"]
)
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_invalid_pressure_parameter_scaling(settings, field, value):
    data = settings.model_dump()
    data[field] = value
    with pytest.raises(pydantic.ValidationError):
        optimization_settings.OptimizationSettings.model_validate(data)


def test_finite_beta_input_validation(boundary, settings):
    problem = problems.MaximumBetaQIStellarator()
    with pytest.raises(ValueError, match="design with pressure"):
        runner.run(problems.Design(boundary=boundary), settings, problem)
    candidate = problems.Design(
        boundary=boundary,
        pressure=pressure_profile.IntegratedBernsteinProfile(axis_pressure=1000),
    )
    settings.forward_model_settings.qi_settings = None
    with pytest.raises(ValueError, match="QI and Boozer"):
        runner.run(candidate, settings, problem)
