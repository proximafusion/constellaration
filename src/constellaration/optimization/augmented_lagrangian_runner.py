import functools
import multiprocessing
import tempfile
import time
from collections.abc import Callable
from concurrent import futures

import jax.numpy as jnp
import nevergrad
import numpy as np
from nevergrad.parametrization import parameter as param

import constellaration.geometry.surface_rz_fourier as rz_fourier
import constellaration.optimization.augmented_lagrangian as al
import constellaration.optimization.settings as optimization_settings
from constellaration import forward_model, problems
from constellaration.mhd import ideal_mhd_parameters_utils, pressure_profile
from constellaration.utils import pytree

pytree.register_pydantic_data(
    rz_fourier.SurfaceRZFourier,
    meta_fields=["n_field_periods", "is_stellarator_symmetric"],
)

NAN_TO_HIGH_VALUE = 10.0


pytree.register_pydantic_data(problems.Design)
pytree.register_pydantic_data(pressure_profile.IntegratedBernsteinProfile)


def _unravel_design(
    x: jnp.ndarray,
    unravel_log_design: Callable[[jnp.ndarray], problems.Design],
    optimize_pressure: bool,
) -> problems.Design:
    """Restore physical axis pressure from the masked log-pressure coordinates."""
    design = unravel_log_design(x)
    if optimize_pressure and design.pressure is not None:
        design = design.model_copy(
            update={
                "pressure": pressure_profile.IntegratedBernsteinProfile(
                    axis_pressure=float(np.exp(design.pressure.axis_pressure)),
                    gradient_logits=design.pressure.gradient_logits,
                )
            }
        )
    return design


def _finite_beta_objective_constraints(
    design: problems.Design,
    problem: problems.MaximumBetaQIStellarator
    | problems.SimpleToBuildFiniteBetaQIStellarator,
    settings: forward_model.ConstellarationSettings,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray], forward_model.ConstellarationMetrics | None
]:
    """Evaluate joint candidates using the problem's objective and constraints."""
    assert design.pressure is not None
    try:
        metrics, _ = forward_model.forward_model(
            boundary=design.boundary,
            ideal_mhd_parameters=ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters(
                design
            ),
            settings=settings,
        )
        value, minimize = problem.get_objective(metrics)
        constraints = problem._normalized_constraint_violations(metrics)
        if not np.isfinite(value) or not np.all(np.isfinite(constraints)):
            raise RuntimeError("required metric is missing or nonfinite")
        return (
            (jnp.array(value if minimize else -value), jnp.array(constraints)),
            metrics,
        )
    except (RuntimeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc) not in (
            "Minimum of B is at the boundary.",
            "Not enough crossings found.",
        ):
            raise
        return (
            (
                jnp.array(NAN_TO_HIGH_VALUE),
                jnp.full(problem.n_constraints, NAN_TO_HIGH_VALUE),
            ),
            None,
        )


def objective_constraints(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator,
    unravel_and_unmask_fn: Callable[[jnp.ndarray], problems.Design],
    settings: forward_model.ConstellarationSettings,
    aspect_ratio_upper_bound: float | None,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray], forward_model.ConstellarationMetrics | None
]:
    with tempfile.TemporaryDirectory() as _:
        design = unravel_and_unmask_fn(jnp.asarray(x * scale))
        if isinstance(
            problem,
            (
                problems.MaximumBetaQIStellarator,
                problems.SimpleToBuildFiniteBetaQIStellarator,
            ),
        ):
            return _finite_beta_objective_constraints(design, problem, settings)

        metrics = None
        try:
            metrics, _ = forward_model.forward_model(
                boundary=design.boundary,
                ideal_mhd_parameters=ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters(
                    design
                ),
                settings=settings,
            )
        except Exception as _:
            pass

        if isinstance(problem, problems.GeometricalProblem):
            if metrics is None:
                objective = jnp.array(NAN_TO_HIGH_VALUE)
                constraints = jnp.ones(3) * NAN_TO_HIGH_VALUE
            else:
                objective = jnp.array(metrics.max_elongation)
                constraints = jnp.array(
                    [
                        metrics.aspect_ratio - problem._aspect_ratio_upper_bound,
                        metrics.average_triangularity
                        - problem._average_triangularity_upper_bound,
                        problem._edge_rotational_transform_over_n_field_periods_lower_bound
                        - metrics.edge_rotational_transform_over_n_field_periods,
                    ]
                )
        elif isinstance(problem, problems.SimpleToBuildQIStellarator):
            if metrics is None:
                objective = jnp.array(NAN_TO_HIGH_VALUE)
                constraints = jnp.ones(5) * NAN_TO_HIGH_VALUE
            else:
                objective = jnp.array(
                    20.0 - metrics.minimum_normalized_magnetic_gradient_scale_length
                )
                constraints = jnp.array(
                    [
                        metrics.aspect_ratio - problem._aspect_ratio_upper_bound,
                        problem._edge_rotational_transform_over_n_field_periods_lower_bound
                        - metrics.edge_rotational_transform_over_n_field_periods,
                        jnp.log10(metrics.qi) - problem._log10_qi_upper_bound
                        if metrics.qi is not None
                        else 1.0,
                        metrics.edge_magnetic_mirror_ratio
                        - problem._edge_magnetic_mirror_ratio_upper_bound,
                        metrics.max_elongation - problem._max_elongation_upper_bound,
                    ]
                )
        elif isinstance(problem, problems.MHDStableQIStellarator):
            if metrics is None:
                objective = jnp.array(NAN_TO_HIGH_VALUE)
                constraints = jnp.ones(6) * NAN_TO_HIGH_VALUE
            else:
                assert aspect_ratio_upper_bound is not None
                objective = jnp.array(
                    20.0 - metrics.minimum_normalized_magnetic_gradient_scale_length
                )
                constraints = jnp.array(
                    [
                        metrics.aspect_ratio - aspect_ratio_upper_bound,
                        problem._edge_rotational_transform_over_n_field_periods_lower_bound
                        - metrics.edge_rotational_transform_over_n_field_periods,
                        jnp.log10(metrics.qi) - problem._log10_qi_upper_bound
                        if metrics.qi is not None
                        else 1.0,
                        metrics.edge_magnetic_mirror_ratio
                        - problem._edge_magnetic_mirror_ratio_upper_bound,
                        metrics.flux_compression_in_regions_of_bad_curvature
                        - problem._flux_compression_in_regions_of_bad_curvature_upper_bound  # noqa: E501
                        if metrics.flux_compression_in_regions_of_bad_curvature
                        is not None
                        else 1.0,
                        100 * problem._vacuum_well_lower_bound
                        - 100 * metrics.vacuum_well,
                    ]
                )
        else:
            raise RuntimeError("Problem not supported")

        return ((objective, constraints), metrics)


def run(
    design: problems.Design,
    settings: optimization_settings.OptimizationSettings,
    problem: problems.MHDStableQIStellarator | problems.SingleObjectiveProblem,
    aspect_ratio_upper_bound: float | None = None,
) -> tuple[problems.Design, forward_model.ConstellarationMetrics | None]:
    """Run optimization using the Augmented Lagrangian method.

    Args:
        design: The initial boundary and optional pressure. Finite-beta problems
            optimize log-axis pressure and gradient logits alongside the masked
            Fourier coefficients. Other problems keep supplied pressure fixed.
            With pressure=None, pressure has no optimization DOFs: equilibria use
            p(s) = 1 Pa * (1 - s), zero toroidal current, and edge toroidal flux
            pi * a**2 * 1 T, where s is normalized toroidal flux and a is the
            candidate boundary's minor radius. This is a near-vacuum approximation,
            not exactly zero pressure; finite-beta problems require explicit pressure.
            Pressure coordinate scales are set by settings.log_axis_pressure_scaling
            and settings.pressure_gradient_logits_scaling (both default to 1).
        settings: Optimization settings including AL method parameters.
        problem: The problem to solve, either MHDStableQIStellarator or
            SingleObjectiveProblem.
        aspect_ratio_upper_bound: Optional upper bound for the aspect ratio constraint.

    Returns:
        The final recommended design and its metrics (None on
        evaluation failure). This does not certify feasibility; use the problem's
        is_feasible method on the returned metrics.
    """

    assert isinstance(
        settings.optimizer_settings,
        optimization_settings.AugmentedLagrangianMethodSettings,
    )
    finite_beta = isinstance(
        problem,
        (
            problems.MaximumBetaQIStellarator,
            problems.SimpleToBuildFiniteBetaQIStellarator,
        ),
    )
    if finite_beta and design.pressure is None:
        raise ValueError("finite-beta problems require a design with pressure")
    if finite_beta and (
        settings.forward_model_settings.qi_settings is None
        or settings.forward_model_settings.boozer_preset_settings is None
    ):
        raise ValueError("QI and Boozer settings are required")

    boundary = rz_fourier.set_max_mode_numbers(
        surface=design.boundary,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    )

    mask = rz_fourier.build_mask(
        boundary,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    ).model_copy(update={"r_sin": None, "z_cos": None})

    scale = rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=boundary.poloidal_modes.flatten(),
        toroidal_modes=boundary.toroidal_modes.flatten(),
        alpha=settings.infinity_norm_spectrum_scaling,
    ).reshape(boundary.poloidal_modes.shape)
    scale = jnp.array(np.concatenate([scale[mask.r_cos], scale[mask.z_sin]]))
    pressure = design.pressure
    pressure_mask = None
    if pressure is not None:
        if finite_beta:
            # Only this internal pytree stores log(p0) in the axis-pressure field.
            pressure = pressure.model_copy(
                update={"axis_pressure": float(np.log(pressure.axis_pressure))}
            )
            scale = jnp.concatenate(
                (
                    scale,
                    jnp.array(
                        [
                            settings.log_axis_pressure_scaling,
                            *(
                                [settings.pressure_gradient_logits_scaling]
                                * len(pressure.gradient_logits)
                            ),
                        ]
                    ),
                )
            )
        pressure_mask = pressure.model_copy(
            update={
                "axis_pressure": float(finite_beta),
                "gradient_logits": (float(finite_beta),)
                * len(pressure.gradient_logits),
            }
        )
    initial_guess, unravel_log_design = pytree.mask_and_ravel(
        pytree=problems.Design.model_construct(boundary=boundary, pressure=pressure),
        mask=problems.Design.model_construct(boundary=mask, pressure=pressure_mask),
    )
    unravel_and_unmask_fn = functools.partial(
        _unravel_design,
        unravel_log_design=unravel_log_design,
        optimize_pressure=finite_beta,
    )
    x0 = jnp.array(initial_guess) / scale

    (objective, constraints), metrics = objective_constraints(
        x0,
        scale,
        problem,
        unravel_and_unmask_fn,
        settings.forward_model_settings,
        aspect_ratio_upper_bound,
    )
    n_function_evals = 0

    state = al.AugmentedLagrangianState(
        x=jnp.copy(x0),
        multipliers=jnp.zeros_like(constraints),
        penalty_parameters=jnp.ones_like(constraints),
        objective=objective,
        constraints=constraints,
        bounds=jnp.ones_like(x0) * settings.optimizer_settings.bounds_initial,
    )
    _logging(0, n_function_evals, state)

    mp_context = multiprocessing.get_context("forkserver")

    budget = settings.optimizer_settings.oracle_settings.budget_initial

    for k in range(settings.optimizer_settings.maxit):
        parametrization = nevergrad.p.Array(
            init=np.array(state.x),
            lower=np.array(state.x - state.bounds),
            upper=np.array(state.x + state.bounds),
        )
        random_state = np.random.get_state()  # noqa: NPY002
        parametrization.random_state.set_state(random_state)

        oracle = nevergrad.optimizers.NGOpt(
            parametrization=parametrization,
            budget=settings.optimizer_settings.oracle_settings.budget_initial,
            num_workers=settings.optimizer_settings.oracle_settings.num_workers,
        )
        oracle.suggest(np.array(state.x))

        t0 = time.time()
        with futures.ProcessPoolExecutor(
            max_workers=settings.optimizer_settings.oracle_settings.num_workers,
            mp_context=mp_context,
        ) as executor:
            rest_budget = budget

            running_evaluations: list[tuple[futures.Future, param.Parameter]] = []

            while (rest_budget or running_evaluations) and (
                settings.optimizer_settings.oracle_settings.max_time is None
                or time.time()
                < t0 + settings.optimizer_settings.oracle_settings.max_time
            ):
                while len(running_evaluations) < min(
                    settings.optimizer_settings.oracle_settings.num_workers,
                    rest_budget,
                ):
                    candidate = oracle.ask()

                    future = executor.submit(
                        objective_constraints,
                        jnp.array(candidate.value),
                        scale,
                        problem,
                        unravel_and_unmask_fn,
                        settings.forward_model_settings,
                        aspect_ratio_upper_bound,
                    )
                    running_evaluations.append((future, candidate))
                    rest_budget -= 1

                # Wait for at least one to complete
                return_when = (
                    futures.ALL_COMPLETED
                    if settings.optimizer_settings.oracle_settings.batch_mode
                    else futures.FIRST_COMPLETED
                )
                completed, _ = futures.wait(
                    [fut for fut, _ in running_evaluations],
                    return_when=return_when,
                )

                for future, candidate in running_evaluations:
                    if future in completed:
                        n_function_evals += 1

                        (objective, constraints), _ = future.result()

                        oracle.tell(
                            candidate,
                            al.augmented_lagrangian_function(
                                objective, constraints, state
                            ).item(),
                        )

                # Remove completed from the running list
                running_evaluations = [
                    (fut, cand)
                    for fut, cand in running_evaluations
                    if fut not in completed
                ]

            recommendation = oracle.provide_recommendation()
            x = recommendation.value

        (objective, constraints), metrics = objective_constraints(
            x,
            scale,
            problem,
            unravel_and_unmask_fn,
            settings.forward_model_settings,
            aspect_ratio_upper_bound,
        )

        state = al.update_augmented_lagrangian_state(
            x=jnp.copy(x),
            objective=objective,
            constraints=constraints,
            state=state,
            settings=settings.optimizer_settings.augmented_lagrangian_settings,
        )

        budget = int(
            jnp.minimum(
                settings.optimizer_settings.oracle_settings.budget_max,
                budget + settings.optimizer_settings.oracle_settings.budget_increment,
            )
        )

        _logging(k + 1, n_function_evals, state)

    return unravel_and_unmask_fn(state.x * scale), metrics


def _logging(k: int, n_function_evals: int, state: al.AugmentedLagrangianState):
    optimizer_stats = {
        "n_function_evals": n_function_evals,
        "n_gradient_evals": 0,
        "objective": state.objective,
        "feas": jnp.linalg.norm(jnp.maximum(0.0, state.constraints), 2),
    }

    print(k, optimizer_stats)
