import multiprocessing
import tempfile
import time
from concurrent import futures
from typing import Callable

import jax.numpy as jnp
import nevergrad
import numpy as np
from nevergrad.parametrization import parameter as param

import constellaration.forward_model as forward_model
import constellaration.geometry.surface_rz_fourier as rz_fourier
import constellaration.optimization.augmented_lagrangian as al
import constellaration.optimization.settings as optimization_settings
import constellaration.problems as problems
from constellaration.utils import pytree

pytree.register_pydantic_data(
    rz_fourier.SurfaceRZFourier,
    meta_fields=["n_field_periods", "is_stellarator_symmetric"],
)

NAN_TO_HIGH_VALUE = 10.0


def objective_constraints(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator,
    unravel_and_unmask_fn: Callable[[jnp.ndarray], rz_fourier.SurfaceRZFourier],
    settings: forward_model.ConstellarationSettings,
    aspect_ratio_upper_bound: float | None,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray], forward_model.ConstellarationMetrics | None
]:
    with tempfile.TemporaryDirectory() as _:
        boundary = unravel_and_unmask_fn(jnp.asarray(x * scale))

        metrics = None
        try:
            metrics, _ = forward_model.forward_model(
                boundary=boundary,
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
    boundary: rz_fourier.SurfaceRZFourier,
    settings: optimization_settings.OptimizationSettings,
    problem: problems.MHDStableQIStellarator | problems.SingleObjectiveProblem,
    aspect_ratio_upper_bound: float | None = None,
):
    """Run optimization using the Augmented Lagrangian method.

    Args:
        boundary: The initial surface configuration.
        settings: Optimization settings including AL method parameters.
        problem: The problem to solve, either MHDStableQIStellarator or
            SingleObjectiveProblem.
        aspect_ratio_upper_bound: Optional upper bound for the aspect ratio constraint.

    Returns:
        The optimized surface configuration and optimization metrics.
    """

    assert isinstance(
        settings.optimizer_settings,
        optimization_settings.AugmentedLagrangianMethodSettings,
    )
    boundary = rz_fourier.set_max_mode_numbers(
        surface=boundary,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    )

    mask = rz_fourier.build_mask(
        boundary,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    )

    initial_guess, unravel_and_unmask_fn = pytree.mask_and_ravel(
        pytree=boundary,
        mask=mask,
    )

    scale = rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=boundary.poloidal_modes.flatten(),
        toroidal_modes=boundary.toroidal_modes.flatten(),
        alpha=settings.infinity_norm_spectrum_scaling,
    ).reshape(boundary.poloidal_modes.shape)
    scale = jnp.array(np.concatenate([scale[mask.r_cos], scale[mask.z_sin]]))

    x0 = jnp.array(initial_guess) / scale

    (objective, constraints), _ = objective_constraints(
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

        (objective, constraints), _ = objective_constraints(
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


def _logging(k: int, n_function_evals: int, state: al.AugmentedLagrangianState):
    optimizer_stats = {
        "n_function_evals": n_function_evals,
        "n_gradient_evals": 0,
        "objective": state.objective,
        "feas": jnp.linalg.norm(jnp.maximum(0.0, state.constraints), 2),
    }

    print(k, optimizer_stats)
