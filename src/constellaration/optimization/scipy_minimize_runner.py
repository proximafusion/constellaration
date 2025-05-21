import functools
import tempfile
from typing import Callable

import jax.numpy as jnp
import numpy as np
from scipy import optimize

import constellaration.forward_model as forward_model
import constellaration.geometry.surface_rz_fourier as rz_fourier
import constellaration.optimization.settings as optimization_settings
import constellaration.problems as problems
from constellaration.utils import pytree

pytree.register_pydantic_data(
    rz_fourier.SurfaceRZFourier,
    meta_fields=["n_field_periods", "is_stellarator_symmetric"],
)

NAN_TO_HIGH_VALUE = 10.0


global n_function_evals
n_function_evals = 0


def objective_fun(
    x: jnp.ndarray,
    forward_model: Callable[[np.ndarray], forward_model.ConstellarationMetrics | None],
    problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator,
) -> jnp.ndarray:
    global n_function_evals
    n_function_evals += 1
    metrics = forward_model(np.asarray(x))
    if isinstance(problem, problems.GeometricalProblem):
        if metrics is None:
            objective = jnp.array(NAN_TO_HIGH_VALUE)
        else:
            objective = jnp.array(metrics.max_elongation)
    elif isinstance(problem, problems.SimpleToBuildQIStellarator):
        if metrics is None:
            objective = jnp.array(NAN_TO_HIGH_VALUE)
        else:
            objective = jnp.array(
                20.0 - metrics.minimum_normalized_magnetic_gradient_scale_length
            )
    elif isinstance(problem, problems.MHDStableQIStellarator):
        if metrics is None:
            objective = jnp.array(NAN_TO_HIGH_VALUE)
        else:
            objective = jnp.array(
                20.0 - metrics.minimum_normalized_magnetic_gradient_scale_length
            )
    else:
        raise RuntimeError("Problem not supported")

    return objective


def constraints_fun(
    x: jnp.ndarray,
    forward_model: Callable[[np.ndarray], forward_model.ConstellarationMetrics | None],
    problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator,
    aspect_ratio_upper_bound: float | None,
) -> jnp.ndarray:
    metrics = forward_model(np.asarray(x))
    if isinstance(problem, problems.GeometricalProblem):
        if metrics is None:
            constraints = jnp.ones(3) * NAN_TO_HIGH_VALUE
        else:
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
            constraints = jnp.ones(5) * NAN_TO_HIGH_VALUE
        else:
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
            constraints = jnp.ones(6) * NAN_TO_HIGH_VALUE
        else:
            assert aspect_ratio_upper_bound is not None
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
                    if metrics.flux_compression_in_regions_of_bad_curvature is not None
                    else 1.0,
                    100 * problem._vacuum_well_lower_bound - 100 * metrics.vacuum_well,
                ]
            )
    else:
        raise RuntimeError("Problem not supported")

    return constraints


def run(
    boundary: rz_fourier.SurfaceRZFourier,
    settings: optimization_settings.OptimizationSettings,
    problem: problems.MHDStableQIStellarator | problems.SingleObjectiveProblem,
    aspect_ratio_upper_bound: float | None = None,
):
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

    fm = functools.partial(
        _forward_model,
        scale=scale,
        unravel_and_unmask_fn=unravel_and_unmask_fn,
        settings=settings.forward_model_settings,
    )

    objective = objective_fun(x=x0, forward_model=fm, problem=problem)

    constraints = constraints_fun(
        x=x0,
        forward_model=fm,
        problem=problem,
        aspect_ratio_upper_bound=aspect_ratio_upper_bound,
    )

    _logging(n_function_evals, objective, constraints)

    def callback(x, res=None):  # noqa: ARG001
        objective = objective_fun(x=x, forward_model=fm, problem=problem)

        constraints = constraints_fun(
            x=x,
            forward_model=fm,
            problem=problem,
            aspect_ratio_upper_bound=aspect_ratio_upper_bound,
        )

        _logging(n_function_evals, objective, constraints)

    # convert self.problem.constraints into scipy format
    constraints = []

    constraints.append(
        optimize.NonlinearConstraint(
            fun=lambda x: constraints_fun(
                x=x,
                forward_model=fm,
                problem=problem,
                aspect_ratio_upper_bound=aspect_ratio_upper_bound,
            ),
            ub=0,
            lb=-jnp.inf,
        )
    )

    assert isinstance(
        settings.optimizer_settings, optimization_settings.ScipyMinimizeSettings
    )
    optimize.minimize(
        fun=lambda x: objective_fun(
            x=x,
            forward_model=fm,
            problem=problem,
        ),
        x0=x0,
        method=settings.optimizer_settings.method,
        options=settings.optimizer_settings.options,
        constraints=constraints,
        callback=callback,
    )


def _forward_model(
    x: np.ndarray,
    scale: jnp.ndarray,
    unravel_and_unmask_fn: Callable[[jnp.ndarray], rz_fourier.SurfaceRZFourier],
    settings: (forward_model.ConstellarationSettings),
) -> forward_model.ConstellarationMetrics | None:
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

        return metrics


def _logging(n_function_evals: int, objective: jnp.ndarray, constraints: jnp.ndarray):
    optimizer_stats = {
        "n_function_evals": n_function_evals,
        "n_gradient_evals": 0,
        "objective": objective,
        "feas": jnp.linalg.norm(jnp.maximum(0.0, constraints), 2),
    }

    print(optimizer_stats)
