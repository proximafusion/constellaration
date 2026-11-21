import functools
import multiprocessing
import tempfile
import time
from collections.abc import Callable
from concurrent import futures

import jax
import jax.numpy as jnp
import nevergrad
import numpy as np
from nevergrad.parametrization import parameter as param
from scipy import optimize
from vmecpp import enable_x64  # pyright: ignore[reportAttributeAccessIssue]

import constellaration.geometry.surface_rz_fourier as rz_fourier
import constellaration.optimization.augmented_lagrangian as al
import constellaration.optimization.settings as optimization_settings
from constellaration import differentiable_forward_model, forward_model, problems
from constellaration.mhd import ideal_mhd_parameters_utils, pressure_profile
from constellaration.utils import pytree

NAN_TO_HIGH_VALUE = 10.0


def _unravel_design(
    x: jnp.ndarray,
    unravel_log_design: Callable[[jnp.ndarray], problems.Design],
    optimize_pressure: bool,
    *,
    validate: bool = True,
) -> problems.Design:
    """Restore physical axis pressure from the masked log-pressure coordinates."""
    design = unravel_log_design(x)
    if optimize_pressure and design.pressure is not None:
        p0 = jnp.exp(design.pressure.axis_pressure)
        if validate:
            profile = pressure_profile.IntegratedBernsteinProfile(
                axis_pressure=float(p0),
                gradient_logits=tuple(
                    float(v) for v in design.pressure.gradient_logits
                ),
            )
        else:
            profile = design.pressure.model_copy(update={"axis_pressure": p0})
        design = design.model_copy(update={"pressure": profile})
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

        if isinstance(
            problem,
            (
                problems.GeometricalProblem,
                problems.SimpleToBuildQIStellarator,
                problems.MHDStableQIStellarator,
            ),
        ):
            if metrics is None:
                objective = jnp.array(NAN_TO_HIGH_VALUE)
                constraints = jnp.full(problem.alm_constraint_count, NAN_TO_HIGH_VALUE)
            else:
                value, residuals = problem.alm_objective_constraints(metrics)
                objective, constraints = jnp.array(value), jnp.array(residuals)
        else:
            raise RuntimeError("Problem not supported")

        return ((objective, constraints), metrics)


def run(
    design: problems.Design,
    settings: optimization_settings.OptimizationSettings,
    problem: problems.MHDStableQIStellarator | problems.SingleObjectiveProblem,
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

    Returns:
        The final recommended design and its metrics (None on
        evaluation failure). This does not certify feasibility; use the problem's
        is_feasible method on the returned metrics.
    """

    if not isinstance(
        settings.optimizer_settings,
        optimization_settings.AugmentedLagrangianMethodSettings,
    ):
        raise ValueError("Expected augmented Lagrangian settings")
    if isinstance(
        settings.optimizer_settings.oracle_settings,
        optimization_settings.LBFGSBSettings,
    ):
        with enable_x64(True):
            return _run(design, settings, problem, settings.optimizer_settings)
    return _run(design, settings, problem, settings.optimizer_settings)


def _run(design, settings, problem, alm_settings):
    oracle_settings = alm_settings.oracle_settings
    use_lbfgsb = isinstance(oracle_settings, optimization_settings.LBFGSBSettings)
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

    subproblem = None
    evaluate: Callable[
        [jax.Array],
        tuple[tuple[jax.Array, jax.Array], forward_model.ConstellarationMetrics | None],
    ]
    if use_lbfgsb:
        forward_settings = settings.forward_model_settings
        if not isinstance(
            problem,
            (
                problems.GeometricalProblem,
                problems.SimpleToBuildQIStellarator,
                problems.MHDStableQIStellarator,
            ),
        ):
            raise ValueError("Problem not supported")
        if problem._does_it_require_qi and (
            forward_settings.qi_settings is None
            or forward_settings.boozer_preset_settings is None
        ):
            raise ValueError("This problem requires QI and Boozer settings")
        if isinstance(problem, problems.MHDStableQIStellarator) and (
            forward_settings.turbulent_settings is None
        ):
            raise ValueError("MHD-stable AL requires flux compression settings")
        model = differentiable_forward_model.make_forward_model(
            unravel_and_unmask_fn(x0 * scale), forward_settings
        )

        def evaluate_jax(x):
            design = unravel_and_unmask_fn(x * scale, validate=False)
            metrics, _ = model(design)
            objective, constraints = problem.alm_objective_constraints(metrics)
            return (jnp.asarray(objective), constraints), metrics

        evaluate_jax = jax.jit(evaluate_jax)

        def augmented_value(x, state):
            (objective, constraints), _ = evaluate_jax(x)
            value = al.augmented_lagrangian_function(objective, constraints, state)
            return value, (objective, constraints)

        subproblem = jax.jit(jax.value_and_grad(augmented_value, has_aux=True))

        def evaluate_differentiable(x):
            result, metrics = evaluate_jax(x)
            _require_finite(*result)
            return result, metrics

        evaluate = evaluate_differentiable
    else:
        evaluate = functools.partial(
            objective_constraints,
            scale=scale,
            problem=problem,
            unravel_and_unmask_fn=unravel_and_unmask_fn,
            settings=settings.forward_model_settings,
        )

    (objective, constraints), metrics = evaluate(x0)
    n_function_evals = n_gradient_evals = 0

    state = al.AugmentedLagrangianState(
        x=jnp.copy(x0),
        multipliers=jnp.zeros_like(constraints),
        penalty_parameters=jnp.full_like(
            constraints, alm_settings.penalty_parameters_initial
        ),
        objective=objective,
        constraints=constraints,
        bounds=jnp.ones_like(x0) * alm_settings.bounds_initial,
    )
    _logging(0, n_function_evals, state)

    budget = (
        oracle_settings.budget_initial
        if isinstance(oracle_settings, optimization_settings.NevergradSettings)
        else 0
    )

    for k in range(alm_settings.maxit):
        if isinstance(oracle_settings, optimization_settings.LBFGSBSettings):
            assert subproblem is not None
            x, evaluations = _lbfgsb_step(state, oracle_settings, subproblem)
            n_gradient_evals += evaluations
        else:
            x, evaluations = _nevergrad_step(state, oracle_settings, budget, evaluate)
        n_function_evals += evaluations
        (objective, constraints), metrics = evaluate(x)

        state = al.update_augmented_lagrangian_state(
            x=jnp.copy(x),
            objective=objective,
            constraints=constraints,
            state=state,
            settings=alm_settings.augmented_lagrangian_settings,
        )

        if isinstance(oracle_settings, optimization_settings.NevergradSettings):
            budget = min(
                oracle_settings.budget_max, budget + oracle_settings.budget_increment
            )

        _logging(k + 1, n_function_evals, state, n_gradient_evals)

    if use_lbfgsb:
        assert metrics is not None
        metrics = forward_model.ConstellarationMetrics.model_validate(
            {name: None if value is None else float(value) for name, value in metrics}
        )
    return unravel_and_unmask_fn(state.x * scale), metrics


def _nevergrad_step(state, oracle_settings, budget, evaluate):
    evaluations = 0
    mp_context = multiprocessing.get_context("forkserver")
    parametrization = nevergrad.p.Array(
        init=np.array(state.x),
        lower=np.array(state.x - state.bounds),
        upper=np.array(state.x + state.bounds),
    )
    random_state = np.random.get_state()  # noqa: NPY002
    parametrization.random_state.set_state(random_state)

    oracle = nevergrad.optimizers.NGOpt(
        parametrization=parametrization,
        budget=oracle_settings.budget_initial,
        num_workers=oracle_settings.num_workers,
    )
    oracle.suggest(np.array(state.x))

    t0 = time.time()
    with futures.ProcessPoolExecutor(
        max_workers=oracle_settings.num_workers,
        mp_context=mp_context,
    ) as executor:
        rest_budget = budget

        running_evaluations: list[tuple[futures.Future, param.Parameter]] = []

        while (rest_budget or running_evaluations) and (
            oracle_settings.max_time is None
            or time.time() < t0 + oracle_settings.max_time
        ):
            while len(running_evaluations) < min(
                oracle_settings.num_workers,
                rest_budget,
            ):
                candidate = oracle.ask()

                future = executor.submit(evaluate, jnp.array(candidate.value))
                running_evaluations.append((future, candidate))
                rest_budget -= 1

            # Wait for at least one to complete
            return_when = (
                futures.ALL_COMPLETED
                if oracle_settings.batch_mode
                else futures.FIRST_COMPLETED
            )
            completed, _ = futures.wait(
                [fut for fut, _ in running_evaluations],
                return_when=return_when,
            )

            for future, candidate in running_evaluations:
                if future in completed:
                    evaluations += 1

                    (objective, constraints), _ = future.result()

                    oracle.tell(
                        candidate,
                        al.augmented_lagrangian_function(
                            objective, constraints, state
                        ).item(),
                    )

            # Remove completed from the running list
            running_evaluations = [
                (fut, cand) for fut, cand in running_evaluations if fut not in completed
            ]

        recommendation = oracle.provide_recommendation()
    return recommendation.value, evaluations


def _require_finite(*values):
    if not all(np.all(np.isfinite(value)) for value in values):
        raise FloatingPointError(
            "Non-finite objective, constraints or gradient in the AL subproblem"
        )


def _lbfgsb_step(state, settings, subproblem):
    def value_and_gradient(x):
        (value, (objective, constraints)), gradient = subproblem(jnp.asarray(x), state)
        _require_finite(value, gradient, objective, constraints)
        return float(value), np.asarray(gradient)

    lower, upper = np.asarray(state.x - state.bounds), np.asarray(
        state.x + state.bounds
    )
    bounds = list(zip(lower.tolist(), upper.tolist()))
    result = optimize.minimize(
        value_and_gradient,
        np.asarray(state.x),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": settings.maxiter,
            "maxfun": settings.maxfun,
            "gtol": settings.gtol,
        },
    )
    if not result.success and result.status != 1:  # budget exhaustion is expected
        raise RuntimeError(f"L-BFGS-B subproblem failed: {result.message}")
    return jnp.asarray(result.x), result.nfev


def _logging(
    k: int,
    n_function_evals: int,
    state: al.AugmentedLagrangianState,
    n_gradient_evals: int = 0,
):
    optimizer_stats = {
        "n_function_evals": n_function_evals,
        "n_gradient_evals": n_gradient_evals,
        "objective": state.objective,
        "feas": jnp.linalg.norm(jnp.maximum(0.0, state.constraints), 2),
    }

    print(k, optimizer_stats)
