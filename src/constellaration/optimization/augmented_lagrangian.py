import jax.numpy as jnp
import pydantic


class AugmentedLagrangianState(pydantic.BaseModel, arbitrary_types_allowed=True):
    x: jnp.ndarray
    multipliers: jnp.ndarray
    penalty_parameters: jnp.ndarray
    objective: jnp.ndarray
    constraints: jnp.ndarray
    bounds: jnp.ndarray


class AugmentedLagrangianSettings(pydantic.BaseModel):
    constraint_violation_tolerance_reduction_factor: float = 0.5
    """Decrement constraint violation."""

    penalty_parameters_increase_factor: float = 2
    """Increment penalty parameter."""

    bounds_reduction_factor: float = 0.95
    """Decrement trust region bounds."""

    penalty_parameters_max: float = 1e8
    """Upper cutoff for penalty parameter."""

    bounds_min: float = 0.05
    """Lower cutoff for trust region bounds."""


def augmented_lagrangian_function(
    objective: jnp.ndarray, constraints: jnp.ndarray, state: AugmentedLagrangianState
) -> jnp.ndarray:
    """Updates the augmented Lagrangian state based on the current optimization state.

    This function updates the Lagrange multipliers and penalty parameters based on
    the current constraints violation and settings.

    Args:
        x: Current point in the optimization space.
        objective: Current objective function value.
        constraints: Current constraint values.
        state: Current augmented Lagrangian state.
        settings: Settings for the augmented Lagrangian method.
        penalty_parameters: Optional custom penalty parameters. If None, they will be
        updated based on constraint violations.
        bounds: Optional custom bounds. If None, they will be updated based on settings.

    Returns:
        Updated augmented Lagrangian state.
    """
    value = objective + jnp.sum(
        0.5
        * state.penalty_parameters
        * (
            jnp.maximum(
                0.0,
                state.multipliers / state.penalty_parameters + constraints,
            )
            ** 2
            - (state.multipliers / state.penalty_parameters) ** 2
        )
    )
    return value


def update_augmented_lagrangian_state(
    x: jnp.ndarray,
    objective: jnp.ndarray,
    constraints: jnp.ndarray,
    state: AugmentedLagrangianState,
    settings: AugmentedLagrangianSettings,
    penalty_parameters: jnp.ndarray | None = None,
    bounds: jnp.ndarray | None = None,
) -> AugmentedLagrangianState:
    """Updates the augmented Lagrangian state based on the current optimization state.

    This function updates the Lagrange multipliers and penalty parameters based on
    the current constraints violation and settings.

    Args:
      x: Current point in the optimization space.
      objective: Current objective function value.
      constraints: Current constraint values.
      state: Current augmented Lagrangian state.
      settings: Settings for the augmented Lagrangian method.
      penalty_parameters: Optional custom penalty parameters. If None, they will be
        updated based on constraint violations.
      bounds: Optional custom bounds. If None, they will be updated based on settings.

    Returns:
      Updated augmented Lagrangian state.
    """
    multipliers = jnp.maximum(
        0.0,
        state.multipliers + state.penalty_parameters * constraints,
    )

    if penalty_parameters is None:
        penalty_parameters = jnp.where(
            jnp.maximum(0.0, constraints)
            > settings.constraint_violation_tolerance_reduction_factor
            * jnp.maximum(0.0, state.constraints),
            jnp.minimum(
                settings.penalty_parameters_max,
                settings.penalty_parameters_increase_factor * state.penalty_parameters,
            ),
            state.penalty_parameters,
        )

    if bounds is None:
        bounds = jnp.maximum(settings.bounds_min, state.bounds)

    return AugmentedLagrangianState(
        x=x,
        multipliers=multipliers,
        penalty_parameters=penalty_parameters,
        objective=objective,
        constraints=constraints,
        bounds=bounds,
    )
