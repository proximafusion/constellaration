import pydantic

import constellaration.forward_model as forward_model
import constellaration.optimization.augmented_lagrangian as al


class NevergradSettings(pydantic.BaseModel):
    """Settings for using Nevergrad as an oracle within the augmented
    Lagrangian method."""

    budget_initial: int
    """Initial number of function evaluations."""

    budget_increment: int
    """Number of additional evaluations per iteration."""

    budget_max: int
    """Maximum total number of function evaluations."""

    max_time: float | None
    """Maximum time in seconds for optimization, or None for no limit."""

    num_workers: int
    """Number of parallel workers for function evaluations."""

    batch_mode: bool
    """If True, the next call to ngopt.tell() is delayed until all "num_workers"
    parallel function evaluations have completed. If set to False, tell() is called
    as soon as the first evaluation is available.

    WARNING: For reproducibility when num_workers > 1, batch_mode must be set to True.
    """


class AugmentedLagrangianMethodSettings(pydantic.BaseModel):
    """Settings for the augmented Lagrangian optimization method."""

    maxit: int
    """Maximum number of outer iterations for the optimization."""

    penalty_parameters_initial: float
    """Initial value for the penalty parameter used in the augmented
    Lagrangian formulation."""

    bounds_initial: float
    """Initial bound on constraint violations for setting up the trust region."""

    augmented_lagrangian_settings: al.AugmentedLagrangianSettings
    """Settings specific to the internal augmented Lagrangian solver
    (e.g., tolerances, stopping criteria)."""

    oracle_settings: NevergradSettings
    """Settings for the oracle optimizer (e.g., Nevergrad) used to solve subproblems."""


class ScipyMinimizeSettings(pydantic.BaseModel):
    """Settings for using SciPy's `minimize` function as an optimization backend."""

    options: dict
    """Dictionary of solver-specific options passed to SciPy's `minimize`."""

    method: str
    """The optimization algorithm to use (e.g., 'COBYQA', 'trust-constr')."""


class OptimizationSettings(pydantic.BaseModel):
    """Top-level configuration for the full optimization procedure."""

    optimizer_settings: AugmentedLagrangianMethodSettings | ScipyMinimizeSettings
    """Settings for the chosen optimization algorithm
    (e.g., augmented Lagrangian or SciPy)."""

    forward_model_settings: forward_model.ConstellarationSettings
    """Settings for the forward model used to evaluate the objective and constraints."""

    max_poloidal_mode: int
    """Maximum poloidal mode number used in the spectral representation."""

    max_toroidal_mode: int
    """Maximum toroidal mode number used in the spectral representation."""

    infinity_norm_spectrum_scaling: float
    """Scaling factor for the spectrum when applying the infinity norm
    in the objective or constraints."""
