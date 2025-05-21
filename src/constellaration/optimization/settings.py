import pydantic

import constellaration.forward_model as forward_model
import constellaration.optimization.augmented_lagrangian as al


class NevergradSettings(pydantic.BaseModel):
    """Settings for Nevergrad optimization.

    Attributes:
        budget_initial: Initial number of function evaluations.
        budget_increment: Number of additional evaluations per iteration.
        budget_max: Maximum total number of function evaluations.
        max_time: Maximum time in seconds for optimization, or None for no limit.
        num_workers: Number of parallel workers for function evaluations.
    """

    budget_initial: int
    budget_increment: int
    budget_max: int
    max_time: float | None
    num_workers: int


class AugmentedLagrangianMethodSettings(pydantic.BaseModel):
    maxit: int
    penalty_parameters_initial: float
    bounds_initial: float
    augmented_lagrangian_settings: al.AugmentedLagrangianSettings
    oracle_settings: NevergradSettings


class ScipyMinimizeSettings(pydantic.BaseModel):
    options: dict
    method: str


class OptimizationSettings(pydantic.BaseModel):
    optimizer_settings: AugmentedLagrangianMethodSettings | ScipyMinimizeSettings
    forward_model_settings: forward_model.ConstellarationSettings
    max_poloidal_mode: int
    max_toroidal_mode: int
    infinity_norm_spectrum_scaling: float
