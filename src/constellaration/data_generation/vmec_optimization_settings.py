import pydantic


class OmnigenousFieldVmecOptimizationSettings(pydantic.BaseModel):
    infinity_norm_spectrum_scaling: float = 1.5
    """The scaling factor for the infinity norm spectrum to use during optimization."""

    max_poloidal_mode: int = 1
    """The maximum poloidal mode number of the resulting configuration."""

    max_toroidal_mode: int = 1
    """The maximum toroidal mode number of the resulting configuration."""

    n_inner_optimizations: int = 1
    """The number of inner optimization to perform to obtain the final configuration."""

    gradient_free_budget_per_design_variable: int = 100
    """The budget for the gradient free optimization per design variable.

    The total budget is the number of design variables times this value. Set it to 0 to
    disable the gradient free optimization.
    """

    gradient_free_max_time: int | None = None
    """The maximum time in seconds for the gradient free optimization.

    If None, the optimization will run until the budget is exhausted.
    """

    gradient_free_optimization_hypercube_bounds: float = 1.0
    """The bounds of the unit hypercube for the gradient free optimization."""

    gradient_based_relative_objectives_tolerance: float = 1e-3
    """The relative tolerance termination criterion for the gradient based
    optimization."""

    verbose: bool = False
    """Whether to print the optimization progress to the console."""
