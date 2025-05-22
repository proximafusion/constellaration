"""A general module to optimize a function while satisfying constraints using MCMC
sampling.

"""
from typing import Callable

import numpy as np
import pydantic
import scipy.stats as stats

from constellaration.generative_model import mcmc_algo


class OptimizeWithMcmcSettings(pydantic.BaseModel):
    num_samples: int = 1000
    smooth_constraints: bool = False
    prior_scale: float = 4
    k: int = 3
    sigma: float = 0.01
    beta: float = 1.0


def optimize_with_mcmc(
    function: Callable[[np.ndarray], tuple],
    x0: np.ndarray,
    settings: OptimizeWithMcmcSettings,
    callback: Callable[[np.ndarray], None] | None = None,
    prior: Callable[[np.ndarray], float] | None = None,
    likelihood: Callable[[np.ndarray], float] | None = None,
    initial_stepsize: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Optimizes a function using MCMC sampling."
    Args:
        function: For a given design variable, a function to compute a tuple of
            objectives and constraints. Constraint negative is considered as
            satisfied.
        x0: Initial guess.
        settings: Settings for the MCMC algorithm.
        callback: A function that is called at each iteration.
        prior: (Optional) Custom log-prior function that accepts an np.ndarray and
                returns a float. If not provided, a default Gaussian prior is used.
        likelihood: (Optional) Custom log-likelihood function that accepts an np.ndarray
                and returns a float. If not provided, a default quasi log-likelihood is
                used.
        **kwargs: Additional keyword arguments to pass to the function.
    Returns:
        An array of samples.
    """

    print(f"The dimension of the design space is {x0.shape[0]}")

    # Define the target log-probability function
    def _default_log_prior(x: np.ndarray) -> float:
        """Computes the log-prior for the design variable."""
        prior_scale = settings.prior_scale  # 4.0
        cov_prior = np.diag(prior_scale * (np.abs(x0)) ** 2 + 1e-6)
        return stats.multivariate_normal(mean=x0, cov=cov_prior).logpdf(x)  # type: ignore

    def _default_log_likelihood(x: np.ndarray) -> float:
        """Computes the quasi log-likelihood for the design variable."""
        try:
            objectives, constraints = function(x, **kwargs)
        except Exception as e:
            print(f"Error in function evaluation: {e}")
            return -np.inf

        # for the objective, quasi-likelihood
        beta = settings.beta  # can be a hyperparameter too. e.g. 1.0
        num_objectives = objectives.shape[0]
        # The is just a basic form to deal with multi-objective optimization
        w = 1 / num_objectives
        quasi_likelihood_obj = np.exp(-beta * np.sum(w * objectives))

        # for the constraints, indicator function
        if settings.smooth_constraints:
            k = settings.k  # more the k, more steep the sigmoid. default 3
            sigma = (
                settings.sigma
            )  # controls the degree of constraint violation allowed, def 0.01
            quasi_likelihood_cons = 1.0
            for con in constraints:
                # sigmoid function
                quasi_likelihood_cons_tmp = 1.0 / (
                    1.0 + np.exp(k * con)
                )  # 1/(1+exp(-k(-x))) -x as we want 1 when constraoints satisfied
                # else 0
                # gaussian
                quasi_likelihood_cons *= np.exp(
                    -0.5 * ((1 - quasi_likelihood_cons_tmp) / sigma) ** 2
                )  # enforcing to be 1 i.e., constraint met
        else:
            # indicator function
            quasi_likelihood_cons = 1.0 if np.all(constraints <= 0) else 0.0
        return np.log(quasi_likelihood_cons + 1e-10) + np.log(
            quasi_likelihood_obj + 1e-10
        )

    # Use the provided functions or fall back to the defaults
    log_prior = prior if prior is not None else _default_log_prior
    log_likelihood = likelihood if likelihood is not None else _default_log_likelihood

    def _log_target(x: np.ndarray, **_) -> float:
        """Computes the log-target for the design variable."""
        return log_prior(x) + log_likelihood(x)

    # Initialize the MCMC algorithm
    print(
        "Running MCMC (Adaptive Random Walk Metrolopis Hastings)"
        f"with {settings.num_samples} samples"
    )
    mcmc = mcmc_algo.RandomWalkMetropolis(target_logprob=_log_target)

    # set the proposal covariance
    if initial_stepsize is not None:
        cov_proposal = np.diag(initial_stepsize**2)
    else:
        cov_proposal = None

    # Run the algorithm
    samples = mcmc.run(
        N=settings.num_samples,
        cov_proposal=cov_proposal,
        x0=x0,
        callback=callback,
        tuning_interval=settings.num_samples // 20,
        **kwargs,
    )
    return samples
