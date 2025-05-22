from typing import Callable

import numpy as np
import scipy.stats as stats
from tqdm import tqdm


class RandomWalkMetropolis:
    def __init__(self, target_logprob: Callable[[np.ndarray], float]):
        """Initializes the MCMC algorithm with the given target log-probability
        function.

        Args:
            target_logprob (callable): A function that computes the log-probability of
             the target distribution.
        """
        self._target_log_prob = target_logprob
        self.proposal_scale = None
        self.acceptance_ratio = None

    def run(
        self,
        N: int,
        cov_proposal: np.ndarray | None,
        x0: np.ndarray,
        burnin: int = 0,
        tuning_interval: int = 100,
        callback: Callable | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Runs the Random Walk Metropolis-Hastings algorithm with adaptive proposal
        scaling.

        Args:
            N: Number of iterations.
            cov_proposal: Covariance matrix for the proposal distribution.
            x0: Initial sample.
            burnin: Number of burn-in samples to discard. Default is 0.
            tuning_interval: How often to adapt the proposal scale.
            callback: A function that is called at each iteration. Default is None.

        Returns:
            The Markov Chain containing accepted samples. shape = (N, dimx)
        """
        if cov_proposal is None:
            cov_proposal = np.diag(np.abs(x0) ** 2 + 1e-06)  # Default proposal
        else:
            assert cov_proposal.ndim == 2, "Full covariance matrix must be supplied"

        dimx = np.size(x0)
        total_samples = N + burnin  # total iterations needed
        logp = self._target_log_prob(x0, **kwargs)
        accepted = 0

        X_chain = np.zeros((total_samples, dimx))
        X_chain[0, :] = x0  # include initial value in the chain
        proposal_scale = 1.0  # Start with no scaling

        rng = np.random.default_rng()  # Random number generator
        for n in tqdm(range(1, total_samples), desc="MCMC Sampling"):
            # Propose a new state
            cov_scaled = proposal_scale * cov_proposal
            x_proposed = stats.multivariate_normal.rvs(mean=x0, cov=cov_scaled)  # type: ignore
            logp_proposed = self._target_log_prob(x_proposed, **kwargs)

            # Metropolis acceptance criterion
            if np.log(rng.random()) <= logp_proposed - logp:
                x0 = x_proposed
                logp = logp_proposed
                accepted += 1
                is_accepted = True
            else:
                is_accepted = False

            X_chain[n, :] = x0

            # prepare the state dict for the callback
            if callback is not None:
                state = {
                    "iteration": n,
                    "current_sample": x0,
                    "chain": X_chain[: n + 1, :],
                    "acceptance_ratio": accepted / (n + 1),
                    "proposal_scale": proposal_scale,
                    "logp": logp,
                    "is_accepted": is_accepted,
                }
                callback(state)

            # Adaptive tuning of proposal scale
            # TODO: smarter way to define tuning interval
            if (n + 1) % tuning_interval == 0:
                proposal_scale = self._tune_scale_covariance(
                    proposal_scale, accepted / (n + 1)
                )

        self.proposal_scale = proposal_scale
        self.acceptance_ratio = accepted / N

        print(
            f"Final Acceptance Ratio: {self.acceptance_ratio:.3f}, "
            f"final Proposal Scale: {proposal_scale:.3f}"
        )

        # Return the chain after discarding the burn-in samples (if any)
        return X_chain[burnin:, :]

    def _tune_scale_covariance(
        self, proposal_scale: float, accept_rate: float
    ) -> float:
        r"""Tune the acceptance rate according to the last tuning interval. If higher
        acceptance rate , means you need to expand your search field or increase
        variance(its too small currently)

        The goal is an acceptance rate within 20\% - 50\%.
        The (acceptance) rate is adapted according to the following rule:

            Acceptance Rate    Variance adaptation factor
            ---------------    --------------------------
            <0.001                       x 0.1
            <0.05                        x 0.5
            <0.2                         x 0.9
            >0.5                         x 1.1
            >0.75                        x 2
            >0.95                        x 10

        The implementation is modified from [1].

        Reference:
        [1]: https://github.com/pymc-devs/pymc3/blob/master/pymc3/step_methods/metropolis.py
        """

        if accept_rate < 0.001:
            proposal_scale *= 0.1
        elif accept_rate < 0.05:
            proposal_scale *= 0.5
        elif accept_rate < 0.2:
            proposal_scale *= 0.9
        elif accept_rate < 0.5:
            pass  # No change
        elif accept_rate < 0.75:
            proposal_scale *= 1.1
        elif accept_rate < 0.95:
            proposal_scale *= 2.0
        else:
            proposal_scale *= 10.0

        return proposal_scale
