"""An integrated Bernstein-gradient radial profile."""

import math

import numpy as np
import pydantic

from constellaration.mhd import flux_power_series


class IntegratedBernsteinProfile(pydantic.BaseModel):
    r"""A profile with a normalized Bernstein gradient in :math:`s=\rho^2`.

    For :math:`n` free logits, let :math:`z=(z_0,\ldots,z_{n-1},0)`,
    :math:`w_k=e^{z_k}/\sum_{j=0}^{n} e^{z_j}`, and
    :math:`B_{k,n}(s)=\binom{n}{k}s^k(1-s)^{n-k}`. Then

    .. math::
        q(s) = (n+1)\sum_{k=0}^{n} w_k B_{k,n}(s), \qquad
        p(s) = p_0\int_s^1 q(t)\,dt.

    The final logit is fixed to zero to remove the softmax's redundant degree
    of freedom. Since :math:`q\geq0` and integrates to one, :math:`p(0)=p_0`,
    :math:`p(1)=0`, and :math:`dp/d\rho=0` at the axis.
    """

    model_config = pydantic.ConfigDict(allow_inf_nan=False)

    axis_pressure: float = pydantic.Field(gt=0)
    """Profile value :math:`p_0` at the magnetic axis, in pascals."""

    gradient_logits: tuple[float, ...] = (0.0, 0.0, 0.0)
    """Free softmax logits; their count sets the Bernstein polynomial degree.

    The default gives a cubic gradient. An empty tuple gives a constant gradient.
    """


def to_power_series(
    profile: IntegratedBernsteinProfile,
) -> flux_power_series.FluxPowerSeriesProfile:
    """Convert the analytically integrated profile to powers of normalized flux."""
    logits = np.array([*profile.gradient_logits, 0.0])
    weights = np.exp(logits - np.max(logits))
    weights /= np.sum(weights)
    degree = len(profile.gradient_logits)
    gradient = np.zeros(degree + 1)
    for k, weight in enumerate(weights):
        for j in range(degree - k + 1):
            gradient[k + j] += (
                (degree + 1)
                * weight
                * math.comb(degree, k)
                * math.comb(degree - k, j)
                * (-1) ** j
            )
    coefficients = [profile.axis_pressure] + [
        -profile.axis_pressure * c / (i + 1) for i, c in enumerate(gradient)
    ]
    return flux_power_series.FluxPowerSeriesProfile(coefficients=coefficients)
