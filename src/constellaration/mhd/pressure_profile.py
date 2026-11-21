"""An integrated Bernstein-gradient radial profile."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from vmecpp import enable_x64  # pyright: ignore[reportAttributeAccessIssue]

from constellaration.mhd import flux_power_series
from constellaration.utils import pytree


@pytree.register_pydantic_data
class IntegratedBernsteinProfile(pydantic.BaseModel):
    r"""A profile with a normalized Bernstein gradient in :math:`s=\rho^2`.

    For :math:`n` free logits, let :math:`z=(z_0,\ldots,z_{n-1},0)`,
    :math:`w_k=e^{z_k}/\sum_{j=0}^{n} e^{z_j}`, and
    :math:`B_{k,n}(s)=\binom{n}{k}s^k(1-s)^{n-k}`. Then

    .. math::
        q(s) = (n+1)\sum_{k=0}^{n} w_k B_{k,n}(s), \qquad
        p(s) = p_0\int_s^1 q(t)\,dt.

    Axis pressure and the free logits are differentiable JAX PyTree leaves.
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


def power_series_basis(degree: int) -> np.ndarray:
    """Columns map normalized Bernstein weights to pressure power coefficients."""
    basis = np.zeros((degree + 2, degree + 1))
    basis[0] = 1
    for k in range(degree + 1):
        for j in range(degree - k + 1):
            i = k + j + 1
            basis[i, k] = (
                -(degree + 1)
                * math.comb(degree, k)
                * math.comb(degree - k, j)
                * (-1) ** j
                / i
            )
    return basis


def power_series_coefficients(profile: IntegratedBernsteinProfile) -> jax.Array:
    """JAX power-series coefficients in ascending powers of normalized flux."""
    logits = jnp.stack((*profile.gradient_logits, 0.0))
    weights = jax.nn.softmax(logits)
    coefficients = profile.axis_pressure * (
        jnp.asarray(power_series_basis(len(profile.gradient_logits))) @ weights
    )
    return coefficients.at[0].set(profile.axis_pressure)


def to_power_series(
    profile: IntegratedBernsteinProfile,
) -> flux_power_series.FluxPowerSeriesProfile:
    """Convert a concrete profile to serializable VMEC pressure coefficients."""
    with enable_x64(True):
        coefficients = power_series_coefficients(profile).tolist()
    return flux_power_series.FluxPowerSeriesProfile(coefficients=coefficients)
