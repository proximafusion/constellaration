import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pydantic
from vmecpp import enable_x64  # pyright: ignore[reportAttributeAccessIssue]

from constellaration.mhd import jax_turbulent_transport, vmec_utils


class IdealMHDTurbulentTransportMetricsSettings(
    pydantic.BaseModel, arbitrary_types_allowed=True
):
    """Settings for the turbulent transport metrics of an ideal-MHD equilibrium."""

    normalized_toroidal_flux: jt.Float[np.ndarray, " n_surfaces"]
    """The normalized toroidal flux of the flux surfaces on which the turbulent
    transport metrics are evaluated."""

    n_field_lines: int
    """The number of field lines to use."""

    n_toroidal_points: int
    """The number of points along field lines to use."""


_FIELDS = (
    "rmnc",
    "zmns",
    "lmns",
    "gmnc",
    "bmnc",
    "iotas",
    "bsubsmns",
    "bsubumnc",
    "bsubvmnc",
    "bsupvmnc",
    "pres",
    "phi",
    "phipf",
    "Aminor_p",
)
_MODES = ("xm", "xn", "xm_nyq", "xn_nyq")
_reporting_kernel = jax.jit(
    jax_turbulent_transport.flux_compression,
    static_argnums=(3, 4, 5),
    static_argnames=("smoothing",),
)


def compute_flux_compression_in_regions_of_bad_curvature(
    equilibrium: vmec_utils.VmecppWOut,
    settings: IdealMHDTurbulentTransportMetricsSettings,
) -> float:
    """Evaluate the historical hard-mask metric using JAX field lines."""
    if equilibrium.lasym:
        raise NotImplementedError(
            "Only stellarator-symmetric field lines are supported"
        )
    with enable_x64(True):
        wout = {key: jnp.asarray(getattr(equilibrium, key)) for key in _FIELDS}
        modes = {key: jnp.asarray(getattr(equilibrium, key)) for key in _MODES}
        return float(
            _reporting_kernel(
                wout,
                modes,
                settings.normalized_toroidal_flux,
                equilibrium.nfp,
                settings.n_field_lines,
                settings.n_toroidal_points,
                smoothing=0.0,
            ).mean()
        )
