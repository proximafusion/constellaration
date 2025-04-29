import numpy as np
import pydantic
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import flux_power_series


class IdealMHDParameters(pydantic.BaseModel):
    """Parameters that define an ideal-MHD equilibrium problem."""

    pressure: flux_power_series.FluxPowerSeriesProfile
    """A 1D radial profile describing the plasma pressure on a flux surface."""

    toroidal_current: flux_power_series.FluxPowerSeriesProfile
    """A 1D radial profile describing the integrated toroidal current within a flux
    surface."""

    boundary_toroidal_flux: float
    """The magnetic toroidal flux at the boundary of the plasma."""


def boundary_to_ideal_mhd_parameters(
    boundary: surface_rz_fourier.SurfaceRZFourier,
) -> IdealMHDParameters:
    r"""Creates ideal-magnetohydrodynamic (MHD) parameters from a surface.

    The edge toroidal flux is computed following the approximation that the magnetic
    field strength is constant and the cross-section is circular. The resulting
    edge toroidal flux is given by:

    .. math::
        \Phi_{edge} = \pi a^2

    where :math:`a` is the minor radius of the plasma.

    Args:
        boundary: The boundary surface of the plasma, represented as a Fourier
            series in RZ coordinates.

    Returns:
        The ideal-MHD parameters, including pressure, toroidal current, and boundary
        toroidal flux.
    """
    minor_radius = surface_rz_fourier.evaluate_minor_radius(surface=boundary)
    boundary_toroidal_flux = np.pi * minor_radius**2
    return IdealMHDParameters(
        pressure=flux_power_series.FluxPowerSeriesProfile(
            coefficients=[1.0, -1.0],
        ),
        toroidal_current=flux_power_series.FluxPowerSeriesProfile(
            coefficients=[0.0],
        ),
        boundary_toroidal_flux=boundary_toroidal_flux,
    )
