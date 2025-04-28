import pydantic
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
