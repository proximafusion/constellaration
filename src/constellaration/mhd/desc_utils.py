"""DESC ideal-MHD equilibria: a serializable representation, conversion to and from
DESC objects, and the pipeline that re-solves a VMEC++ equilibrium with DESC.

The `desc_equilibrium` subset of the ConStellaration Hugging Face dataset stores one
`DescEquilibrium` JSON per plasma configuration, produced by
`solve_desc_equilibrium_from_vmecpp_wout`.
"""

import pathlib
import tempfile

import jaxtyping as jt
import numpy as np
import pydantic
from desc import equilibrium as desc_equilibrium
from desc import profiles as desc_profiles
from desc import vmec as desc_vmec
from typing_extensions import Self
from vmecpp import _pydantic_numpy as pydantic_numpy

from constellaration.mhd import vmec_utils
from constellaration.utils import file_exporter


class DescSplineProfile(pydantic_numpy.BaseModelWithNumpy):
    """A radial profile represented as a DESC `SplineProfile`."""

    method: str = "cubic2"
    """Spline interpolation method, e.g. `"linear"`, `"cubic"` or `"cubic2"`."""

    name: str = ""
    """Name of the profile, e.g. `"pressure"` or `"current"`."""

    knots: jt.Float[np.ndarray, " n_knots"]
    """Knots of the spline in the normalized effective radius `rho`."""

    params: jt.Float[np.ndarray, " n_knots"]
    """Values of the profile at the knots."""


class DescEquilibrium(pydantic_numpy.BaseModelWithNumpy):
    """A fixed-boundary DESC equilibrium.

    The geometry is represented on a Fourier-Zernike basis, see
    https://desc-docs.readthedocs.io/en/stable/notebooks/basis_grid.html.
    """

    r_lmn: jt.Float[np.ndarray, " n_modes_r"]
    """Fourier-Zernike coefficients of the cylindrical coordinate R."""

    z_lmn: jt.Float[np.ndarray, " n_modes_z"]
    """Fourier-Zernike coefficients of the cylindrical coordinate Z."""

    lambda_lmn: jt.Float[np.ndarray, " n_modes_lambda"]
    """Fourier-Zernike coefficients of the poloidal stream function lambda."""

    radial_resolution: pydantic.NonNegativeInt
    """Radial resolution L of the Fourier-Zernike basis."""

    poloidal_resolution: pydantic.NonNegativeInt
    """Poloidal resolution M of the Fourier-Zernike basis."""

    toroidal_resolution: pydantic.NonNegativeInt
    """Toroidal resolution N of the Fourier-Zernike basis."""

    spectral_indexing: str = "ansi"
    """Indexing convention of the Zernike polynomials, `"ansi"` or `"fringe"`."""

    n_field_periods: pydantic.PositiveInt
    """Number of toroidal field periods."""

    is_stellarator_symmetric: bool = True
    """Whether the equilibrium is stellarator symmetric."""

    total_toroidal_flux: float
    """Toroidal magnetic flux enclosed by the plasma boundary, in Wb."""

    pressure_profile: DescSplineProfile
    """Plasma pressure, in Pa."""

    current_profile: DescSplineProfile | None = None
    """Enclosed toroidal current, in A.

    Exactly one of `current_profile` and `iota_profile` is set.
    """

    iota_profile: DescSplineProfile | None = None
    """Rotational transform.

    Exactly one of `current_profile` and `iota_profile` is set.
    """

    r_axis: jt.Float[np.ndarray, " n_modes_r_axis"] | None = None
    """Fourier coefficients of the R coordinate of the magnetic axis."""

    z_axis: jt.Float[np.ndarray, " n_modes_z_axis"] | None = None
    """Fourier coefficients of the Z coordinate of the magnetic axis."""

    r_boundary_lmn: jt.Float[np.ndarray, " n_modes_r_boundary"] | None = None
    """Fourier coefficients of the R coordinate of the plasma boundary."""

    z_boundary_lmn: jt.Float[np.ndarray, " n_modes_z_boundary"] | None = None
    """Fourier coefficients of the Z coordinate of the plasma boundary."""

    @pydantic.model_validator(mode="after")
    def _check_exactly_one_of_current_or_iota(self) -> Self:
        if (self.current_profile is None) == (self.iota_profile is None):
            raise ValueError(
                "Exactly one of current_profile and iota_profile must be provided."
            )
        return self


class DescEquilibriumSolution(pydantic_numpy.BaseModelWithNumpy):
    """A DESC equilibrium solved for force balance, with its residual force error."""

    equilibrium: DescEquilibrium

    normalized_mean_absolute_force_error: float
    """Mean absolute force-balance error of `equilibrium`, as reported by DESC's
    `ForceBalance` objective, normalized by its characteristic force."""

    normalized_max_absolute_force_error: float
    """Maximum absolute force-balance error of `equilibrium`, as reported by DESC's
    `ForceBalance` objective, normalized by its characteristic force."""


def solve_desc_equilibrium_from_vmecpp_wout(
    vmecpp_wout: vmec_utils.VmecppWOut,
) -> DescEquilibriumSolution:
    """Re-solve a VMEC++ equilibrium with DESC.

    This is the pipeline that produced the `desc_equilibrium` subset of the
    ConStellaration dataset: DESC is initialized from the VMEC++ equilibrium and
    then solved for force balance with fixed boundary, pressure and toroidal current.
    """
    return solve_desc_equilibrium(desc_equilibrium_from_vmecpp_wout(vmecpp_wout))


def desc_equilibrium_from_vmecpp_wout(
    vmecpp_wout: vmec_utils.VmecppWOut,
) -> DescEquilibrium:
    """Fit a DESC equilibrium to a VMEC++ equilibrium, without solving it.

    The Fourier-Zernike resolution follows DESC's defaults for the given VMEC
    resolution, and the toroidal current (not the rotational transform) is taken
    as the profile constraint, as in the VMEC++ run.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        wout_path = pathlib.Path(tmp_dir) / "wout.nc"
        file_exporter.to_vmec2000_wout_file(vmecpp_wout, wout_path)
        equilibrium = desc_vmec.VMECIO.load(wout_path, profile="current")
    return from_desc_equilibrium(equilibrium)


def solve_desc_equilibrium(
    equilibrium: DescEquilibrium,
    verbose: int = 0,
) -> DescEquilibriumSolution:
    """Solve a DESC equilibrium for force balance with fixed boundary and profiles."""
    desc_object = to_desc_equilibrium(equilibrium)
    solved, result = desc_object.solve(objective="force", verbose=verbose)
    force_error = result["Objective values"]["Force error: "][0]
    return DescEquilibriumSolution(
        equilibrium=from_desc_equilibrium(solved),
        normalized_mean_absolute_force_error=float(force_error["f_mean_norm"]),
        normalized_max_absolute_force_error=float(force_error["f_max_norm"]),
    )


def to_desc_equilibrium(
    equilibrium: DescEquilibrium,
) -> desc_equilibrium.Equilibrium:
    """Converts a `DescEquilibrium` to a `desc.equilibrium.Equilibrium`."""
    desc_object = desc_equilibrium.Equilibrium(
        R_lmn=equilibrium.r_lmn,
        Z_lmn=equilibrium.z_lmn,
        L_lmn=equilibrium.lambda_lmn,
        L=equilibrium.radial_resolution,
        M=equilibrium.poloidal_resolution,
        N=equilibrium.toroidal_resolution,
        spectral_indexing=equilibrium.spectral_indexing,
        Psi=equilibrium.total_toroidal_flux,
        NFP=equilibrium.n_field_periods,
        sym=equilibrium.is_stellarator_symmetric,
        # The coefficients come from a DESC equilibrium, so they are already
        # right-handed.
        check_orientation=False,
        pressure=_to_desc_spline_profile(equilibrium.pressure_profile),
        current=(
            _to_desc_spline_profile(equilibrium.current_profile)
            if equilibrium.current_profile is not None
            else None
        ),
        iota=(
            _to_desc_spline_profile(equilibrium.iota_profile)
            if equilibrium.iota_profile is not None
            else None
        ),
    )
    # Without a `surface` argument the constructor installs a default boundary and
    # axis; a fixed-boundary solve would then be constrained to the wrong boundary.
    if equilibrium.r_axis is not None:
        desc_object.Ra_n = equilibrium.r_axis
    if equilibrium.z_axis is not None:
        desc_object.Za_n = equilibrium.z_axis
    if equilibrium.r_boundary_lmn is not None:
        desc_object.Rb_lmn = equilibrium.r_boundary_lmn
    if equilibrium.z_boundary_lmn is not None:
        desc_object.Zb_lmn = equilibrium.z_boundary_lmn
    return desc_object


def from_desc_equilibrium(
    desc_object: desc_equilibrium.Equilibrium,
) -> DescEquilibrium:
    """Converts a `desc.equilibrium.Equilibrium` to a `DescEquilibrium`.

    Profiles that are not `SplineProfile`s are resampled onto one. If both a current
    and a rotational transform profile are set, only the current is kept.
    """
    if desc_object.pressure is None:
        raise ValueError("The DESC equilibrium must have a pressure profile.")
    current = (
        _from_desc_profile(desc_object.current)
        if desc_object.current is not None
        else None
    )
    iota = (
        _from_desc_profile(desc_object.iota)
        if desc_object.iota is not None and current is None
        else None
    )
    params = desc_object.params_dict
    # DESC types this as optional, but an Equilibrium always has a basis indexing.
    spectral_indexing = desc_object.spectral_indexing
    assert spectral_indexing is not None
    return DescEquilibrium(
        r_lmn=np.asarray(desc_object.R_lmn),
        z_lmn=np.asarray(desc_object.Z_lmn),
        lambda_lmn=np.asarray(desc_object.L_lmn),
        radial_resolution=desc_object.L,
        poloidal_resolution=desc_object.M,
        toroidal_resolution=desc_object.N,
        spectral_indexing=spectral_indexing,
        n_field_periods=desc_object.NFP,
        is_stellarator_symmetric=bool(desc_object.sym),
        total_toroidal_flux=float(desc_object.Psi),
        pressure_profile=_from_desc_profile(desc_object.pressure),
        current_profile=current,
        iota_profile=iota,
        r_axis=np.asarray(params["Ra_n"]),
        z_axis=np.asarray(params["Za_n"]),
        r_boundary_lmn=np.asarray(params["Rb_lmn"]),
        z_boundary_lmn=np.asarray(params["Zb_lmn"]),
    )


def _to_desc_spline_profile(
    profile: DescSplineProfile,
) -> desc_profiles.SplineProfile:
    return desc_profiles.SplineProfile(
        values=profile.params,
        knots=profile.knots,
        method=profile.method,
        name=profile.name,
    )


def _from_desc_profile(profile: desc_profiles._Profile) -> DescSplineProfile:
    if not isinstance(profile, desc_profiles.SplineProfile):
        knots = np.linspace(0.0, 1.0, 20)
        profile = desc_profiles.SplineProfile(
            values=np.asarray(profile(knots)),
            knots=knots,
            method="cubic2",
            name=profile.name,
        )
    return DescSplineProfile(
        method=profile._method,
        name=profile.name,
        knots=np.asarray(profile.knots),
        params=np.asarray(profile.params),
    )
