import enum
import pathlib
import tempfile

import jaxtyping as jt
import numpy as np
import vmecpp
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import (
    flux_power_series,
    ideal_mhd_parameters,
    vmec_settings,
)
from scipy import interpolate
from simsopt import mhd
from vmecpp import _pydantic_numpy as pydantic_numpy


# TODO(mariap): Remove this and use the class from vmecpp
class VmecppWOut(pydantic_numpy.BaseModelWithNumpy):
    version: str
    input_extension: str = ""
    sign_of_jacobian: int
    gamma: float
    pcurr_type: str
    pmass_type: str
    piota_type: str
    am: jt.Float[np.ndarray, "..."]
    ac: jt.Float[np.ndarray, "..."]
    ai: jt.Float[np.ndarray, "..."]
    am_aux_s: jt.Float[np.ndarray, "..."]
    am_aux_f: jt.Float[np.ndarray, "..."]
    ac_aux_s: jt.Float[np.ndarray, "..."]
    ac_aux_f: jt.Float[np.ndarray, "..."]
    ai_aux_s: jt.Float[np.ndarray, "..."]
    ai_aux_f: jt.Float[np.ndarray, "..."]
    nfp: int
    mpol: int
    ntor: int
    lasym: bool
    ns: int
    ftolv: float
    maximum_iterations: int
    lfreeb: bool
    mgrid_file: str
    extcur: jt.Float[np.ndarray, "..."]
    mgrid_mode: str
    wb: float
    wp: float
    rmax_surf: float
    rmin_surf: float
    zmax_surf: float
    mnmax: int
    mnmax_nyq: int
    ier_flag: int
    aspect: float
    betatot: float
    betapol: float
    betator: float
    betaxis: float
    b0: float
    rbtor0: float
    rbtor: float
    IonLarmor: float
    VolAvgB: float
    ctor: float
    Aminor_p: float
    Rmajor_p: float
    volume_p: float
    fsqr: float
    fsqz: float
    fsql: float
    itfsq: int = 0  # Number of iterations
    fsqt: jt.Float[np.ndarray, "..."] = np.array([])  # Force residual
    wdot: jt.Float[np.ndarray, "..."] = np.array([])  # Energy decay
    iota_full: jt.Float[np.ndarray, "..."]
    safety_factor: jt.Float[np.ndarray, "..."]
    pressure_full: jt.Float[np.ndarray, "..."]
    toroidal_flux: jt.Float[np.ndarray, "..."]
    phipf: jt.Float[np.ndarray, "..."]
    poloidal_flux: jt.Float[np.ndarray, "..."]
    chipf: jt.Float[np.ndarray, "..."]
    jcuru: jt.Float[np.ndarray, "..."]
    jcurv: jt.Float[np.ndarray, "..."]
    iota_half: jt.Float[np.ndarray, "..."]
    mass: jt.Float[np.ndarray, "..."]
    pressure_half: jt.Float[np.ndarray, "..."]
    beta: jt.Float[np.ndarray, "..."]
    buco: jt.Float[np.ndarray, "..."]
    bvco: jt.Float[np.ndarray, "..."]
    dVds: jt.Float[np.ndarray, "..."]
    spectral_width: jt.Float[np.ndarray, "..."]
    phips: jt.Float[np.ndarray, "..."]
    overr: jt.Float[np.ndarray, "..."]
    # For compatibility with wout results produced by
    # vmecpp versions <0.4.0
    bdotb: jt.Float[np.ndarray, "..."] = np.array([])
    jdotb: jt.Float[np.ndarray, "..."]
    bdotgradv: jt.Float[np.ndarray, "..."]
    DMerc: jt.Float[np.ndarray, "..."]
    Dshear: jt.Float[np.ndarray, "..."]
    Dwell: jt.Float[np.ndarray, "..."]
    Dcurr: jt.Float[np.ndarray, "..."]
    Dgeod: jt.Float[np.ndarray, "..."]
    equif: jt.Float[np.ndarray, "..."]
    curlabel: list[str]
    potvac: jt.Float[np.ndarray, "..."]
    xm: jt.Int[np.ndarray, "..."]
    xn: jt.Int[np.ndarray, "..."]
    xm_nyq: jt.Int[np.ndarray, "..."]
    xn_nyq: jt.Int[np.ndarray, "..."]
    raxis_c: jt.Float[np.ndarray, "..."]
    zaxis_s: jt.Float[np.ndarray, "..."]
    rmnc: jt.Float[np.ndarray, "..."]
    zmns: jt.Float[np.ndarray, "..."]
    lmns_full: jt.Float[np.ndarray, "..."]
    lmns: jt.Float[np.ndarray, "..."]
    gmnc: jt.Float[np.ndarray, "..."]
    bmnc: jt.Float[np.ndarray, "..."]
    bsubumnc: jt.Float[np.ndarray, "..."]
    bsubvmnc: jt.Float[np.ndarray, "..."]
    bsubsmns: jt.Float[np.ndarray, "..."]
    bsubsmns_full: jt.Float[np.ndarray, "..."]
    bsupumnc: jt.Float[np.ndarray, "..."]
    bsupvmnc: jt.Float[np.ndarray, "..."]
    raxis_s: jt.Float[np.ndarray, "..."]
    zaxis_c: jt.Float[np.ndarray, "..."]
    rmns: jt.Float[np.ndarray, "..."]
    zmnc: jt.Float[np.ndarray, "..."]
    lmnc_full: jt.Float[np.ndarray, "..."]
    lmnc: jt.Float[np.ndarray, "..."]
    gmns: jt.Float[np.ndarray, "..."]
    bmns: jt.Float[np.ndarray, "..."]
    bsubumns: jt.Float[np.ndarray, "..."]
    bsubvmns: jt.Float[np.ndarray, "..."]
    bsubsmnc: jt.Float[np.ndarray, "..."]
    bsubsmnc_full: jt.Float[np.ndarray, "..."]
    bsupumns: jt.Float[np.ndarray, "..."]
    bsupvmns: jt.Float[np.ndarray, "..."]

    @property
    def nextcur(self) -> int:
        return len(self.extcur)

    @property
    def n_field_periods(self) -> int:
        return self.nfp

    @property
    def normalized_toroidal_flux_full_grid_mesh(
        self,
    ) -> jt.Float[np.ndarray, " n_surfaces"]:
        return np.linspace(0, 1, self.ns)

    @property
    def normalized_toroidal_flux_half_grid_mesh(
        self,
    ) -> jt.Float[np.ndarray, " n_surfaces"]:
        full_grid = self.normalized_toroidal_flux_full_grid_mesh
        ds = full_grid[1] - full_grid[0]
        return full_grid - ds / 2

    @property
    def nyquist_poloidal_mode_numbers(
        self,
    ) -> jt.Int[np.ndarray, " n_fourier_coefficients_nyquist"]:
        return self.xm_nyq

    @property
    def nyquist_toroidal_mode_numbers(
        self,
    ) -> jt.Int[np.ndarray, " n_fourier_coefficients_nyquist"]:
        return self.xn_nyq

    @property
    def poloidal_mode_numbers(
        self,
    ) -> jt.Int[np.ndarray, " n_fourier_coefficients"]:
        """The poloidal mode numbers (m-values) corresponding to each Fourier mode.

        Example for `mpol=4`, `ntor=1`: xm = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        """
        return self.xm

    @property
    def toroidal_mode_numbers(
        self,
    ) -> jt.Int[np.ndarray, " n_fourier_coefficients"]:
        """The toroidal mode numbers (n-values) corresponding to each Fourier mode.

        Note that they including the factor `nfp`, i.e. for `mpol=4`, `ntor=1`, `nfp=5`:
        xn = [0, 5,-5, 0, 5,-5, 0, 5,-5, 0, 5]
        """
        return self.xn


def run_vmec(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    mhd_parameters: ideal_mhd_parameters.IdealMHDParameters,
    vmec_settings: vmec_settings.VmecSettings,
) -> VmecppWOut:
    """Run VMEC++ in fixed-boundary mode."""
    vmec_indata = build_vmecpp_indata(
        mhd_parameters=mhd_parameters,
        boundary=boundary,
        vmec_settings=vmec_settings,
    )

    output_quantities = vmecpp.run(
        vmec_indata,
        verbose=vmec_settings.verbose,
        max_threads=vmec_settings.max_threads,
    )
    return vmecppwout_from_wout(output_quantities.wout)


def build_vmecpp_indata(
    mhd_parameters: ideal_mhd_parameters.IdealMHDParameters,
    boundary: surface_rz_fourier.SurfaceRZFourier,
    vmec_settings: vmec_settings.VmecSettings,
) -> vmecpp.VmecInput:
    if not boundary.is_stellarator_symmetric:
        raise NotImplementedError("Only stellarator symmetric surfaces are supported.")

    indata = vmecpp.VmecInput.default()

    indata.lasym = not boundary.is_stellarator_symmetric
    indata.nfp = boundary.n_field_periods

    # mpol, ntor cannot be assigned directly for consistency reasons
    indata.mpol = vmec_settings.n_poloidal_modes
    indata.ntor = vmec_settings.max_toroidal_mode

    indata.ntheta = vmec_settings.n_poloidal_grid_points
    indata.nzeta = vmec_settings.n_toroidal_grid_points

    indata.ns_array = np.array(
        [iteration.n_flux_surfaces for iteration in vmec_settings.multigrid_steps],
        dtype=int,
    )

    indata.ftol_array = np.array(
        [iteration.force_tolerance for iteration in vmec_settings.multigrid_steps],
        dtype=float,
    )

    indata.niter_array = np.array(
        [iteration.n_max_iterations for iteration in vmec_settings.multigrid_steps],
        dtype=int,
    )

    indata.phiedge = mhd_parameters.boundary_toroidal_flux

    # This is the switch to solve the equilibrium with iota or toroidal current as a
    # constraint.
    # 0 means constrained iota, 1 means constrained current
    # for our purposes (QI stellarators) we usually want to solve the equilibrium
    # constraining on the toroidal current since we expect it to be negligible.
    indata.ncurr = 1

    indata.pmass_type = "power_series"
    indata.pres_scale = mhd_parameters.pressure.coefficients[0]
    if indata.pres_scale == 0.0:
        indata.am = np.zeros_like(mhd_parameters.pressure.coefficients)
    else:
        indata.am = np.array(mhd_parameters.pressure.coefficients) / indata.pres_scale

    # using default indata.am_aux_s and indata.am_aux_f

    # indata.gamma and indata.spres_ped is left to its default value

    # indata.piota_type is left to its default value

    # see vmecpp/vmec/radial_profiles/radial_profiles.cc for why we need
    # power_series instead of power_series_i: the latter interprets ac
    # as the current profile, while "power_series" interprets it as the
    # _derivative_ of the current profile.
    indata.pcurr_type = "power_series"
    d_toroidal_current_d_s = flux_power_series.evaluate_derivative(
        mhd_parameters.toroidal_current
    )
    indata.ac = np.array(d_toroidal_current_d_s.coefficients)

    indata.curtor = flux_power_series.evaluate_at_normalized_effective_radius(
        mhd_parameters.toroidal_current,
        np.array([1.0]),
    ).item()

    # indata.bloat is left to its default value

    indata.lfreeb = False

    # indata.{mgrid_file,free_boundary_method} are not used

    indata.nstep = 250  # a sensible fixed value

    # indata.aphi is left to its default value

    indata.delt = vmec_settings.time_step

    # indata.{tcon0,lforbal} are left to its default value

    # indata.raxis_c and indata.zaxis_s are left to their default values so that we get
    # VMEC's initial guess for the magnetic axis
    indata.raxis_c = np.zeros(indata.ntor + 1)
    indata.zaxis_s = np.zeros(indata.ntor + 1)

    # rbc, zbs
    # (rbs and zbc are not set: we only support stellarator-symmetric boundaries)
    indata.rbc = vmecpp.VmecInput.resize_2d_coeff(
        boundary.r_cos,
        indata.mpol,
        indata.ntor,
    )
    indata.zbs = vmecpp.VmecInput.resize_2d_coeff(
        boundary.z_sin,
        indata.mpol,
        indata.ntor,
    )

    return vmecpp.VmecInput.model_validate(indata)


def vmecppwout_from_wout(
    wout: vmecpp.VmecWOut,
) -> VmecppWOut:
    cpp_wout = wout._to_cpp_wout()
    assert isinstance(cpp_wout, vmecpp._vmecpp.WOutFileContents)

    attributes = VmecppWOut.__annotations__.keys()
    return VmecppWOut(**{attr: getattr(cpp_wout, attr) for attr in attributes})


def as_simsopt_vmec(equilibrium: VmecppWOut) -> mhd.Vmec:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        wout_path = tmpdir / "wout_temp.nc"
        wout = vmecpp.VmecWOut._from_cpp_wout(equilibrium)
        wout.save(wout_path)
        return mhd.Vmec(wout_path.as_posix())


# TODO(mariap): Remove this class
class _FourierBasis(enum.Enum):
    COSINE = "cosine"
    SINE = "sine"


def magnetic_field_magnitude(
    equilibrium: VmecppWOut,
    s_theta_phi: jt.Float[np.ndarray, "n_s n_theta n_phi 3"],
) -> jt.Float[np.ndarray, " *dims"]:
    """Computes the magnetic field magnitude on a set of of (s, theta, phi) points."""
    magnetic_field_interpolator = _build_radial_interpolator(
        equilibrium=equilibrium,
        fourier_coefficients=equilibrium.bmnc,
        is_on_full_mesh=False,
    )
    magnetic_field_fourier_coefficients = _interpolate_radially(
        interpolator=magnetic_field_interpolator,
        normalized_toroidal_flux=s_theta_phi[..., 0],
    )
    return _inverse_fourier_transform(
        fourier_coefficients=magnetic_field_fourier_coefficients,
        poloidal_mode_numbers=equilibrium.nyquist_poloidal_mode_numbers,
        toroidal_mode_numbers=equilibrium.nyquist_toroidal_mode_numbers,
        s_theta_phi=s_theta_phi,
        basis=_FourierBasis.COSINE,
    )


def _build_radial_interpolator(
    equilibrium: VmecppWOut,
    fourier_coefficients: jt.Float[np.ndarray, "n_surfaces n_fourier_coefficients"],
    is_on_full_mesh: bool,
) -> interpolate.interp1d:
    if is_on_full_mesh:
        x = equilibrium.normalized_toroidal_flux_full_grid_mesh
    else:
        x = equilibrium.normalized_toroidal_flux_half_grid_mesh[1:]
    return interpolate.interp1d(
        x=x,
        y=fourier_coefficients,
        axis=0,
        fill_value="extrapolate",  # pyright: ignore
    )


def _interpolate_radially(
    interpolator: interpolate.interp1d,
    normalized_toroidal_flux: jt.Float[np.ndarray, "n_surfaces *dims"],  # noqa: F821
) -> jt.Float[np.ndarray, "n_surfaces *dims"]:  # noqa: F821
    if np.any(normalized_toroidal_flux < 0) or np.any(normalized_toroidal_flux > 1):
        raise ValueError("Normalized toroidal flux must be in [0, 1].")
    return interpolator(normalized_toroidal_flux)


def _inverse_fourier_transform(
    fourier_coefficients: jt.Float[np.ndarray, "n_surfaces n_fourier_coefficients"],
    poloidal_mode_numbers: jt.Int[np.ndarray, " n_fourier_coefficients"],
    toroidal_mode_numbers: jt.Int[np.ndarray, " n_fourier_coefficients"],
    s_theta_phi: jt.Float[np.ndarray, "n_s n_theta n_phi 3"],
    basis: _FourierBasis,
) -> jt.Float[np.ndarray, "n_s n_theta n_phi"]:
    # pyright gets confused about the type of the arrays for some reason
    angle = (  # pyright: ignore
        poloidal_mode_numbers[np.newaxis, np.newaxis, :]
        * s_theta_phi[:, :, :, 1, np.newaxis]
        - toroidal_mode_numbers[np.newaxis, np.newaxis, :]
        * s_theta_phi[:, :, :, 2, np.newaxis]
    )
    if basis == _FourierBasis.COSINE:
        return np.sum(fourier_coefficients * np.cos(angle), axis=-1)
    elif basis == _FourierBasis.SINE:
        return np.sum(fourier_coefficients * np.sin(angle), axis=-1)
