import jaxtyping as jt
import numpy as np
import pydantic
from constellaration.mhd import vmec_utils
from simsopt import mhd

RADIAL_PROFILE = jt.Float[np.ndarray, " n_flux_surfaces"]
FOURIER_MODES = jt.Integer[np.ndarray, " n_fourier_modes"]
FOURIER_COEFFICIENTS = jt.Float[np.ndarray, " n_fourier_modes n_flux_surfaces"]


class BoozerOutput(pydantic.BaseModel, arbitrary_types_allowed=True):
    is_stellarator_symmetric: bool
    """Whether the equilibrium is stellarator symmetric (`asym` in booz_xform)."""

    n_field_periods: int
    """The equilibrium number of field periods."""

    xm_b: FOURIER_MODES
    """The poloidal mode numbers."""

    xn_b: FOURIER_MODES
    """The toroidal mode numbers."""

    boozer_normalized_toroidal_flux: RADIAL_PROFILE
    """The normalized toroidal flux at which the Boozer transformation has been
    performed."""

    normalized_toroidal_flux: RADIAL_PROFILE
    """The normalized toroidal flux at which the input equilibrium is defined."""

    boozer_flux_surface_indices: list[int]
    """The flux surface indices for which the Boozer transformation has been
    performed."""

    aspect: float
    """The aspect ratio of the equilibrium."""

    n_boozer_poloidal_modes: int
    """The number of Boozer poloidal modes."""

    max_boozer_toroidal_mode: int
    """The number of Boozer toroidal modes."""

    n_poloidal_modes: int
    """The number of poloidal modes."""

    max_toroidal_mode: int
    """The maximum toroidal mode number."""

    n_nyquist_poloidal_modes: int
    """The number of Nyquist poloidal modes."""

    max_nyquist_toroidal_mode: int
    """The maximum Nyquist toroidal mode number."""

    toroidal_flux: float
    """The toroidal flux of the equilibrium."""

    xm: FOURIER_MODES
    """The poloidal mode numbers."""

    xm_nyq: FOURIER_MODES
    """The Nyquist poloidal mode numbers."""

    xn: FOURIER_MODES
    """The toroidal mode numbers."""

    xn_nyq: FOURIER_MODES
    """The Nyquist toroidal mode numbers."""

    phi: RADIAL_PROFILE
    """The toroidal flux."""

    phip: RADIAL_PROFILE
    """The toroidal flux derivative."""

    G: RADIAL_PROFILE
    """The external poloidal current (`Boozer_G` in booz_xform) at
    `normalized_toroidal_flux`."""

    I: RADIAL_PROFILE  # noqa: E741
    """The external toroidal current (`Boozer_I` in booz_xform) at
    `normalized_toroidal_flux`."""

    G_all_flux_surfaces: RADIAL_PROFILE
    """The external poloidal current for all surfaces (`Boozer_G_all` in booz_xform)."""

    I_all_flux_surfaces: RADIAL_PROFILE
    """The external toroidal current for all surfaces (`Boozer_I_all` in booz_xform)."""

    iota: RADIAL_PROFILE
    """The rotational transform."""

    bmnc_b: FOURIER_COEFFICIENTS
    """The magnetic field strength cosine Fourier coefficients in Boozer coordinates."""

    bmns_b: FOURIER_COEFFICIENTS
    """The magnetic field strength sine Fourier coefficients in Boozer coordinates."""

    gmnc_b: FOURIER_COEFFICIENTS
    """The equilibrium jacobian cosine Fourier coefficients in Boozer coordinates."""

    gmns_b: FOURIER_COEFFICIENTS
    """The equilibrium jacobian sine Fourier coefficients in Boozer coordinates."""

    rmnc_b: FOURIER_COEFFICIENTS
    """The equilibrium flux surfaces R coordinate cosine Fourier coefficients in Boozer
    coordinates."""

    rmns_b: FOURIER_COEFFICIENTS
    """The equilibrium flux surfaces R coordinate sine Fourier coefficients in Boozer
    coordinates."""

    zmnc_b: FOURIER_COEFFICIENTS
    """The equilibrium flux surfaces Z coordinate cosine Fourier coefficients in Boozer
    coordinates."""

    zmns_b: FOURIER_COEFFICIENTS
    """The equilibrium flux surfaces Z coordinate sine Fourier coefficients in Boozer
    coordinates."""

    bmnc: FOURIER_COEFFICIENTS
    """The magnetic field strength cosine Fourier coefficients."""

    bmns: FOURIER_COEFFICIENTS
    """The magnetic field strength sine Fourier coefficients."""

    bsubumnc: FOURIER_COEFFICIENTS
    """The covariant u-component of the magnetic field cosine Fourier coefficients."""

    bsubumns: FOURIER_COEFFICIENTS
    """The covariant u-component of the magnetic field sine Fourier coefficients."""

    bsubvmnc: FOURIER_COEFFICIENTS
    """The covariant v-component of the magnetic field cosine Fourier coefficients."""

    bsubvmns: FOURIER_COEFFICIENTS
    """The covariant v-component of the magnetic field sine Fourier coefficients."""

    numnc_b: FOURIER_COEFFICIENTS
    """The nu cosine Fourier coefficients."""

    numns_b: FOURIER_COEFFICIENTS
    """The nu sine Fourier coefficients."""

    rmnc: FOURIER_COEFFICIENTS
    """The R coordinate cosine Fourier coefficients for all surfaces."""

    rmns: FOURIER_COEFFICIENTS
    """The R coordinate sine Fourier coefficients for all surfaces."""

    zmnc: FOURIER_COEFFICIENTS
    """The Z coordinate cosine Fourier coefficients for all surfaces."""

    zmns: FOURIER_COEFFICIENTS
    """The Z coordinate sine Fourier coefficients for all surfaces."""

    lmnc: FOURIER_COEFFICIENTS
    """The lambda cosine Fourier coefficients."""

    lmns: FOURIER_COEFFICIENTS
    """The lambda sine Fourier coefficients."""

    @property
    def n_boozer_fourier_modes(self) -> int:
        return (
            self.n_boozer_poloidal_modes * (2 * self.max_boozer_toroidal_mode + 1)
            - self.max_boozer_toroidal_mode
        )

    @property
    def n_fourier_modes(self) -> int:
        return (
            self.n_poloidal_modes * (2 * self.max_toroidal_mode + 1)
            - self.max_toroidal_mode
        )

    @property
    def n_nyquist_fourier_modes(self) -> int:
        return (
            self.n_nyquist_poloidal_modes * (2 * self.max_nyquist_toroidal_mode + 1)
            - self.max_nyquist_toroidal_mode
        )

    @property
    def n_boozer_flux_surfaces(self) -> int:
        return len(self.boozer_normalized_toroidal_flux)

    @property
    def n_flux_surfaces(self) -> int:
        return self.bmnc.shape[1]


class BoozerSettings(pydantic.BaseModel):
    normalized_toroidal_flux: list[float] | None = None
    """The normalized toroidal flux values at which the transformation to Boozer
    coordinates is performed.

    If None, the transformation to Boozer coordinates will be run on all flux surfaces.
    """

    n_poloidal_modes: int = 32
    """The number of poloidal modes with which the transformation to Boozer coordinates
    is performed.

    Corresponds to mboz.
    """

    max_toroidal_mode: int = 32
    """The maximum toroidal mode number with which the transformation to Boozer
    coordinates is performed.

    Corresponds to nboz.
    """

    verbose: bool = False
    """If True, running booz_xform in verbose mode."""


class BoozerPresetSettings(pydantic.BaseModel):
    """Presets to derive the BoozerSettings from equilibrium_resolution."""

    normalized_toroidal_flux: list[float] | None = pydantic.Field(min_length=1)
    """The normalized toroidal flux values at which the transformation to Boozer
    coordinates is performed.

    If None, the transformation to Boozer coordinates will be run on all flux surfaces.
    """

    verbose: bool = False
    """If True, running booz_xform in verbose mode."""


def run_boozer(
    equilibrium: vmec_utils.VmecppWOut,
    settings: BoozerSettings,
) -> BoozerOutput:
    vmec = vmec_utils.as_simsopt_vmec(equilibrium)
    boozer = mhd.Boozer(
        equil=vmec,
        mpol=settings.n_poloidal_modes,
        ntor=settings.max_toroidal_mode,
        verbose=settings.verbose,
    )
    if settings.normalized_toroidal_flux is not None:
        boozer.register(settings.normalized_toroidal_flux)

    boozer.run()

    return BoozerOutput(
        is_stellarator_symmetric=boozer.bx.asym,
        n_field_periods=boozer.bx.nfp,
        xm_b=boozer.bx.xm_b,
        xn_b=boozer.bx.xn_b,
        boozer_normalized_toroidal_flux=boozer.bx.s_b,
        normalized_toroidal_flux=boozer.bx.s_in,
        boozer_flux_surface_indices=boozer.bx.compute_surfs,
        aspect=boozer.bx.aspect,
        n_boozer_poloidal_modes=boozer.bx.mboz,
        max_boozer_toroidal_mode=boozer.bx.nboz,
        n_poloidal_modes=boozer.bx.mpol,
        max_toroidal_mode=boozer.bx.ntor,
        n_nyquist_poloidal_modes=boozer.bx.mpol_nyq,
        max_nyquist_toroidal_mode=boozer.bx.ntor_nyq,
        toroidal_flux=boozer.bx.toroidal_flux,
        xm=boozer.bx.xm,
        xm_nyq=boozer.bx.xm_nyq,
        xn=boozer.bx.xn,
        xn_nyq=boozer.bx.xn_nyq,
        phi=boozer.bx.phi,
        phip=boozer.bx.phip,
        G=boozer.bx.Boozer_G,
        I=boozer.bx.Boozer_I,
        G_all_flux_surfaces=boozer.bx.Boozer_G_all,
        I_all_flux_surfaces=boozer.bx.Boozer_I_all,
        iota=boozer.bx.iota,
        bmnc_b=boozer.bx.bmnc_b,
        bmns_b=boozer.bx.bmns_b,
        gmnc_b=boozer.bx.gmnc_b,
        gmns_b=boozer.bx.gmns_b,
        rmnc_b=boozer.bx.rmnc_b,
        rmns_b=boozer.bx.rmns_b,
        zmns_b=boozer.bx.zmns_b,
        zmnc_b=boozer.bx.zmnc_b,
        bmnc=boozer.bx.bmnc,
        bmns=boozer.bx.bmns,
        bsubumnc=boozer.bx.bsubumnc,
        bsubumns=boozer.bx.bsubumns,
        bsubvmnc=boozer.bx.bsubvmnc,
        bsubvmns=boozer.bx.bsubvmns,
        numnc_b=boozer.bx.numnc_b,
        numns_b=boozer.bx.numns_b,
        rmnc=boozer.bx.rmnc,
        rmns=boozer.bx.rmns,
        zmnc=boozer.bx.zmnc,
        zmns=boozer.bx.zmns,
        lmnc=boozer.bx.lmnc,
        lmns=boozer.bx.lmns,
    )


def create_boozer_settings_from_equilibrium_resolution(
    mhd_equilibrium: vmec_utils.VmecppWOut,
    settings: BoozerPresetSettings,
) -> BoozerSettings:
    """Derives Boozer transformation settings from the resolution of the equilibriumn
    according to the given boozer settings preset.

    See the wrapped function boozer.boozer_settings_from_equilibrium_resolution   for
    more details.
    """
    return boozer_settings_from_equilibrium_resolution(
        mhd_equilibrium=mhd_equilibrium,
        normalized_toroidal_flux=settings.normalized_toroidal_flux,
        verbose=settings.verbose,
    )


def boozer_settings_from_equilibrium_resolution(
    mhd_equilibrium: vmec_utils.VmecppWOut,
    normalized_toroidal_flux: list[float] | None = None,
    verbose: bool = False,
) -> BoozerSettings:
    """Set Boozer transformation settings based on the resolution of the equilibrium.

    The resolution is consistent with how the resolution is set in STELLOPT:
    https://github.com/PrincetonUniversity/STELLOPT/blob/develop/BOOZ_XFORM/Sources/read_wout_booz.f#L48

    Args:
        mhd_equilibrium: The VMEC equilibrium.
        normalized_toroidal_flux: The normalized toroidal flux values at which the
            transformation to Boozer coordinates is performed. If None, the
            transformation to Boozer coordinates will be run on all flux surfaces.
        verbose: If True, running booz_xform in verbose mode.
    """
    mpol = max(6 * mhd_equilibrium.mpol, 2)
    ntor = max(2 * mhd_equilibrium.ntor - 1, 0)
    return BoozerSettings(
        normalized_toroidal_flux=normalized_toroidal_flux,
        n_poloidal_modes=mpol,
        max_toroidal_mode=ntor,
        verbose=verbose,
    )
