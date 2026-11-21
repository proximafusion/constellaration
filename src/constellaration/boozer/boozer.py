from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pydantic
from booz_xform_jax.jax_api import (
    BoozerConfig,
    booz_xform_jax_impl,
    prepare_booz_xform_plan,
)
from vmecpp import enable_x64  # pyright: ignore[reportAttributeAccessIssue]

from constellaration.mhd import vmec_utils

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
    """The maximum Boozer toroidal mode number."""

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
    """The equilibrium Jacobian cosine Fourier coefficients in Boozer coordinates."""

    gmns_b: FOURIER_COEFFICIENTS
    """The equilibrium Jacobian sine Fourier coefficients in Boozer coordinates."""

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
    """If True, runs booz_xform in verbose mode."""


class BoozerPresetSettings(pydantic.BaseModel):
    """Presets to derive the BoozerSettings from equilibrium_resolution."""

    normalized_toroidal_flux: list[float] | None = pydantic.Field(min_length=1)
    """The normalized toroidal flux values at which the transformation to Boozer
    coordinates is performed.

    If None, the transformation to Boozer coordinates will be run on all flux surfaces.
    """

    verbose: bool = False
    """If True, runs booz_xform in verbose mode."""


class BoozerSpectrum(NamedTuple):
    """Selected half-grid surfaces. Spectra are indexed (Fourier mode, surface)."""

    xm_b: jax.Array
    xn_b: jax.Array
    s: jax.Array
    iota: jax.Array
    bmnc_b: jax.Array
    bmns_b: jax.Array | None


def make_transform(
    modes: Mapping[str, jax.typing.ArrayLike],
    *,
    nfp: int,
    ns: int,
    mboz: int,
    nboz: int,
    surfaces: Sequence[float] | None,
    asym: bool = False,
    magnetic_only: bool = False,
) -> Callable[[Mapping[str, jax.Array]], tuple[BoozerSpectrum, dict]]:
    """Prepare a JAX transform for VMEC output at a fixed resolution.

    This adapter converts VMEC's full-grid geometry to Boozer's half-grid
    inputs and prepares the quadrature and surface selection outside tracing.
    """
    mode_arrays = {
        name: jnp.asarray(modes[name]) for name in ("xm", "xn", "xm_nyq", "xn_nyq")
    }
    plan = prepare_booz_xform_plan(
        nfp=nfp,
        asym=asym,
        xm=np.asarray(modes["xm"]).tolist(),
        xn=np.asarray(modes["xn"]).tolist(),
        xm_nyq=np.asarray(modes["xm_nyq"]).tolist(),
        xn_nyq=np.asarray(modes["xn_nyq"]).tolist(),
        config=BoozerConfig(mboz=mboz, nboz=nboz, trig_f32=False),
    )
    s_full = np.linspace(0, 1, ns)
    s_half = (s_full[1:] + s_full[:-1]) / 2
    selected = (
        np.arange(ns - 1)
        if surfaces is None
        else np.unique([np.argmin(np.abs(s_half - s)) for s in surfaces])
    )
    s = jnp.asarray(s_half[selected])
    indices = jnp.asarray(selected)
    rho = jnp.asarray(np.sqrt(s_full)).at[0].set(1)
    rho_half = jnp.asarray(np.sqrt(s_half))
    odd = mode_arrays["xm"] % 2 != 0

    def half_geometry(coefficients):
        c = jnp.asarray(coefficients).T
        regular = c / rho[:, None]
        mean = (c[:-1] + c[1:]) / 2
        odd_mean = (regular[:-1] + regular[1:]) * (rho_half[:, None] / 2)
        half = jnp.where(odd, odd_mean, mean)
        # VMEC's m=1 coefficients are linear in rho at the axis; extrapolate
        # c/rho using the two first interior points, as in BOOZ_XFORM.
        first = rho_half[0] * (1.5 * regular[1] - 0.5 * regular[2])
        return half.at[0].set(jnp.where(mode_arrays["xm"] == 1, first, half[0]))

    def transform(wout):
        fields: dict[str, Any] = {
            name: half_geometry(wout[name]) for name in ("rmnc", "zmns")
        }
        half_names = ["lmns", "bmnc", "bsubumnc", "bsubvmnc"]
        if asym:
            fields.update(
                {name: half_geometry(wout[name]) for name in ("rmns", "zmnc")}
            )
            half_names += ["lmnc", "bmns", "bsubumns", "bsubvmns"]
        fields.update({name: jnp.asarray(wout[name]).T[1:] for name in half_names})
        out = booz_xform_jax_impl(
            **fields,
            iota=jnp.asarray(wout["iotas"])[1:],
            xm=mode_arrays["xm"],
            xn=mode_arrays["xn"],
            xm_nyq=mode_arrays["xm_nyq"],
            xn_nyq=mode_arrays["xn_nyq"],
            constants=plan.constants,
            grids=plan.grids,
            surface_indices=indices,
            plan=plan,
            magnetic_only=magnetic_only,
        )
        return (
            BoozerSpectrum(
                out["ixm_b"],
                out["ixn_b"],
                s,
                out["iota_b"],
                out["bmnc_b"].T,
                out["bmns_b"].T if asym else None,
            ),
            out if magnetic_only else {**out, **fields},
        )

    return transform


def run_boozer(
    equilibrium: vmec_utils.VmecppWOut,
    settings: BoozerSettings,
) -> BoozerOutput:
    """Transform a reporting wout using the same JAX kernel as optimization."""
    w = equilibrium
    keys = ("xm", "xn", "xm_nyq", "xn_nyq")
    modes = {name: np.asarray(getattr(w, name)) for name in keys}
    input_names = ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc")
    asym_names = ("rmns", "zmnc", "lmnc", "bmns", "bsubumns", "bsubvmns")
    fields = {
        name: getattr(w, name)
        for name in (*input_names, *(asym_names if w.lasym else ()), "iotas")
    }
    with enable_x64(True):
        transform = make_transform(
            modes,
            nfp=w.nfp,
            ns=w.ns,
            mboz=settings.n_poloidal_modes,
            nboz=settings.max_toroidal_mode,
            surfaces=settings.normalized_toroidal_flux,
            asym=w.lasym,
        )
        spectrum, output = transform(fields)
    empty = np.empty((0, 0))
    symmetric = ("bmnc_b", "gmnc_b", "rmnc_b", "zmns_b", "numns_b")
    asymmetric = ("bmns_b", "gmns_b", "rmns_b", "zmnc_b", "numnc_b")
    return BoozerOutput(
        is_stellarator_symmetric=not w.lasym,
        n_field_periods=w.nfp,
        xm_b=np.asarray(spectrum.xm_b),
        xn_b=np.asarray(spectrum.xn_b),
        boozer_normalized_toroidal_flux=np.asarray(spectrum.s),
        normalized_toroidal_flux=w.normalized_toroidal_flux_half_grid_mesh[1:],
        boozer_flux_surface_indices=(np.asarray(output["jlist"]) - 2).tolist(),
        aspect=float(w.aspect),
        n_boozer_poloidal_modes=settings.n_poloidal_modes,
        max_boozer_toroidal_mode=settings.max_toroidal_mode,
        n_poloidal_modes=w.mpol,
        max_toroidal_mode=w.ntor,
        n_nyquist_poloidal_modes=int(modes["xm_nyq"].max()),
        max_nyquist_toroidal_mode=int(modes["xn_nyq"].max() // w.nfp),
        toroidal_flux=float(w.phi[-1]),
        **modes,
        phi=np.asarray(w.phi),
        phip=np.asarray(w.phipf),
        G=np.asarray(output["bvco_b"]),
        I=np.asarray(output["buco_b"]),
        G_all_flux_surfaces=np.asarray(w.bsubvmnc[0, 1:]),
        I_all_flux_surfaces=np.asarray(w.bsubumnc[0, 1:]),
        iota=np.asarray(w.iotas[1:]),
        **{name: np.asarray(output[name]).T for name in symmetric},
        **{
            name: np.asarray(output[name]).T if w.lasym else empty
            for name in asymmetric
        },
        **{name: np.asarray(output[name]).T for name in input_names},
        **{
            name: np.asarray(output[name]).T if w.lasym else empty
            for name in asym_names
        },
    )


def create_boozer_settings_from_equilibrium_resolution(
    mhd_equilibrium: vmec_utils.VmecppWOut,
    settings: BoozerPresetSettings,
) -> BoozerSettings:
    """Derives Boozer transformation settings from the resolution of the equilibrium
    according to the given Boozer settings preset.

    See the wrapped function boozer.boozer_settings_from_equilibrium_resolution for
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
        verbose: If True, runs booz_xform in verbose mode.
    """
    mpol = max(6 * mhd_equilibrium.mpol, 2)
    ntor = max(2 * mhd_equilibrium.ntor - 1, 0)
    return BoozerSettings(
        normalized_toroidal_flux=normalized_toroidal_flux,
        n_poloidal_modes=mpol,
        max_toroidal_mode=ntor,
        verbose=verbose,
    )
