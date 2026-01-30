import base64
import enum
import io
import pathlib
import tempfile

import jaxtyping as jt
import numpy as np
import pydantic
import vmecpp
from scipy import interpolate
from simsopt import mhd

from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import flux_power_series, ideal_mhd_parameters, vmec_settings

# We know that we are discarding coefficients, reduce output clutter.
vmecpp.logger.setLevel("ERROR")


def _deserialize_np_from_bytes(np_bytes: bytes) -> np.ndarray:
    """Loads a numpy array from raw bytes (dapper blob format)."""
    with io.BytesIO(np_bytes) as in_f:
        return np.load(in_f)


def _decode_blob_field(blob_dict: dict) -> np.ndarray:
    """Decodes a dapper blob-encoded field to a numpy array.

    Args:
        blob_dict: dict with keys: dapper_is_blob, content, file_suffix, content_length

    Returns:
        The decoded numpy array
    """
    if not isinstance(blob_dict, dict) or not blob_dict.get("dapper_is_blob"):
        raise ValueError(f"Expected a blob dict, got {type(blob_dict)}")

    # Decode base64 content to bytes
    encoded_content = blob_dict["content"]
    decoded_bytes = base64.b64decode(encoded_content, validate=True)

    # Deserialize numpy array from bytes
    return _deserialize_np_from_bytes(decoded_bytes)


def _decode_all_blobs_in_dict(data: dict) -> dict:
    """Recursively decodes all blob-encoded fields in a dict.

    Args:
        data: A dict that may contain blob-encoded fields

    Returns:
        The same dict with all blob fields decoded to lists (for Pydantic)
    """
    for key, value in data.items():
        if isinstance(value, dict) and value.get("dapper_is_blob") is True:
            # Decode blob and convert to list for Pydantic validation
            arr = _decode_blob_field(value)
            data[key] = arr.tolist()
    return data


# Renamed fields for backwards compatibility. Renaming is applied first.
RENAMED_FIELDS = {
    "maximum_iterations": "niter",
    "sign_of_jacobian": "signgs",
    "betatot": "betatotal",
    "VolAvgB": "volavgB",
    "iota_full": "iotaf",
    "safety_factor": "q_factor",
    "pressure_full": "presf",
    "pressure_half": "pres",
    "toroidal_flux": "phi",
    "poloidal_flux": "chi",
    "beta": "beta_vol",
    "spectral_width": "specw",
    "Dshear": "DShear",
    "Dwell": "DWell",
    "Dcurr": "DCurr",
    "Dgeod": "DGeod",
    "raxis_c": "raxis_cc",
    "zaxis_s": "zaxis_cs",
    "dVds": "vp",
    "overr": "over_r",
    "iota_half": "iotas",
    "volume_p": "volume",
    "version": "version_",
}
# Fields are first renamed, so use the new names here.
PADDED_FIELDS = [
    "iotas",
    "mass",
    "pres",
    "beta_vol",
    "buco",
    "bvco",
    "vp",
    "phips",
    "over_r",
]

PADDED_AND_TRANSPOSED_FIELDS = [
    "lmns",
    "lmnc",
    "gmnc",
    "gmns",
    "bmnc",
    "bmns",
    "bsubumnc",
    "bsubvmnc",
    "bsupumnc",
    "bsupvmnc",
    "bsubumns",
    "bsubvmns",
    "bsupumns",
    "bsupvmns",
]

ONLY_TRANSPOSED_FIELDS = [
    "rmnc",
    "zmns",
    "rmns",
    "zmnc",
    "lmns_full",
    "lmnc_full",
    "bsubsmns",
    "bsubsmnc",
]


class VmecppWOut(vmecpp.VmecWOut):
    @classmethod
    def convert_cpp_format(cls, data: dict) -> dict:
        for old_name, new_name in RENAMED_FIELDS.items():
            if old_name in data:
                data[new_name] = data.pop(old_name)

        # Previously defaults could be either empty list or None when unset,
        # now we want to let vmecpp handle defaulting fields correctly.
        LEGAL_EMPTY_LISTS = ["curlabel", "extcur"]
        keys_to_remove = [
            key
            for key in data
            if key not in LEGAL_EMPTY_LISTS and (data[key] is None or data[key] == [])
        ]
        for field in keys_to_remove:
            del data[field]

        for field in PADDED_FIELDS:
            if field in data:
                data[field] = [0.0] + data[field]

        for field in PADDED_AND_TRANSPOSED_FIELDS:
            if field in data:
                arr = np.array(data[field])
                data[field] = (
                    vmecpp._pad_and_transpose(arr, arr.shape[1])
                ).tolist()  # pyright: ignore[reportOptionalMemberAccess]

        for field in ONLY_TRANSPOSED_FIELDS:
            if field in data:
                data[field] = (np.array(data[field]).T).tolist()

        return data

    @classmethod
    def decode_binary_blob_data(cls, data):
        """Handle binary blob-encoded arrays, base64-encoded numpy arrays.
        They are stored in dicts with the following structure:
           - dapper_is_blob: True
           - content: base64-encoded string
           - file_suffix: file extension (e.g., ".npy")
           - content_length: size in bytes
        """
        if isinstance(data, dict):
            has_blob_encoded_fields = any(
                isinstance(v, dict) and v.get("dapper_is_blob") is True
                for v in data.values()
            )

            if has_blob_encoded_fields:
                # Decode all blob-encoded fields to lists for Pydantic validation
                data = _decode_all_blobs_in_dict(data)
        return data

    @pydantic.model_validator(mode="before")
    @classmethod
    def ensure_backwards_compatibility(cls, data):
        """Ensure backwards compatibility with older versions of the wout file."""
        data = cls.decode_binary_blob_data(data)

        # assert isinstance(data, dict)
        if "iota_half" in data:
            # Old naming convention, probably from constellaration <= 0.2.2
            ns = data["DMerc"]
            assert "dVds" in data
            assert "Dshear" in data
            assert "DShear" not in data
            # Make sure the old fields are padded as expected
            ns = data["ns"]
            assert len(data["dVds"]) == ns - 1

            data = VmecppWOut.convert_cpp_format(data)

            assert "iota_half" not in data
            assert "iotas" in data

        return data

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
    return VmecppWOut.model_validate(wout, from_attributes=True)


def as_simsopt_vmec(equilibrium: VmecppWOut) -> mhd.Vmec:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        wout_path = tmpdir / "wout_temp.nc"
        extras = None
        if equilibrium.model_extra is not None:
            extras = set(equilibrium.model_extra.keys())
        VmecppWOut.model_validate(equilibrium.model_dump(exclude=extras)).save(
            wout_path,
        )
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
        fourier_coefficients=equilibrium.bmnc.T[1:, :],
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
