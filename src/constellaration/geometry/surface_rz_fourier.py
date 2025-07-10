from typing import Sequence

import jaxtyping as jt
import numpy as np
import pydantic
from jax import numpy as jnp
from simsopt import geo
from typing_extensions import Self
from vmecpp import _pydantic_numpy as pydantic_numpy

from constellaration.geometry import surface_utils
from constellaration.utils.types import NpOrJaxArray

FourierCoefficients = jt.Float[np.ndarray, "n_poloidal_modes n_toroidal_modes"]
FourierModes = jt.Int[np.ndarray, "n_poloidal_modes n_toroidal_modes"]


class SurfaceRZFourier(pydantic_numpy.BaseModelWithNumpy):
    r"""Represents a toroidal (homeomorphic to a torus) surface as a Fourier series.

    The surface maps the polodial angle theta and the toroidal angle phi to points in
    3D space expressed in cylindrical coordinates (r, phi, z).

        r(theta, phi) = sum_{m, n} r_{m, n}^{cos} cos(m theta - NFP n phi)
                             + r_{m, n}^{sin} sin(m theta - NFP n phi)
        z(theta, phi) = sum_{m, n} z_{m, n}^{sin} sin(m theta - NFP n phi)
                                + z_{m, n}^{cos} cos(m theta - n phi)
        phi(theta, phi) = phi

    where theta is in [0, 2 pi] and phi is in [0, 2 pi / NFP], and the sum is over
    integers m and n, where m is the poloidal mode index and n is the toroidal
    mode index, and NFP is the number of field periods, representing the degree
    of toroidal symmetry of the surface, meaning that:
        r(theta, phi + 2 pi / NFP) = r(theta, phi)
        z(theta, phi + 2 pi / NFP) = z(theta, phi)
    Note that phi can also be provided for the full range [0, 2 pi], but the results
    will be symmetric under a shift by 2 pi / NFP.

    The Fourier coefficients are stored in the following arrays:
    - r_cos: r_{m, n}^{cos}
    - r_sin: r_{m, n}^{sin}
    - z_sin: z_{m, n}^{sin}
    - z_cos: z_{m, n}^{cos}

    If r_sin and z_cos are None, then stellarator symmetry is assumed and viceversa.
    """

    r_cos: FourierCoefficients
    z_sin: FourierCoefficients
    r_sin: FourierCoefficients | None = None
    z_cos: FourierCoefficients | None = None

    n_field_periods: int = 1
    """Number of toroidal field periods of the surface."""

    is_stellarator_symmetric: bool = True
    """Indicates whether the surface possesses stellarator symmetry, which implies that
    r_sin and z_cos are identically zero and the arrays r_sin and z_cos are therefore
    set to None."""

    @property
    def n_poloidal_modes(self) -> int:
        """The number of poloidal modes in the Fourier series."""
        return self.r_cos.shape[0]

    @property
    def n_toroidal_modes(self) -> int:
        """The number of toroidal modes in the Fourier series."""
        return self.r_cos.shape[1]

    @property
    def max_poloidal_mode(self) -> int:
        """The maximum poloidal mode index."""
        return self.n_poloidal_modes - 1

    @property
    def max_toroidal_mode(self) -> int:
        """The maximum toroidal mode index."""
        return (self.n_toroidal_modes - 1) // 2

    @property
    def poloidal_modes(self) -> FourierModes:
        """A grid of poloidal mode indices."""
        return np.broadcast_to(
            np.arange(self.n_poloidal_modes)[:, None],
            (self.n_poloidal_modes, self.n_toroidal_modes),
        )

    @property
    def toroidal_modes(self) -> FourierModes:
        """A grid of toroidal mode indices."""
        return np.broadcast_to(
            np.arange(-self.max_toroidal_mode, self.max_toroidal_mode + 1),
            (self.n_poloidal_modes, self.n_toroidal_modes),
        )

    @pydantic.field_validator("r_cos")
    @classmethod
    def _check_odd_toroidal_modes(
        cls, r_cos: FourierCoefficients
    ) -> FourierCoefficients:
        if r_cos.shape[1] % 2 == 0:
            raise ValueError(
                "The number of toroidal modes should be odd: [-n, ..., 0, ..., n]."
            )
        return r_cos

    @pydantic.model_validator(mode="after")
    def _check_consistent_shapes(self) -> Self:
        shape = self.r_cos.shape
        if self.z_sin.shape != shape:
            raise ValueError("The shapes of r_cos and z_sin are different.")

        if not self.is_stellarator_symmetric:
            assert self.r_sin is not None
            if self.r_sin.shape != shape:
                raise ValueError("The shapes of r_cos and r_sin are different.")
            assert self.z_cos is not None
            if self.z_cos.shape != shape:
                raise ValueError("The shapes of r_cos and z_cos are different.")

        return self

    @pydantic.model_validator(mode="after")
    def _check_stellarator_symmetry(self) -> Self:
        if self.is_stellarator_symmetric:
            if self.r_sin is not None or self.z_cos is not None:
                raise ValueError(
                    "r_sin and z_cos should be None if is_stellarator_symmetric."
                )

            ntor = self.max_toroidal_mode
            if any(self.r_cos[0, :ntor] != 0.0):
                raise ValueError(
                    "r_cos for m=0 and n<0 must be 0.0 for "
                    "stellarator symmetric surfaces."
                )
            if any(self.z_sin[0, : ntor + 1] != 0.0):
                raise ValueError(
                    "z_sin for m=0 and n<=0 must be 0.0 for "
                    "stellarator symmetric surfaces."
                )

        elif self.r_sin is None or self.z_cos is None:
            raise ValueError(
                "r_sin and z_cos should not be None if not is_stellarator_symmetric."
            )

        return self


def from_simsopt(surface: geo.SurfaceRZFourier) -> SurfaceRZFourier:
    """Convert a SIMSOPT SurfaceRZFourier to a SurfaceRZFourier."""
    r_cos = surface.rc
    z_sin = surface.zs

    if not surface.stellsym:
        r_sin = surface.rs
        z_cos = surface.zc
    else:
        r_sin = None
        z_cos = None

    return SurfaceRZFourier(
        r_cos=r_cos,
        r_sin=r_sin,
        z_cos=z_cos,
        z_sin=z_sin,
        n_field_periods=int(surface.nfp),
        is_stellarator_symmetric=bool(surface.stellsym),
    )


def to_simsopt(
    surface: SurfaceRZFourier,
    theta_phi: jt.Float[np.ndarray, "n_theta n_phi 2"] | None = None,
) -> geo.SurfaceRZFourier:
    """Convert a surface in the types module to a simsopt surface RZ Fourier."""
    simsopt_surface = geo.SurfaceRZFourier(
        nfp=surface.n_field_periods,
        stellsym=surface.is_stellarator_symmetric,
        mpol=surface.max_poloidal_mode,
        ntor=surface.max_toroidal_mode,
        quadpoints_theta=(
            theta_phi[:, 0, 0] / (2 * np.pi) if theta_phi is not None else None
        ),
        quadpoints_phi=(
            theta_phi[0, :, 1] / (2 * np.pi) if theta_phi is not None else None
        ),
    )

    for m in range(surface.n_poloidal_modes):
        for n in range(
            -surface.max_toroidal_mode,
            surface.max_toroidal_mode + 1,
        ):
            rc = surface.r_cos[m, n + surface.max_toroidal_mode]
            simsopt_surface.set_rc(m, n, rc)

            zs = surface.z_sin[m, n + surface.max_toroidal_mode]
            simsopt_surface.set_zs(m, n, zs)

            if not surface.is_stellarator_symmetric:
                assert surface.r_sin is not None
                rs = surface.r_sin[m, n + surface.max_toroidal_mode]
                simsopt_surface.set_rs(m, n, rs)

                assert surface.z_cos is not None
                zc = surface.z_cos[m, n + surface.max_toroidal_mode]
                simsopt_surface.set_zc(m, n, zc)

    return simsopt_surface


def get_largest_non_zero_modes(
    surface: SurfaceRZFourier,
    tolerance: float = 1.0e-15,
) -> tuple[int, int]:
    """Return the largest non-zero poloidal and toroidal mode numbers of a
    SurfaceRZFourier.

    Args:
        surface: The surface to trim.
        tolerance: The tolerance for considering a coefficient as zero.
    """
    coeff_arrays = [surface.r_cos, surface.z_sin]
    if surface.r_sin is not None:
        coeff_arrays.append(surface.r_sin)
    if surface.z_cos is not None:
        coeff_arrays.append(surface.z_cos)

    max_m = 0
    max_n = 0

    for coeff in coeff_arrays:
        non_zero = np.abs(coeff) > tolerance
        if not np.any(non_zero):
            continue
        m_indices, n_indices = np.nonzero(non_zero)
        # Toroidal modes are stored as [-ntor, ..., 0, ..., ntor]
        # Shift n_indices such that it is the largest toroidal mode
        n_indices -= surface.max_toroidal_mode
        current_max_m = m_indices.max()
        current_max_n = n_indices.max()
        if current_max_m > max_m:
            max_m = current_max_m
        if current_max_n > max_n:
            max_n = current_max_n

    # Ensure at least one mode is retained
    return max(max_m, 0), max(max_n, 0)


def evaluate_minor_radius(
    surface: SurfaceRZFourier,
    n_theta: int = 50,
    n_phi: int = 51,
) -> pydantic.NonNegativeFloat:
    """Return the minor radius of the surface defined as the radius of the circle with
    the same area as the average cross sectional area of the surface.

    Args:
        surface: The surface to compute the minor radius of.
        n_theta: Number of quadrature points in the theta dimension of the surface,
            used for the numerical integration.
        n_phi: Number of quadrature points in the phi dimension of the surface,
            used for the numerical integration.
    """
    return np.sqrt(
        compute_mean_cross_sectional_area(surface=surface, n_theta=n_theta, n_phi=n_phi)
        / np.pi
    )


def compute_mean_cross_sectional_area(
    surface: SurfaceRZFourier,
    n_theta: int = 50,
    n_phi: int = 51,
) -> pydantic.NonNegativeFloat:
    """Compute the mean cross sectional area of the surface.

    The mean cross sectional area is defined as the average of the cross sectional areas
    of the surface along the toroidal direction.

    The code is taken from Simsopt, please refer to the documentation here
    https://simsopt.readthedocs.io/en/latest/simsopt.geo.html#simsopt.geo.surface.Surface.mean_cross_sectional_area

    The code provides a numerical integration of the cross sectional area of the surface
    and an average across the toroidal direction, which is general enough to not assume
    that phi is the real toroidal angle.

    Args:
        surface: The surface to compute the mean cross sectional area of.
        n_theta: Number of quadrature points in the theta dimension of the surface,
            used for the numerical integration.
        n_phi: Number of quadrature points in the phi dimension of the surface,
            used for the numerical integration.
    """
    # n_theta - 1, n_phi - 1 is to make sure this calculation is equivalent to using
    # Simsopt
    theta_phi_grid = surface_utils.make_theta_phi_grid(
        n_theta - 1, n_phi - 1, phi_upper_bound=2 * np.pi, include_endpoints=False
    )
    xyz = evaluate_points_xyz(surface, theta_phi_grid)
    x2y2 = xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2
    dgamma1 = evaluate_dxyz_dphi(surface, theta_phi_grid) * 2 * np.pi
    dgamma2 = evaluate_dxyz_dtheta(surface, theta_phi_grid) * 2 * np.pi

    # compute the average cross sectional area
    J = np.zeros((xyz.shape[0], xyz.shape[1], 2, 2))
    J[:, :, 0, 0] = (
        xyz[:, :, 0] * dgamma1[:, :, 1] - xyz[:, :, 1] * dgamma1[:, :, 0]
    ) / x2y2
    J[:, :, 0, 1] = (
        xyz[:, :, 0] * dgamma2[:, :, 1] - xyz[:, :, 1] * dgamma2[:, :, 0]
    ) / x2y2
    J[:, :, 1, 0] = 0.0
    J[:, :, 1, 1] = 1.0

    detJ = np.linalg.det(J)
    Jinv = np.linalg.inv(J)

    dZ_dtheta = (
        dgamma1[:, :, 2] * Jinv[:, :, 0, 1] + dgamma2[:, :, 2] * Jinv[:, :, 1, 1]
    )
    mean_cross_sectional_area = np.abs(np.mean(np.sqrt(x2y2) * dZ_dtheta * detJ)) / (
        2 * np.pi
    )
    return mean_cross_sectional_area


def evaluate_points_xyz(
    surface: SurfaceRZFourier, theta_phi: jt.Float[np.ndarray, "*dims 2"]
) -> jt.Float[np.ndarray, "*dims 3"]:
    """Evaluate the X, Y, and Z coordinates of the surface at the given theta and phi
    coordinates.

    Args:
        surface: The surface to evaluate.
        theta_phi: The theta and phi coordinates at which to evaluate the surface.
            The last dimension is supposed to index theta and phi.

    Returns:
        The X, Y, and Z coordinates of the surface at the given
            theta and phi coordinates.
        The last dimension indexes X, Y, and Z.
    """
    rz = evaluate_points_rz(surface, theta_phi)
    phi = theta_phi[..., 1]
    x = rz[..., 0] * np.cos(phi)
    y = rz[..., 0] * np.sin(phi)
    z = rz[..., 1]
    return np.stack((x, y, z), axis=-1)


def evaluate_points_rz(
    surface: SurfaceRZFourier,
    theta_phi: jt.Float[np.ndarray, "*dims 2"],
) -> jt.Float[np.ndarray, "*dims 2"]:
    """Evaluate the R and Z coordinates of the surface at the given theta and phi
    coordinates.

    Args:
        surface: The surface to evaluate.
        theta_phi: The theta and phi coordinates at which to evaluate the surface.
            The last dimension is supposed to index theta and phi.

    Returns:
        The R and Z coordinates of the surface at the given theta and phi coordinates.
        The last dimension indexes R and Z.
    """
    angle = _compute_angle(surface, theta_phi)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    # r_cos is (n_poloidal_modes, n_toroidal_modes)
    r = np.sum(surface.r_cos[np.newaxis, :, :] * cos_angle, axis=(-1, -2))
    z = np.sum(surface.z_sin[np.newaxis, :, :] * sin_angle, axis=(-1, -2))
    if not surface.is_stellarator_symmetric:
        assert surface.r_sin is not None
        assert surface.z_cos is not None
        r += np.sum(surface.r_sin[np.newaxis, :, :] * sin_angle, axis=(-1, -2))
        z += np.sum(surface.z_cos[np.newaxis, :, :] * cos_angle, axis=(-1, -2))
    return np.stack((r, z), axis=-1)


def evaluate_dxyz_dphi(
    surface: SurfaceRZFourier, theta_phi: jt.Float[np.ndarray, "*dims 2"]
) -> jt.Float[np.ndarray, "*dims 3"]:
    """Evaluate the derivatives of the X, Y, and Z coordinates of the surface with
    respect to phi.

    Args:
        surface: The surface to evaluate.
        theta_phi: The theta and phi coordinates at which to evaluate the surface.
            The last dimension is supposed to index theta and phi.
            The grid is expected to be using the `indexing='ij'` ordering in the jargon
            of `np.meshgrid`, which is also what `make_theta_phi_grid` produces.

    Returns:
        The derivatives of the X, Y, and Z coordinates of the surface with respect to
            phi.
        The last dimension indexes X, Y, and Z.
    """
    r = evaluate_points_rz(surface, theta_phi)[..., 0]
    dr_dphi = _evaluate_dr_dphi(surface, theta_phi)
    dz_dphi = _evaluate_dz_dphi(surface, theta_phi)
    phi = theta_phi[..., 1]
    dx_dphi = dr_dphi * np.cos(phi) - r * np.sin(phi)
    dy_dphi = dr_dphi * np.sin(phi) + r * np.cos(phi)
    dz_dphi = dz_dphi
    return np.stack((dx_dphi, dy_dphi, dz_dphi), axis=-1)


def evaluate_dxyz_dtheta(
    surface: SurfaceRZFourier, theta_phi: jt.Float[np.ndarray, "*dims 2"]
) -> jt.Float[np.ndarray, "*dims 3"]:
    """Evaluate the derivatives of the X, Y, and Z coordinates of the surface with
    respect to theta.

    Args:
        surface: The surface to evaluate.
        theta_phi: The theta and phi coordinates at which to evaluate the surface.
            The last dimension is supposed to index theta and phi.

    Returns:
        The derivatives of the X, Y, and Z coordinates of the surface with respect to
            theta.
        The last dimension indexes X, Y, and Z.
    """
    dr_dtheta = _evaluate_dr_dtheta(surface, theta_phi)
    dz_dtheta = _evaluate_dz_dtheta(surface, theta_phi)
    phi = theta_phi[..., 1]
    dx_dtheta = dr_dtheta * np.cos(phi)
    dy_dtheta = dr_dtheta * np.sin(phi)
    dz_dtheta = dz_dtheta
    return np.stack((dx_dtheta, dy_dtheta, dz_dtheta), axis=-1)


def set_max_mode_numbers(
    surface: SurfaceRZFourier,
    max_poloidal_mode: int,
    max_toroidal_mode: int,
) -> SurfaceRZFourier:
    """Adjusts the Fourier coefficient arrays of a SurfaceRZFourier object to the
    specified maximum poloidal and toroidal mode numbers.

    Coefficient arrays will be resized accordingly, and coefficients will be copied over
    where modes overlap. If the new mode numbers are larger, the arrays will be padded
    with zeros. If smaller, the arrays will be truncated.

    Args:
        surface: The SurfaceRZFourier object to adjust.
        max_poloidal_mode: The new maximum poloidal mode number (m).
        max_toroidal_mode: The new maximum toroidal mode number (n).

    Returns:
        A new SurfaceRZFourier object with adjusted Fourier coefficient arrays.
    """

    # New array sizes
    new_n_poloidal_modes = max_poloidal_mode + 1
    new_n_toroidal_modes = 2 * max_toroidal_mode + 1  # Indices from -max_n to +max_n

    # Existing array sizes
    old_n_poloidal_modes = surface.n_poloidal_modes
    old_max_toroidal_mode = surface.max_toroidal_mode

    # Create new arrays filled with zeros
    new_r_cos = np.zeros(
        (new_n_poloidal_modes, new_n_toroidal_modes), dtype=surface.r_cos.dtype
    )
    new_z_sin = np.zeros_like(new_r_cos)

    if surface.r_sin is not None:
        new_r_sin = np.zeros_like(new_r_cos)
    else:
        new_r_sin = None

    if surface.z_cos is not None:
        new_z_cos = np.zeros_like(new_r_cos)
    else:
        new_z_cos = None

    # Determine overlapping m indices
    m_end = min(old_n_poloidal_modes, new_n_poloidal_modes)

    # Determine overlapping n values
    overlapping_n_start = -min(old_max_toroidal_mode, max_toroidal_mode)
    overlapping_n_end = min(old_max_toroidal_mode, max_toroidal_mode)

    # Compute indices in old and new arrays
    old_n_idx_start = overlapping_n_start + old_max_toroidal_mode
    old_n_idx_end = overlapping_n_end + old_max_toroidal_mode + 1

    new_n_idx_start = overlapping_n_start + max_toroidal_mode
    new_n_idx_end = overlapping_n_end + max_toroidal_mode + 1

    # Copy over the overlapping coefficients
    new_r_cos[:m_end, new_n_idx_start:new_n_idx_end] = surface.r_cos[
        :m_end, old_n_idx_start:old_n_idx_end
    ]

    new_z_sin[:m_end, new_n_idx_start:new_n_idx_end] = surface.z_sin[
        :m_end, old_n_idx_start:old_n_idx_end
    ]

    if surface.r_sin is not None and new_r_sin is not None:
        new_r_sin[:m_end, new_n_idx_start:new_n_idx_end] = surface.r_sin[
            :m_end, old_n_idx_start:old_n_idx_end
        ]

    if surface.z_cos is not None and new_z_cos is not None:
        new_z_cos[:m_end, new_n_idx_start:new_n_idx_end] = surface.z_cos[
            :m_end, old_n_idx_start:old_n_idx_end
        ]

    # Create a new SurfaceRZFourier object with the new arrays
    return surface.model_copy(
        update=dict(
            r_cos=new_r_cos,
            z_sin=new_z_sin,
            r_sin=new_r_sin,
            z_cos=new_z_cos,
        )
    )


def compute_infinity_norm_spectrum_scaling_fun(
    poloidal_modes: jt.Int[np.ndarray, " n_modes"],
    toroidal_modes: jt.Int[np.ndarray, " n_modes"],
    alpha: float,
) -> jt.Float[np.ndarray, " n_modes"]:
    r"""Compute a scale for SurfaceRZFourier Fourier coefficients based on a L-infinity
    norm of the modes.

    The spectrum scaling is computed as:

    ... math::

        e^{-alpha * max(|m|, |n|)}

    where `alpha` is a constant that determines the scaling of the Fourier
    spectrum.
    """
    infinity_norm = np.maximum(np.abs(poloidal_modes), np.abs(toroidal_modes))
    return np.exp(-1.0 * alpha * infinity_norm)


def build_mask(
    surface: SurfaceRZFourier,
    max_poloidal_mode: int,
    max_toroidal_mode: int,
) -> jt.PyTree[bool]:
    """Build an almost rectangular, boolean mask for Fourier coefficients of surfaces.

    Note: Almost rectangular because the coefficients where poloidal indices are zero
     AND toroidal indices are non-positive are always masked out.

    Args:
        surface: The surface to derive the mask from, based on its modes.
        max_poloidal_mode: The maximum poloidal mode to include.
        max_toroidal_mode: The maximum toroidal mode to include.

    Returns:
        A surface-like pytree with booleans for Fourier coefficients.
    """
    if not surface.is_stellarator_symmetric:
        raise ValueError("Masks for non-stellarator-symmetric surfaces not supported.")
    fourier_coefficients_mask = jnp.asarray(
        (surface.poloidal_modes > 0)
        | ((surface.poloidal_modes == 0) & (surface.toroidal_modes >= 1))
    )
    fourier_coefficients_mask &= (surface.poloidal_modes <= max_poloidal_mode) & (
        np.abs(surface.toroidal_modes) <= max_toroidal_mode
    )
    return surface.model_copy(
        update=dict(
            r_cos=fourier_coefficients_mask,
            z_sin=fourier_coefficients_mask,
            r_sin=False,
            z_cos=False,
        ),
    )


def get_named_mode_values(
    boundary: SurfaceRZFourier,
) -> dict[str, float]:
    """Extract the Fourier mode values from a SurfaceRZFourier object and return them as
    a dictionary with named keys.

    Args:
        boundary: The boundary to extract the mode values from.

    Returns:
        A dictionary with the mode values as floats, indexed by the mode names.

    Example:
    ```python
    boundary = rz_fourier_types.SurfaceRZFourier(
        r_cos = np.array([[0.0, 1.0, 3.0], [4.0, 5.0, 6.0]]),
        z_sin = np.array([[0.0, 0.0, 0.1], [10.0, 11.0, 12.0]]),
        n_field_periods = 2,
        is_stellarator_symmetric = True,
    )
    mode_values = get_named_mode_values(boundary)

    print(mode_values)

    # Returns:
    # {'r_cos(0, 0)': 1.0,
    # 'r_cos(0, 1)': 3.0,
    # 'r_cos(1, -1)': 4.0,
    # 'r_cos(1, 0)': 5.0,
    # 'r_cos(1, 1)': 6.0,
    # 'z_sin(0, 1)': 0.1,
    # 'z_sin(1, -1)': 10.0,
    # 'z_sin(1, 0)': 11.0,
    # 'z_sin(1, 1)': 12.0}
    ```
    """

    def _get_m_n_from_row_col(row: int, col: int) -> tuple[int, int]:
        return (
            int(boundary.poloidal_modes[row, col]),
            int(boundary.toroidal_modes[row, col]),
        )

    # r_cos
    mode_values = {
        f"r_cos{_get_m_n_from_row_col(row, col)}": boundary.r_cos[row, col]
        for row in range(boundary.n_poloidal_modes)
        for col in range(boundary.n_toroidal_modes)
        if not (
            boundary.poloidal_modes[row, col] == 0
            and boundary.toroidal_modes[row, col] < 0
            and boundary.is_stellarator_symmetric
        )
    }
    # z_sine
    mode_values.update(
        {
            f"z_sin{_get_m_n_from_row_col(row, col)}": boundary.z_sin[row, col]
            for row in range(boundary.n_poloidal_modes)
            for col in range(boundary.n_toroidal_modes)
            if not (
                boundary.poloidal_modes[row, col] == 0
                and boundary.toroidal_modes[row, col] <= 0
                and boundary.is_stellarator_symmetric
            )
        }
    )
    if boundary.r_sin is not None:
        # r_sine
        mode_values.update(
            {
                f"r_sin{_get_m_n_from_row_col(row, col)}": boundary.r_sin[row, col]
                for row in range(boundary.n_poloidal_modes)
                for col in range(boundary.n_toroidal_modes)
            }
        )
    if boundary.z_cos is not None:
        # z_cosine
        mode_values.update(
            {
                f"z_cos{_get_m_n_from_row_col(row, col)}": boundary.z_cos[row, col]
                for row in range(boundary.n_poloidal_modes)
                for col in range(boundary.n_toroidal_modes)
            }
        )

    return mode_values


def boundary_from_named_modes(
    named_fourier_modes: dict[str, float],
    is_stellarator_symmetric: bool,
    n_field_periods: int,
) -> SurfaceRZFourier:
    """Constructs a SurfaceRZFourier object from named Fourier modes.

    Args:
        named_fourier_modes: A dictionary where keys are named Fourier modes
            (e.g., "r_cos(0, 0)", "z_sin(1, 2)") and values are the Fourier
            coefficients.
        is_stellarator_symmetric: Indicates whether the surface has stellarator
            symmetry.
        n_field_periods: The number of toroidal field periods of the surface.

    Returns:
        A SurfaceRZFourier object reconstructed from the named Fourier modes.
    """
    # Extract maximum poloidal and toroidal mode indices from the keys
    max_m = max(int(key.split("(")[1].split(",")[0]) for key in named_fourier_modes)
    max_n = max(
        abs(int(key.split("(")[1].split(",")[1][:-1])) for key in named_fourier_modes
    )

    # Initialize Fourier coefficient arrays
    r_cos = np.zeros((max_m + 1, 2 * max_n + 1))
    z_sin = np.zeros((max_m + 1, 2 * max_n + 1))
    r_sin = None
    z_cos = None

    if not is_stellarator_symmetric:
        r_sin = np.zeros((max_m + 1, 2 * max_n + 1))
        z_cos = np.zeros((max_m + 1, 2 * max_n + 1))

    # Populate Fourier coefficient arrays
    for key, value in named_fourier_modes.items():
        mode_type, indices = key.split("(")
        m, n = map(int, indices[:-1].split(","))
        n_shifted = n + max_n  # Adjust n index to match array dimensions

        if mode_type == "r_cos":
            r_cos[m, n_shifted] = value
        elif mode_type == "z_sin":
            z_sin[m, n_shifted] = value
        elif (
            mode_type == "r_sin" and not is_stellarator_symmetric and r_sin is not None
        ):
            r_sin[m, n_shifted] = value
        elif (
            mode_type == "z_cos" and not is_stellarator_symmetric and z_cos is not None
        ):
            z_cos[m, n_shifted] = value

    return SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        r_sin=r_sin,
        z_cos=z_cos,
        n_field_periods=n_field_periods,
        is_stellarator_symmetric=is_stellarator_symmetric,
    )


def _compute_angle(
    surface: SurfaceRZFourier,
    theta_phi: jt.Float[np.ndarray, "*dims 2"],
) -> jt.Float[np.ndarray, "*dims n_poloidal_modes n_toroidal_modes"]:
    # angle is the argument of sin and cos in the Fourier series
    # angle = m*theta - NFP*n*phi
    angle: jt.Float[np.ndarray, "*dims n_poloidal_modes n_toroidal_modes"] = (
        surface.poloidal_modes * theta_phi[..., 0][..., np.newaxis, np.newaxis]
        - surface.n_field_periods
        * surface.toroidal_modes
        * theta_phi[..., 1][..., np.newaxis, np.newaxis]
    )
    return angle


def _evaluate_dr_dphi(
    surface: SurfaceRZFourier, theta_phi: jt.Float[np.ndarray, "*dims 2"]
) -> jt.Float[np.ndarray, "*dims"]:
    angle = _compute_angle(surface, theta_phi)
    sin_angle = np.sin(angle)
    # r_cos is (n_poloidal_modes, n_toroidal_modes)
    dr_dphi = np.sum(
        surface.r_cos[np.newaxis, :, :]
        * surface.n_field_periods
        * surface.toroidal_modes
        * sin_angle,
        axis=(-1, -2),
    )
    if not surface.is_stellarator_symmetric:
        assert surface.r_sin is not None
        cos_angle = np.cos(angle)
        dr_dphi += np.sum(
            surface.r_sin[np.newaxis, :, :]
            * surface.n_field_periods
            * surface.toroidal_modes
            * (-1)
            * cos_angle,
            axis=(-1, -2),
        )
    return dr_dphi


def _evaluate_dz_dphi(
    surface: SurfaceRZFourier, theta_phi: jt.Float[np.ndarray, "*dims 2"]
) -> jt.Float[np.ndarray, "*dims"]:
    angle = _compute_angle(surface, theta_phi)
    cos_angle = np.cos(angle)
    # z_sin is (n_poloidal_modes, n_toroidal_modes)
    dz_dphi = np.sum(
        surface.z_sin[np.newaxis, :, :]
        * surface.n_field_periods
        * surface.toroidal_modes
        * (-1)
        * cos_angle,
        axis=(-1, -2),
    )
    if not surface.is_stellarator_symmetric:
        assert surface.z_cos is not None
        sin_angle = np.sin(angle)
        dz_dphi += np.sum(
            surface.z_cos[np.newaxis, :, :]
            * surface.n_field_periods
            * surface.toroidal_modes
            * sin_angle,
            axis=(-1, -2),
        )
    return dz_dphi


def _evaluate_dr_dtheta(
    surface: SurfaceRZFourier, theta_phi: jt.Float[np.ndarray, "*dims 2"]
) -> jt.Float[np.ndarray, "*dims"]:
    angle = _compute_angle(surface, theta_phi)
    sin_angle = np.sin(angle)
    # r_cos is (n_poloidal_modes, n_toroidal_modes)
    dr_dtheta = np.sum(
        surface.r_cos[np.newaxis, :, :] * surface.poloidal_modes * (-1) * sin_angle,
        axis=(-1, -2),
    )
    if not surface.is_stellarator_symmetric:
        assert surface.r_sin is not None
        cos_angle = np.cos(angle)
        dr_dtheta += np.sum(
            surface.r_sin[np.newaxis, :, :] * surface.poloidal_modes * cos_angle,
            axis=(-1, -2),
        )
    return dr_dtheta


def _evaluate_dz_dtheta(
    surface: SurfaceRZFourier, theta_phi: jt.Float[np.ndarray, "*dims 2"]
) -> jt.Float[np.ndarray, "*dims"]:
    angle = _compute_angle(surface, theta_phi)
    cos_angle = np.cos(angle)
    # z_sin is (n_poloidal_modes, n_toroidal_modes)
    dz_dtheta = np.sum(
        surface.z_sin[np.newaxis, :, :] * surface.poloidal_modes * cos_angle,
        axis=(-1, -2),
    )
    if not surface.is_stellarator_symmetric:
        assert surface.z_cos is not None
        sin_angle = np.sin(angle)
        dz_dtheta += np.sum(
            surface.z_cos[np.newaxis, :, :] * surface.poloidal_modes * (-1) * sin_angle,
            axis=(-1, -2),
        )
    return dz_dtheta


def build_surface_rz_fourier_mask(
    surface: SurfaceRZFourier,
    max_poloidal_mode: int,
    max_toroidal_mode: int,
) -> SurfaceRZFourier:
    fourier_coefficients_mask = jnp.asarray(
        (surface.poloidal_modes > 0)
        | ((surface.poloidal_modes == 0) & (surface.toroidal_modes >= 1))
    )
    fourier_coefficients_mask &= (surface.poloidal_modes <= max_poloidal_mode) & (
        np.abs(surface.toroidal_modes) <= max_toroidal_mode
    )
    return surface.model_copy(
        update=dict(
            r_cos=fourier_coefficients_mask,
            z_sin=fourier_coefficients_mask,
        ),
    )


def spectral_width(
    arrays: Sequence[jt.Float[NpOrJaxArray, "n_poloidal_modes n_toroidal_modes"]],
    p: int,
    q: int,
    normalize: bool = True,
) -> jt.Float[NpOrJaxArray, ""]:
    r"""Computes the spectral width of a sequence of Fourier coefficients.

    The spectral width is defined as:

    .. math::
        \frac{\sum_{m,n} m^{p+q} (x_{mn}^2 + ...)}{\sum_{m,n} m^p (x_{mn}^2 + ...)}

    Args:
        arrays: the sequence of Fourier coefficients.
        p, q: see expression above.
        normalize: whether to normalize the spectral width by the sum of the
            coefficients. See expression above. Defaults to True.
    """
    for array in arrays:
        assert array.shape == arrays[0].shape
    max_m = arrays[0].shape[0] - 1

    xm = jnp.arange(max_m + 1)[:, None]
    coefficients = jnp.column_stack(arrays)
    denominator = xm**p * coefficients**2
    numerator = xm**q * denominator

    if not normalize:
        return jnp.sum(numerator)

    denominator_sum = jnp.sum(denominator)
    condition = denominator_sum == 0.0
    return jnp.where(condition, 1.0, jnp.sum(numerator) / denominator_sum)
