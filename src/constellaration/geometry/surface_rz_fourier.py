import jaxtyping as jt
import numpy as np
import pydantic
from constellaration.geometry import surface_utils
from simsopt import geo
from typing_extensions import Self
from vmecpp import _pydantic_numpy as pydantic_numpy

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
