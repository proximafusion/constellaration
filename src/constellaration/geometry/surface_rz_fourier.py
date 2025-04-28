import jaxtyping as jt
import numpy as np
import pydantic
from typing_extensions import Self

FourierCoefficients = jt.Float[np.ndarray, "n_poloidal_modes n_toroidal_modes"]
FourierModes = jt.Int[np.ndarray, "n_poloidal_modes n_toroidal_modes"]


class SurfaceRZFourier(pydantic.BaseModel, arbitrary_types_allowed=True):
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
