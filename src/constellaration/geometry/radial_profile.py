import jaxtyping as jt
import numpy as np
import pydantic
from scipy import interpolate
from typing_extensions import Self


class InterpolatedRadialProfile(pydantic.BaseModel, arbitrary_types_allowed=True):
    """A radial profile whose values are defined by an interpolation on a given
    normalized effective radius grid."""

    # Note: Avoid making these binary, they are usually small arrays and should be
    # parseable outside of Python, for example by Stardash.
    rho: jt.Float[np.ndarray, " n_points"]
    values: jt.Float[np.ndarray, " n_points"]

    @pydantic.field_validator("rho", "values")
    @classmethod
    def _check_at_least_two_points(
        cls, x: jt.Float[np.ndarray, " n_points"]
    ) -> jt.Float[np.ndarray, " n_points"]:
        if len(x) < 2:
            raise ValueError("At least two points needed for the interpolation.")
        return x

    @pydantic.model_validator(mode="after")
    def _check_rho_greater_than_zero(self) -> Self:
        if np.any(self.rho < 0):
            raise ValueError("rho must be non-negative.")
        return self

    @pydantic.model_validator(mode="after")
    def _check_value_consistency(self) -> Self:
        if len(self.rho) != len(self.values):
            raise ValueError("Provide a grid and values of the same length.")
        return self

    def __add__(
        self, other: "InterpolatedRadialProfile"
    ) -> "InterpolatedRadialProfile":
        """Add two interpolated radial profiles."""
        if np.array_equal(self.rho, other.rho):
            values = self.values + other.values
            return InterpolatedRadialProfile(rho=self.rho, values=values)
        common_rho, values_self, values_other = _get_profiles_onto_common_rho_grid(
            self, other
        )
        values = np.asarray(values_self) + np.asarray(values_other)
        return InterpolatedRadialProfile(rho=common_rho, values=values)

    def __mul__(
        self, other: "InterpolatedRadialProfile | float | int"
    ) -> "InterpolatedRadialProfile":
        if isinstance(other, type(self)):
            if np.array_equal(self.rho, other.rho):
                values = self.values * other.values
                return type(self)(rho=self.rho, values=values)
            common_rho, values_self, values_other = _get_profiles_onto_common_rho_grid(
                self, other
            )
            values = np.asarray(values_self) * np.asarray(values_other)
            return type(self)(rho=common_rho, values=values)
        elif isinstance(other, (int, float)):
            values = self.values * other
            return type(self)(rho=self.rho, values=values)
        else:
            return NotImplemented

    def __rmul__(self, other: float | int) -> "InterpolatedRadialProfile":
        return self.__mul__(other)

    def __truediv__(
        self, other: "InterpolatedRadialProfile | float | int"
    ) -> "InterpolatedRadialProfile":
        if isinstance(other, type(self)):
            if np.array_equal(self.rho, other.rho):
                values = self.values / other.values
                return type(self)(rho=self.rho, values=values)
            common_rho, values_self, values_other = _get_profiles_onto_common_rho_grid(
                self, other
            )
            values = np.asarray(values_self) / np.asarray(values_other)
            return type(self)(rho=common_rho, values=values)
        elif isinstance(other, (int, float)):
            values = self.values / other
            return type(self)(rho=self.rho, values=values)
        else:
            return NotImplemented

    def __rtruediv__(self, other: float | int) -> "InterpolatedRadialProfile":
        if isinstance(other, (int, float)):
            values = other / self.values
            return type(self)(rho=self.rho, values=values)
        else:
            return NotImplemented


def evaluate_at_normalized_effective_radius(
    profile: InterpolatedRadialProfile,
    normalized_effective_radius: jt.Float[np.ndarray, " n_points"],
) -> jt.Float[np.ndarray, " n_points"]:
    interpolator = _get_interpolator(profile)
    return np.array(interpolator(normalized_effective_radius))


def _get_interpolator(
    profile: InterpolatedRadialProfile,
) -> interpolate.InterpolatedUnivariateSpline:
    # Make sure that the first derivative is zero at the magnetic axis by mirroring the
    # data around rho==0.
    if profile.rho[0] == 0.0:
        # do not mirror 0.: we don't want two zeroes in the middle of the mirrored array
        mirror_begin_idx = 1
    else:
        mirror_begin_idx = 0

    full_rho = np.concatenate(
        [
            -1.0 * profile.rho[mirror_begin_idx:][::-1],
            profile.rho,
        ]
    )
    full_values = np.concatenate(
        [
            profile.values[mirror_begin_idx:][::-1],
            profile.values,
        ]
    )

    return interpolate.InterpolatedUnivariateSpline(
        x=full_rho,
        y=full_values,
    )


def _get_profiles_onto_common_rho_grid(
    profile: InterpolatedRadialProfile,
    other_profile: InterpolatedRadialProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_rho = np.union1d(profile.rho, other_profile.rho)
    spline_order_self = min(3, len(profile.rho) - 1)
    # ext=1 means values outside the range of rho are zero!
    values = np.asarray(
        interpolate.InterpolatedUnivariateSpline(
            x=profile.rho, y=profile.values, ext=1, k=spline_order_self
        )(common_rho)
    )
    spline_order_other = min(3, len(other_profile.rho) - 1)
    values_other = np.asarray(
        interpolate.InterpolatedUnivariateSpline(
            x=other_profile.rho, y=other_profile.values, ext=1, k=spline_order_other
        )(common_rho)
    )
    return common_rho, values, values_other
