from typing import Literal

import jaxtyping as jt
import numpy as np
import pydantic
import qic
import qic.util
from constellaration.geometry import surface_rz_fourier
from scipy import optimize
from simsopt import geo

NpOrJaxArray = np.ndarray | jt.Array
_MAX_VMEC_N_MODES = 100


class NearAxisConfiguration(pydantic.BaseModel, arbitrary_types_allowed=True):
    n_field_periods: int
    """The number of field periods."""

    r_cos: jt.Float[NpOrJaxArray, " n_fourier_modes"]
    """The Fourier cosine coefficients of the r coordinate of the magnetic axis."""

    z_sin: jt.Float[NpOrJaxArray, " n_fourier_modes"]
    """The Fourier sine coefficients of the z coordinate of the magnetic axis."""

    B0_cos: jt.Float[NpOrJaxArray, " n_fourier_modes"]
    """The magnetic field strength on axis Fourier cosine coefficients."""

    d_over_curvature: float = 0.0
    """The component of d linearly proportional to the curvature of the magnetic
    axis."""

    d_over_curvature_cos: jt.Float[NpOrJaxArray, " n_fourier_modes"] = np.array([])
    """The Fourier cosine coefficients of d_over_curvature."""

    d_sin: jt.Float[NpOrJaxArray, " n_fourier_modes"] = np.array([])
    """The Fourier sine coefficients of d.

    d provides information on the first-order magnetic field strength.
    """

    omnigeneity_method: Literal[
        "non-zone", "buffer", "non-zone-fourier", "non-zone-smoother"
    ] = "non-zone-smoother"

    k_buffer: int = 2

    p_buffer: int = 3


class NearAxisToPlasmaBoundarySettings(pydantic.BaseModel):
    """Settings for the first-order to boundary translation."""

    minor_radius: float = 0.1

    n_toroidal_points: int = 31

    n_poloidal_points: int = 20


def generate(
    mirror_ratio: float,
    min_iota: float,
    max_elongation: float,
    torsion: float,
    n_field_periods: int = 1,
    major_radius: float = 1.0,
    max_toroidal_mode: int = 3,
    verbose: bool = False,
) -> NearAxisConfiguration:
    """Generates a near-axis configuration that matches the given properties.

    The configuration features:

    - Magnetic field strength on axis with a single well and `mirror_ratio` as the
        ratio of the maximum to the minimum magnetic field strength.
    - Extrema of the magnetic field strength for phi=0 and phi=pi/NFP.
    - Zero curvature of the magnetic axis at those extrema.
    - QI magnetic field up to first order.
    - Elongation of the flux surfaces below `max_elongation`.
    - Rotational transform at the magnetic axis equal above `min_iota`.

    The initial guess for the magnetic axis follows section III of:
    Goodman, Alan, et al. "Constructing Precisely Quasi-Isodynamic
    Magnetic Fields." Journal of Plasma Physics 89.5 (2023): 905890504.

    Args:
        mirror_ratio: the ratio of the maximum to the minimum magnetic field strength
            on axis.
        min_iota: the minimum rotational transform at the magnetic axis.
        max_elongation: the maximum elongation of the flux surfaces.
        torsion: Value to use for the (m,n)=(0,1) Fourier modes, which
            controls the torsion of the magnetic axis.
        n_field_periods: the number of field periods.
        major_radius: the major radius of the magnetic axis.
        max_toroidal_mode: the maximum toroidal mode number to generate.
        verbose: whether to print information about the optimization.
    """

    # Obtain a positive iota value by setting rc_{0,1} to a negative value and
    # z_{0,1} to a positive value.
    absolute_torsion = np.abs(torsion)

    r_cos_n_1_initial_guess = -absolute_torsion
    r_cos = np.array(
        [
            major_radius,
            r_cos_n_1_initial_guess,
            -major_radius / (1 + 4 * n_field_periods**2),
            -r_cos_n_1_initial_guess
            * (n_field_periods**2 + 1)
            / (9 * n_field_periods**2 + 1),
        ]
    )
    if max_toroidal_mode > 3:
        r_cos = np.pad(
            r_cos,
            (0, max_toroidal_mode - 3),
            mode="constant",
            constant_values=0.0,
        )

    z_sin_n_1_initial_guess = absolute_torsion
    z_sin_n_2_initial_guess = 0.02
    z_sin_n_3_initial_guess = 0.01
    z_sin = np.array(
        [0.0, z_sin_n_1_initial_guess, z_sin_n_2_initial_guess, z_sin_n_3_initial_guess]
    )
    if max_toroidal_mode > 3:
        z_sin = np.pad(
            z_sin,
            (0, max_toroidal_mode - 3),
            mode="constant",
            constant_values=0.0,
        )

    B0_cos = np.array([1.0, mirror_ratio])

    configuration = NearAxisConfiguration(
        n_field_periods=n_field_periods,
        r_cos=r_cos,
        z_sin=z_sin,
        B0_cos=B0_cos,
        d_over_curvature=1.0,
    )

    toroidal_mode_numbers = np.arange(0, max_toroidal_mode + 1)
    r_cos_mask = np.ones_like(toroidal_mode_numbers, dtype=bool)
    r_cos_mask[[0, 2, 3]] = False
    z_sin_mask = np.ones_like(toroidal_mode_numbers, dtype=bool)
    z_sin_mask[[0]] = False

    def to_x(
        near_axis_configuration: NearAxisConfiguration,
    ) -> np.ndarray:
        return np.concatenate(
            [
                near_axis_configuration.r_cos[r_cos_mask],
                near_axis_configuration.z_sin[z_sin_mask],
                [near_axis_configuration.d_over_curvature],
            ]
        )

    def from_x(x: np.ndarray) -> NearAxisConfiguration:
        return configuration.model_copy(
            update=dict(
                r_cos=np.concatenate(
                    [
                        [major_radius],
                        [x[0]],
                        [-major_radius / (1 + 4 * n_field_periods**2)],
                        [
                            -x[0]
                            * (n_field_periods**2 + 1)
                            / (9 * n_field_periods**2 + 1)
                        ],
                        x[1 : max_toroidal_mode - 2],
                    ]
                ),
                z_sin=np.concatenate(
                    [
                        [0.0],
                        x[max_toroidal_mode - 2 : 2 * max_toroidal_mode - 2],
                    ]
                ),
                d_over_curvature=x[-1],
            ),
        )

    def normalized_barrier(value: float, goal: float, is_max: bool) -> float:
        if is_max:
            barrier = np.maximum(0.0, value - goal)
        else:
            barrier = np.maximum(0.0, goal - value)
        return (barrier / goal) ** 2

    def fun(x: np.ndarray) -> float:
        configuration = from_x(x)
        stellarator = near_axis_configuration_to_pyqic(configuration)
        # Support only first order for now.
        objective = stellarator.min_geo_qi_consistency(order=1)
        if verbose:
            string = "QI Residual: {:.2e}".format(objective)
            string += " Max Elongation: {:.2e}".format(np.max(stellarator.elongation))  # type: ignore
            string += " Iota: {:.2e}".format(np.abs(stellarator.iota))  # type: ignore
            print(string)
        weight = 1.0e2
        objective += weight * normalized_barrier(
            value=np.max(stellarator.elongation),  # type: ignore
            goal=max_elongation,
            is_max=True,
        )
        objective += weight * normalized_barrier(
            value=np.abs(stellarator.iota), goal=min_iota, is_max=False  # type: ignore
        )
        return objective

    x0 = to_x(configuration)

    scale = 0.5 * np.concatenate(
        [
            10.0 ** (-toroidal_mode_numbers[r_cos_mask] / 5.0),
            10.0 ** (-toroidal_mode_numbers[z_sin_mask] / 5.0),
            [1.0],
        ]
    )

    lower_bound = x0 - scale
    upper_bound = x0 + scale

    bounds = list(zip(lower_bound, upper_bound))

    # Set bounds for rc_{0,1} and z_{0,1} such that the iota stays positive.
    bounds[0] = (-absolute_torsion, 0.0)
    bounds[max_toroidal_mode - 2] = (0.0, absolute_torsion)

    fun_at_x0 = fun(x0)

    result = optimize.minimize(
        # Scale the objective to be of order ~1.
        fun=lambda x: fun(x) / fun_at_x0,
        x0=x0,
        method="Nelder-Mead",
        bounds=bounds,
    )
    result = optimize.minimize(
        fun=lambda x: fun(x) / fun_at_x0,
        x0=result.x,
        method="L-BFGS-B",
        bounds=bounds,
    )

    return from_x(result.x)


def near_axis_configuration_to_pyqic(
    near_axis_configuration: NearAxisConfiguration,
) -> qic.Qic:
    return qic.Qic(
        rc=near_axis_configuration.r_cos.tolist(),
        zs=near_axis_configuration.z_sin.tolist(),
        nfp=near_axis_configuration.n_field_periods,
        B0_vals=near_axis_configuration.B0_cos.tolist(),
        omn=True,
        omn_method=near_axis_configuration.omnigeneity_method,
        k_buffer=near_axis_configuration.k_buffer,
        p_buffer=near_axis_configuration.p_buffer,
        order="r1",
        d_over_curvature=near_axis_configuration.d_over_curvature,  # type: ignore
        d_svals=near_axis_configuration.d_sin.tolist(),
        # Set vacuum.
        p2=0.0,
        # Set current to zero.
        I2=0.0,
    )


def near_axis_configuration_to_plasma_boundary(
    near_axis_configuration: NearAxisConfiguration,
    settings: NearAxisToPlasmaBoundarySettings,
) -> surface_rz_fourier.SurfaceRZFourier:
    """Converts a first-order near-axis configuration to a boundary surface."""
    stellarator = near_axis_configuration_to_pyqic(near_axis_configuration)

    R_2D, Z_2D, _ = stellarator.Frenet_to_cylindrical(
        r=settings.minor_radius, ntheta=settings.n_poloidal_points
    )

    r_cos, _, _, z_sin = qic.util.to_Fourier(
        R_2D=R_2D,
        Z_2D=Z_2D,
        nfp=near_axis_configuration.n_field_periods,
        mpol=int(min(settings.n_poloidal_points / 2, _MAX_VMEC_N_MODES)),
        ntor=int(min(settings.n_toroidal_points / 2, _MAX_VMEC_N_MODES)),
        lasym=False,
    )

    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos.T,
        z_sin=z_sin.T,
        n_field_periods=near_axis_configuration.n_field_periods,
        is_stellarator_symmetric=True,
    )


def smooth_and_set_max_mode_numbers(
    surface: surface_rz_fourier.SurfaceRZFourier,
    max_poloidal_mode: int,
    max_toroidal_mode: int,
) -> surface_rz_fourier.SurfaceRZFourier:
    """Smooths the boundary and sets the maximum mode numbers.

    Surfaces obtained via the near-axis expansion may have high-frequency modes.
    Ideally, we would like to perform a spectral condensation to remove these modes.
    However, this is not implemented yet. For now, we smooth the boundary by fitting the
    surface gamma with a lower capacity surface.
    """
    simsopt_boundary = surface_rz_fourier.to_simsopt(surface=surface)
    new_simsopt_boundary = geo.SurfaceRZFourier(
        nfp=simsopt_boundary.nfp,
        stellsym=simsopt_boundary.stellsym,
        mpol=max_poloidal_mode,
        ntor=max_toroidal_mode,
    )
    new_simsopt_boundary.least_squares_fit(simsopt_boundary.gamma())
    return surface_rz_fourier.from_simsopt(new_simsopt_boundary)
