import jaxtyping as jt
import numpy as np
from constellaration.geometry import surface_rz_fourier, surface_utils
from constellaration.mhd import vmec_utils
from scipy import optimize, special


def max_elongation(
    equilibrium: vmec_utils.VmecppWOut,
    n_poloidal_points: int,
    n_toroidal_points: int,
) -> float:
    """
    Compute the maximum cross-sectional elongation of the outermost flux surface

    1) Computing the magnetic axis (R_axis, Z_axis) and its tangent at each toroidal
        angle.
    2) For each toroidal angle, finding the poloidal cross-section of the plane
        perpendicular to the axis tangent.
    3) Fitting an ellipse (matching perimeter & area) to that cross-section
        to determine its elongation.
    4) Returning the maximum elongation over all toroidal angles.

    Args:
        equilibrium: A VMEC equilibrium.
        n_poloidal_points: Number of points to sample in the poloidal direction.
        n_toroidal_points: Number of points to sample in the toroidal direction.

    Note: This function is a refactored version for improved readability of
    https://github.com/rogeriojorge/single_stage_optimization/blob/8673154/src/qi_functions.py#L389
    """

    # ------------------------------------------------------------------------
    # 1) Basic checks and load data
    # ------------------------------------------------------------------------
    if equilibrium.lasym:
        raise NotImplementedError("Non-stellarator symmetric equilibria not supported.")

    n_field_periods = equilibrium.nfp

    xm = equilibrium.xm
    xn = equilibrium.xn

    rmnc = equilibrium.rmnc
    zmns = equilibrium.zmns

    # Helper: Return (x,y,z) on the outermost flux surface for angles (theta, phi).
    def boundary_point(theta: float, phi: float) -> np.ndarray:
        rb = np.sum(rmnc[-1, :] * np.cos(xm * theta + xn * phi))
        zb = np.sum(zmns[-1, :] * np.sin(xm * theta + xn * phi))
        xb = rb * np.cos(phi)
        yb = rb * np.sin(phi)
        return np.array([xb, yb, zb])

    raxis_cc = equilibrium.raxis_c
    zaxis_cs = equilibrium.zaxis_s

    # ------------------------------------------------------------------------
    # 2) Compute axis in cylindrical (R, Z) and then in Cartesian (X, Y)
    # ------------------------------------------------------------------------
    theta1d = np.linspace(0, 2 * np.pi, n_poloidal_points)
    phi1d = np.linspace(0, 2 * np.pi / n_field_periods, n_toroidal_points)

    # Slicing xm, xn to efficiently compute the axis geometry
    xn_for_axis = xn[: len(raxis_cc), np.newaxis]

    toroidal_angle = xn_for_axis * phi1d[np.newaxis, :]
    cos_tor = np.cos(toroidal_angle)
    sin_tor = np.sin(toroidal_angle)

    # R, Z of the axis at each phi
    R_axis = np.sum(raxis_cc[:, np.newaxis] * cos_tor, axis=0)
    Z_axis = np.sum(zaxis_cs[:, np.newaxis] * sin_tor, axis=0)

    # Derivatives wrt phi
    dR_axis_dphi = np.sum(-raxis_cc[:, np.newaxis] * xn_for_axis * sin_tor, axis=0)
    dZ_axis_dphi = np.sum(zaxis_cs[:, np.newaxis] * xn_for_axis * cos_tor, axis=0)

    # Convert to Cartesian
    X_axis = R_axis * np.cos(phi1d)
    Y_axis = R_axis * np.sin(phi1d)

    # ------------------------------------------------------------------------
    # 3) Compute unit tangent vectors of the axis
    # ------------------------------------------------------------------------
    d_l_d_phi = np.sqrt(R_axis**2 + dR_axis_dphi**2 + dZ_axis_dphi**2)
    d_r_d_phi_cyl = np.column_stack((dR_axis_dphi, R_axis, dZ_axis_dphi))

    # Tangent in cylindrical
    tangent_cyl = d_r_d_phi_cyl / d_l_d_phi[:, None]
    tR, tPhi, tZ = tangent_cyl[:, 0], tangent_cyl[:, 1], tangent_cyl[:, 2]

    # Convert cylindrical tangents (tR, tPhi, tZ) -> Cartesian (t_x, t_y, t_z)
    tangent_x = tR * np.cos(phi1d) - tPhi * np.sin(phi1d)
    tangent_y = tR * np.sin(phi1d) + tPhi * np.cos(phi1d)
    tangent_z = tZ

    # ------------------------------------------------------------------------
    # 4) For each toroidal angle, find poloidal cross-sections & ellipse elongation
    # ------------------------------------------------------------------------
    elongations = np.zeros(n_toroidal_points)

    # Pre-allocate memory for cross-section coordinates
    cross_section_x = np.zeros(n_poloidal_points)
    cross_section_y = np.zeros(n_poloidal_points)
    cross_section_z = np.zeros(n_poloidal_points)

    # Initial guess for ellipse fitting
    a_guess = 1.0

    for i_phi in range(n_toroidal_points):
        # Axis position & tangent vector at this toroidal slice
        axis_xyz = np.array([X_axis[i_phi], Y_axis[i_phi], Z_axis[i_phi]])
        axis_tan = np.array([tangent_x[i_phi], tangent_y[i_phi], tangent_z[i_phi]])

        # -- (a) Find cross-section points perpendicular to axis_tan
        for i_theta, theta_ in enumerate(theta1d):

            def _dot_tangent(phi_guess):
                """Returns dot(axis_tan, (boundary - axis_xyz))."""
                r_bd = boundary_point(theta_, phi_guess[0])
                return np.dot(axis_tan, r_bd - axis_xyz)

            # Solve for phi that yields dot(tangent, boundary - axis) = 0
            phi_init = phi1d[i_phi]
            phi_perp = float(optimize.fsolve(_dot_tangent, x0=[phi_init])[0])

            r_cross = boundary_point(theta_, phi_perp)
            # Optional cleanup: remove any leftover projection along axis_tan
            r_cross -= np.dot(r_cross, axis_tan) * axis_tan

            cross_section_x[i_theta] = r_cross[0]
            cross_section_y[i_theta] = r_cross[1]
            cross_section_z[i_theta] = r_cross[2]

        # -- (b) Compute perimeter & area of cross-section
        dx = cross_section_x - np.roll(cross_section_x, 1)
        dy = cross_section_y - np.roll(cross_section_y, 1)
        dz = cross_section_z - np.roll(cross_section_z, 1)
        perimeter = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

        area = _get_polygon_area_from_vertices(
            cross_section_x, cross_section_y, cross_section_z
        )

        # -- (c) Ellipse fitting to match perimeter and area
        # area = π * a * b
        # perimeter ≈ 4 a E(e) with e^2 = 1 - (b^2 / a^2)
        # Elongation = a / b

        def ellipse_perimeter_residual(a):
            b = area / (np.pi * a)
            e_sqr = 1.0 - (b / a) ** 2
            return perimeter - 4.0 * a * special.ellipe(e_sqr)

        # Use the previous a_guess to speed up convergence
        a_sol = float(optimize.fsolve(ellipse_perimeter_residual, a_guess)[0])
        a_guess = a_sol

        b_sol = area / (np.pi * a_sol)
        semi_maj = max(a_sol, b_sol)
        semi_min = min(a_sol, b_sol)

        elongations[i_phi] = semi_maj / semi_min

    return float(np.max(elongations))


def average_triangularity(
    surface: surface_rz_fourier.SurfaceRZFourier,
    n_poloidal_points: int = 201,
) -> float:
    r"""Compute the average triangularity of the plasma boundary at the two stellarator
    symmetry planes.

    The triangularity is defined as:

    .. math::
        \delta = \frac{\delta_{top} + \delta_{bottom}}{2}
        \delta_{top} = 2 \frac{R0 - R_{Zmax}}{R_{max} - R_{min}}
        \delta_{bottom} = 2 \frac{R0 - R_{Zmin}}{R_{max} - R_{min}}

    where :math:`R_{max}` and :math:`R_{min}` are the maximum and minimum
    R in the toroidal cross-section, :math:`R0` is the location of the magnetic axis,
    :math:`R_{Zmax}` and :math:`R_{Zmin}` are the R coordinate of the location of the
    maximum and minimum Z in the toroidal cross-section.

    The average triangularity is then computed by averaging the triangularity over the
    two stellarator symmetry planes (phi = 0 and phi = pi/nfp).
    """
    if not surface.is_stellarator_symmetric:
        raise NotImplementedError("Non-stellarator symmetric surfaces not supported.")

    theta_phi = surface_utils.make_theta_phi_grid(
        n_theta=n_poloidal_points,
        n_phi=2,
        phi_upper_bound=np.pi / surface.n_field_periods,
        include_endpoints=True,
    )

    RZ = surface_rz_fourier.evaluate_points_rz(
        surface=surface,
        theta_phi=theta_phi,
    )
    R = RZ[..., 0]
    Z = RZ[..., 1]

    # Approximate the magnetic axis with the surface centroid.
    R0 = np.mean(R, axis=0)

    R_max = np.max(R, axis=0)
    R_min = np.min(R, axis=0)

    minor_radius = (R_max - R_min) / 2

    indices_of_max_Z = np.argmax(Z, axis=0)

    # Top and bottom triangularity at the two stellarator symmetry planes are equal.
    triangularity = (R0 - R[indices_of_max_Z, np.arange(2)]) / minor_radius

    return np.mean(triangularity)


def _get_polygon_area_from_vertices(
    X: jt.Float[np.ndarray, " n_points"],
    Y: jt.Float[np.ndarray, " n_points"],
    Z: jt.Float[np.ndarray, " n_points"],
) -> float:
    """Compute the area of a polygon in 3D space whose vertices are given by (X[i],
    Y[i], Z[i]).

    This routine uses a vector cross-product approach to sum the contributions of each
    edge of the polygon, then projects that sum onto the polygon's plane via a unit
    normal. The resulting projected magnitude, divided by 2,
    is returned as the polygon's area.

    Args:
        X, Y, Z: Arrays of length `n_points` containing the x, y, and z coordinates of\
            the polygon's vertices in 3D.  The polygon is assumed to be closed,
            so the point (X[i], Y[i], Z[i]) is connected to (X[i+1], Y[i+1], Z[i+1])
            for i = 0 to N-1, with the last vertex connecting back to the first.

    Returns:
        The area of the 3D polygon.
    """

    # Initialize a 3D vector (accumulator for cross-product sums)
    total = np.array([0.0, 0.0, 0.0])

    # Loop through all edges of the polygon, treating
    # (X[i],Y[i],Z[i]) -> (X[i+1],Y[i+1],Z[i+1]) as consecutive vertices.
    for i in range(len(X)):
        x1, y1, z1 = X[i], Y[i], Z[i]
        x2, y2, z2 = X[(i + 1) % len(X)], Y[(i + 1) % len(Y)], Z[(i + 1) % len(Z)]

        # Create vectors for the current edge's endpoints
        vi1 = [x1, y1, z1]
        vi2 = [x2, y2, z2]

        total += np.cross(vi1, vi2)

    # Pick three vertices from the polygon to define a plane
    pt0 = np.array([X[0], Y[0], Z[0]])
    pt1 = np.array([X[1], Y[1], Z[1]])
    pt2 = np.array([X[2], Y[2], Z[2]])

    # Dot the accumulated cross product sum with the plane's unit normal
    plane_normal = _get_unit_normal_from_plane_defined_by(pt0, pt1, pt2)
    result = np.dot(total, plane_normal)

    # The area is half the absolute value of that projection
    return abs(result / 2)


def _get_unit_normal_from_plane_defined_by(
    a: jt.Float[np.ndarray, " 3"],
    b: jt.Float[np.ndarray, " 3"],
    c: jt.Float[np.ndarray, " 3"],
) -> jt.Float[np.ndarray, " 3"]:
    """Compute the unit normal vector to the plane defined by three points (or vectors)
    a, b, and c.

    Args:
        a, b, c : Three points (or vectors) in 3D space, each having three coordinates
            (x, y, z).

    Returns:
        A 3-element tuple (nx, ny, nz) representing the unit normal vector to the plane.
        The direction of the normal is determined by the sign conventions
        in the determinant.

    Notes:

    This function uses the determinant-based approach to compute each component
    (x, y, z) of the normal vector.  After obtaining the normal vector, it is
    normalized so that its length is 1.  Concretely:

      nx = det([[1,   a[1], a[2]],
                [1,   b[1], b[2]],
                [1,   c[1], c[2]]])

      ny = det([[a[0], 1,   a[2]],
                [b[0], 1,   b[2]],
                [c[0], 1,   c[2]]])

      nz = det([[a[0], a[1], 1],
                [b[0], b[1], 1],
                [c[0], c[1], 1]])

    The result is then divided by its magnitude to produce a unit vector.
    """
    # Compute the x-component of the normal vector
    x = np.linalg.det([[1, a[1], a[2]], [1, b[1], b[2]], [1, c[1], c[2]]])

    # Compute the y-component of the normal vector
    y = np.linalg.det([[a[0], 1, a[2]], [b[0], 1, b[2]], [c[0], 1, c[2]]])

    # Compute the z-component of the normal vector
    z = np.linalg.det([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]])

    # Compute the magnitude of the normal vector
    magnitude = np.sqrt(x**2 + y**2 + z**2)

    # Return the normalized (unit) normal vector
    return np.array([x / magnitude, y / magnitude, z / magnitude])
