import numpy as np
from desc import geometry as desc_geometry
from desc import vmec_utils as desc_vmec_utils

from constellaration.geometry import surface_rz_fourier


def from_desc_fourier_rz_toroidal_surface(
    surface: desc_geometry.FourierRZToroidalSurface,
) -> surface_rz_fourier.SurfaceRZFourier:
    r"""Converts a DESC FourierRZToroidalSurface (parameterized with coefficients of a
    Double Fourier Series basis [1]) into a SurfaceRZFourier (Parameterized with Fourier
    coefficients in cylindrical coordinates).

    Specifically, this function takes the **double-Fourier** representation
    from DESC:

    .. math::

       R( \theta, \phi) \;=\;\sum_k R_k \,\cos\bigl(\,m_k\,\theta
         \;-\; n_k\,(\mathrm{NFP})\,\phi\bigr),
       \\
       Z(\theta, \phi) \;=\;\sum_k Z_k \,\sin\bigl(\,m_k\,\theta
         \;-\; n_k\,(\mathrm{NFP})\,\phi\bigr),

    where :math:`m_k` and :math:`n_k` are the poloidal/toroidal integers stored in
    the DESC surface bases, and produces *pure* cylindrical expansions:

    .. math::
       R(\theta, \phi)
       \;=\;
       \sum_{m,n}\Bigl(\,
         R_{m,n}^{(\cos)}\,\cos(m\,\theta - n\,\phi)
         \;+\;
         R_{m,n}^{(\sin)}\,\sin(m\,\theta - n\,\phi)\Bigr),
       \\
       Z(\theta, \phi)
       \;=\;
       \sum_{m,n}\Bigl(\,
         Z_{m,n}^{(\cos)}\,\cos(m\,\theta - n\,\phi)
         \;+\;
         Z_{m,n}^{(\sin)}\,\sin(m\,\theta - n\,\phi)\Bigr).

    We use the ``ptolemy_identity_rev`` helper to reorganize
    :math:`\cos(m\,\theta - n\,(\mathrm{NFP})\,\phi) \leftrightarrow \cos(m\,\theta)\,\cos(n\,\phi) \pm \dots`
    into single-angle “cos/sin” expansions. For a surface with **stellarator symmetry**,
    the typical result is that :math:`R(\theta,\phi)` only has nonzero “cos” modes,
    and :math:`Z(\theta,\phi)` has only “sin” modes (but any small asymmetry
    yields nonzero extra terms).

    Args:
        surface: A DESC FourierRZToroidalSurface object.

    Returns

        A SurfaceRZFourier object with the Fourier coefficients in cylindrical
            coordinates.

    References
    .. [1] `DESC Double-Fourier Series Documentation
       <https://desc-docs.readthedocs.io/en/stable/notebooks/basis_grid.html#Double-Fourier-Series>`_
    """  # noqa: E501
    n_field_periods = surface.NFP

    # ------------------ R expansions ------------------
    # "DESC" style has poloidal modes: R_basis.modes[:, 1] => m
    # toroidal modes: R_basis.modes[:, 2] => n
    # but the actual function is cos(mθ - n*(NFP)*φ).
    poloidal_modes_r = surface.R_basis.modes[:, 1]
    toroidal_modes_r = surface.R_basis.modes[:, 2]

    # R_lmn are the double-Fourier coefficients in DESC
    # shape => (num_modes,). ptolemy_identity_rev expects shape (n_surfs, num_modes)
    # but here we have only "one" surface => put them in a row
    r_array = np.expand_dims(surface.R_lmn, axis=0)

    # Convert to "cos(mθ - n(NFP)φ), sin(mθ - n(NFP)φ)" expansions
    m_out_r, n_out_r, sin_r, cos_r = desc_vmec_utils.ptolemy_identity_rev(
        m_1=poloidal_modes_r, n_1=toroidal_modes_r, x=r_array
    )

    # cos_R, sin_R have shape => (1, num_modes)
    rmnc = cos_r[0, :]  # cos expansions => "R_{m,n} cos(mθ - n(NFP)φ)"
    rmns = sin_r[0, :]  # sin expansions => "R_{m,n} sin(mθ - n(NFP)φ)"

    # ------------------ Z expansions ------------------
    poloidal_modes_z = surface.Z_basis.modes[:, 1]
    toroidal_modes_z = surface.Z_basis.modes[:, 2]

    z_array = np.expand_dims(surface.Z_lmn, axis=0)
    m_out_z, n_out_z, sin_z, cos_z = desc_vmec_utils.ptolemy_identity_rev(
        m_1=poloidal_modes_z, n_1=toroidal_modes_z, x=z_array
    )

    zmnc = cos_z[0, :]
    zmns = sin_z[0, :]

    is_stellarator_symmetric = surface.sym

    named_fourier_modes_cylindrical_coordinates = {}

    for m, n, value in zip(m_out_r, n_out_r, rmnc):
        named_fourier_modes_cylindrical_coordinates[f"r_cos({m}, {n})"] = value

    for m, n, value in zip(m_out_z, n_out_z, zmns):
        named_fourier_modes_cylindrical_coordinates[f"z_sin({m}, {n})"] = value

    if not is_stellarator_symmetric:
        for m, n, value in zip(m_out_r, n_out_r, rmns):
            named_fourier_modes_cylindrical_coordinates[f"r_sin({m}, {n})"] = value

        for m, n, value in zip(m_out_z, n_out_z, zmnc):
            named_fourier_modes_cylindrical_coordinates[f"z_cos({m}, {n})"] = value

    return surface_rz_fourier.boundary_from_named_modes(
        named_fourier_modes_cylindrical_coordinates,
        is_stellarator_symmetric=is_stellarator_symmetric,
        n_field_periods=n_field_periods,
    )


def to_desc_fourier_rz_toroidal_surface(
    surface: surface_rz_fourier.SurfaceRZFourier,
) -> desc_geometry.FourierRZToroidalSurface:
    r"""Converts a SurfaceRZFourier (parameterized with Fourier coefficients in
    cylindrical coordinates) into a DESC FourierRZToroidalSurface (parameterized with
    coefficients of a Double Fourier Series basis [1]).

    Specifically, this function takes the *pure* cylindrical expansions:

    .. math::
       R(\theta, \phi)
       \;=\;
       \sum_{m,n}\Bigl(\,
         R_{m,n}^{(\cos)}\,\cos(m\,\theta - n\,\phi)
         \;+\;
         R_{m,n}^{(\sin)}\,\sin(m\,\theta - n\,\phi)\Bigr),
       \\
       Z(\theta, \phi)
       \;=\;
       \sum_{m,n}\Bigl(\,
         Z_{m,n}^{(\cos)}\,\cos(m\,\theta - n\,\phi)
         \;+\;
         Z_{m,n}^{(\sin)}\,\sin(m\,\theta - n\,\phi)\Bigr).

    and produces the **double-Fourier** representation from DESC:

    .. math::

       R( \theta, \phi) \;=\;\sum_k R_k \,\cos\bigl(\,m_k\,\theta
         \;-\; n_k\,(\mathrm{NFP})\,\phi\bigr),
       \\
       Z(\theta, \phi) \;=\;\sum_k Z_k \,\sin\
            \bigl(\,m_k\,\theta \;-\; n_k\,(\mathrm{NFP})\,\phi\bigr),

    where :math:`m_k` and :math:`n_k` are the poloidal/toroidal integers stored in
    the DESC surface bases.


    Args:
        surface: A SurfaceRZFourier object.

    Returns

        A DESC FourierRZToroidalSurface object.


    References

    .. [1] `DESC Double-Fourier Series Documentation
       <https://desc-docs.readthedocs.io/en/stable/notebooks/basis_grid.html#Double-Fourier-Series>`_
    """  # noqa: E501

    poloidal_modes = surface.poloidal_modes.ravel().astype(int)
    toroidal_modes = surface.toroidal_modes.ravel().astype(int)

    rmnc = surface.r_cos.ravel()
    zmns = surface.z_sin.ravel()
    rmns = surface.r_sin.ravel() if surface.r_sin is not None else np.zeros_like(rmnc)
    zmnc = surface.z_cos.ravel() if surface.z_cos is not None else np.zeros_like(zmns)

    is_stellarator_symmetric = surface.is_stellarator_symmetric
    n_field_periods = surface.n_field_periods

    if is_stellarator_symmetric:
        inds = np.where(np.logical_and(poloidal_modes == 0, toroidal_modes < 0))[0]
        poloidal_modes = np.delete(poloidal_modes, inds)
        toroidal_modes = np.delete(toroidal_modes, inds)
        rmnc = np.delete(rmnc, inds)
        zmns = np.delete(zmns, inds)
        rmns = np.delete(rmns, inds)
        zmnc = np.delete(zmnc, inds)

    # R
    m, n, r_lmn = desc_vmec_utils.ptolemy_identity_fwd(
        poloidal_modes, toroidal_modes, s=rmns, c=rmnc
    )

    # Z
    m, n, z_lmn = desc_vmec_utils.ptolemy_identity_fwd(
        poloidal_modes, toroidal_modes, s=zmns, c=zmnc
    )

    surface_parameters = np.vstack((np.zeros_like(m), m, n, r_lmn, z_lmn)).T

    return desc_geometry.FourierRZToroidalSurface(
        R_lmn=surface_parameters[:, 3],
        Z_lmn=surface_parameters[:, 4],
        modes_R=surface_parameters[:, 1:3].astype(int),
        modes_Z=surface_parameters[:, 1:3].astype(int),
        NFP=n_field_periods,
        sym=is_stellarator_symmetric,  # type: ignore
        check_orientation=False,
    )  # type: ignore


def from_qp_model(
    aspect_ratio: float,
    elongation: float,
    mirror_ratio: float,
    torsion: float,
    n_field_periods: int = 1,
    major_radius: float = 1.0,
    is_stellarator_symmetric: bool = True,
    is_iota_positive: bool = True,
) -> surface_rz_fourier.SurfaceRZFourier:
    """Create a `SurfaceRZFourier` from section III (Goodman et al 2023) for
    quasi-poloidal symmetry.

    This function wraps the `from_qp_model` method of a `FourierRZToroidalSurface`.

    Args:
        aspect_ratio: Aspect ratio of the geometry, i.e., the major radius over the
            average cross-sectional area.
        elongation: Elongation of the elliptical surface, i.e., the major axis over
            minor axis.
        mirror_ratio: Mirror ratio generated by toroidal variation of the
            cross-sectional area. Must be <= 1.
        torsion: Vertical extent of the magnetic axis Z coordinate.  Coefficient of
            sin(2*phi).
        n_field_periods: Number of field periods.
        major_radius: Average major radius. Constant term in the R coordinate.
                is_stellarator_symmetric: Whether to enforce stellarator symmetry.
        is_iota_positive: Whether the rotational transform should be positive or
        negative.
    """
    desc_surface = desc_geometry.FourierRZToroidalSurface.from_qp_model(
        major_radius=major_radius,  # type: ignore
        aspect_ratio=aspect_ratio,  # type: ignore
        elongation=elongation,  # type: ignore
        mirror_ratio=mirror_ratio,
        torsion=torsion,  # type: ignore
        NFP=n_field_periods,
        sym=is_stellarator_symmetric,
        positive_iota=is_iota_positive,
    )
    return from_desc_fourier_rz_toroidal_surface(desc_surface)
