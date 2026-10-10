import pathlib

import booz_xform
import matplotlib as mpl
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes
from plotly import graph_objects as go
from scipy import interpolate
from simsopt import mhd

from constellaration.boozer import boozer as boozer_module
from constellaration.geometry import surface_rz_fourier, surface_utils
from constellaration.mhd import spectre, vmec_utils


def plot_surface(
    surface: surface_rz_fourier.SurfaceRZFourier,
    n_theta: int = 50,
    n_phi: int = 51,
    include_endpoints: bool = True,
) -> go.Figure:
    """Plot a continuous surface in 3D space using Plotly.

    Args:
        surface: The surface to plot.
        n_theta: Number of samples in the theta angle.
        n_phi: Number of samples in the phi angle.
        include_endpoints: Whether to include the last point both poloidally and
            toroidally.

    Returns:
        The figure with the surface added.
    """
    fig = go.Figure()

    theta_phi = surface_utils.make_theta_phi_grid(
        n_theta, n_phi, include_endpoints=include_endpoints
    )
    points = surface_rz_fourier.evaluate_points_xyz(surface, theta_phi)

    # Ensure points is a NumPy array with shape (n_phi, n_theta, 3)
    points = np.array(points)
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    fig.add_trace(go.Surface(x=x, y=y, z=z))

    default_layout_kwargs = dict(
        height=600,
        width=600,
        xaxis_title="R",
        yaxis_title="Z",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0, 0, 0, 0)",
    )

    fig.update_layout(
        default_layout_kwargs,
        scene=dict(
            aspectmode="data",  # maintains the true aspect ratio
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
    )

    return fig


def plot_boundary(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    ax: axes.Axes | None = None,
) -> axes.Axes:
    if ax is None:
        _, _ax = plt.subplots()
    else:
        _ax = ax
    theta_phi = surface_utils.make_theta_phi_grid(
        n_theta=64,
        n_phi=5,
        phi_upper_bound=np.pi / boundary.n_field_periods,
        include_endpoints=True,
    )
    rz_points = surface_rz_fourier.evaluate_points_rz(boundary, theta_phi)
    for i in range(theta_phi.shape[1]):
        _ax.plot(
            rz_points[:, i, 0],
            rz_points[:, i, 1],
            label=f"{i}/4" + r"$\frac{\pi}{N_{\text{fp}}}$",
        )
    _ax.set_xlabel("R")
    _ax.set_ylabel("Z")
    _ax.set_aspect("equal")
    _ax.legend()
    return _ax


def plot_boozer_surfaces(
    equilibrium: vmec_utils.VmecppWOut,
    settings: boozer_module.BoozerSettings | None = None,
    save_dir_path: pathlib.Path | None = None,
) -> list[mpl_figure.Figure]:
    """Creates Boozer surface plots."""
    if settings is None:
        settings = boozer_module.BoozerSettings()
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

    figures = []
    for js in range(len(boozer.bx.compute_surfs)):
        plt.figure()
        booz_xform.surfplot(b=boozer.bx, js=js, fill=False)
        fig = plt.gcf()
        figures.append(fig)

    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(figures):
            fig.savefig(save_dir_path / f"surface_plot_{i}.png")

    return figures


def plot_flux_surfaces(
    equilibrium: vmec_utils.VmecppWOut,
    boundary: surface_rz_fourier.SurfaceRZFourier,
    surfaces: list[float] | None = None,
    ntheta: int = 128,
    nphi: int = 4,
    title: str | None = None,
) -> mpl_figure.Figure:
    """Plot the shape of the selected flux surfaces.

    Args:
        equilibrium: the equilibrium object containing the flux surface data.
        boundary: the plasma boundary object.
        surfaces: the flux surface labels to plot.
            If None, a default set of 10 surfaces evenly spaced between 0 and 1 will
            be used. Defaults to None.
        ntheta: the number of poloidal points. Defaults to 128.
        nphi: the number of toroidal points. Defaults to 4.
        title: title for the plot. If None, no title is added. Defaults to None.

    Returns:
        The figure with the flux surfaces plotted.
    """
    fig, ax = plt.subplots()

    if surfaces is None:
        surfaces = list(np.linspace(0, 1.0, 10))

    # Shorthands
    nfp = equilibrium.nfp
    ns = equilibrium.ns
    xm = equilibrium.xm
    xn = equilibrium.xn

    theta = np.linspace(0, 2 * np.pi, num=ntheta)
    if boundary.is_stellarator_symmetric:
        phi = np.linspace(0, np.pi / nfp, num=nphi)
    else:
        phi = np.linspace(0, 2 * np.pi / nfp, num=nphi, endpoint=False)
    phi, theta = np.meshgrid(phi, theta)

    s_full_grid = np.linspace(0, 1, num=ns)
    angle = xm[:, None, None] * theta - xn[:, None, None] * phi

    zmns = interpolate.interp1d(s_full_grid, equilibrium.zmns.T, kind="linear", axis=0)(
        surfaces
    )[..., None, None]
    rmnc = interpolate.interp1d(s_full_grid, equilibrium.rmnc.T, kind="linear", axis=0)(
        surfaces
    )[..., None, None]

    R = np.sum(rmnc * np.cos(angle), axis=1)
    Z = np.sum(zmns * np.sin(angle), axis=1)

    colors = mpl.colormaps["tab10"](np.linspace(0, 1, nphi))

    for i in range(nphi):
        normalized_phi = phi[0, i] / (2 * np.pi / nfp)
        for j in range(len(surfaces)):
            label = (
                r"$\varphi=" + f"{normalized_phi:.2f}" + r"\frac{2\pi}{N_{fp}}$"
                if j == 0
                else None
            )
            ax.plot(R[j, :, i], Z[j, :, i], label=label, c=colors[i])
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, ncol=1)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    if title is not None:
        ax.set_title(title)
    return fig


def _flux_surfaces_rz(
    equilibrium: vmec_utils.VmecppWOut,
    normalized_toroidal_flux: np.ndarray,
    phi: float,
    n_theta: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """R and Z of VMEC flux surfaces on one toroidal plane, as (surface, theta)."""
    theta = np.linspace(0, 2 * np.pi, num=n_theta)
    angle = equilibrium.xm[:, None] * theta[None, :] - equilibrium.xn[:, None] * phi
    s_full_grid = equilibrium.normalized_toroidal_flux_full_grid_mesh
    rmnc = interpolate.interp1d(s_full_grid, equilibrium.rmnc.T, axis=0)(
        normalized_toroidal_flux
    )
    zmns = interpolate.interp1d(s_full_grid, equilibrium.zmns.T, axis=0)(
        normalized_toroidal_flux
    )
    return rmnc @ np.cos(angle), zmns @ np.sin(angle)


def plot_flux_surfaces_cross_section(
    equilibrium: vmec_utils.VmecppWOut,
    normalized_toroidal_angle: float = 0.5,
    n_surfaces: int = 12,
    figsize: tuple[float, float] = (5.0, 6.0),
) -> mpl_figure.Figure:
    """Plot the VMEC flux surfaces on one toroidal plane.

    Args:
        equilibrium: the equilibrium object containing the flux surface data.
        normalized_toroidal_angle: the plane, in units of the field period. 0 and 0.5
            are the two stellarator-symmetric planes.
        n_surfaces: the number of flux surfaces, evenly spaced in toroidal flux.
        figsize: the figure size.

    Returns:
        The figure with the flux surfaces plotted.
    """
    fig, ax = plt.subplots(figsize=figsize)
    phi = normalized_toroidal_angle * 2 * np.pi / equilibrium.nfp
    surfaces = np.linspace(0, 1.0, n_surfaces + 1)
    r, z = _flux_surfaces_rz(equilibrium, surfaces, phi)
    ax.plot(r[1:].T, z[1:].T, c="tab:blue", lw=0.8)
    ax.plot(r[0, 0], z[0, 0], "+", c="tab:blue")
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(
        r"VMEC flux surfaces at $\varphi="
        + f"{normalized_toroidal_angle:g}"
        + r"\,\frac{2\pi}{N_{fp}}$"
    )
    return fig


def plot_rotational_transform(
    equilibrium: vmec_utils.VmecppWOut,
    crossings: list[spectre.ScreenedCrossing] | None = None,
    spectre_profile: tuple[np.ndarray, np.ndarray] | None = None,
    figsize: tuple[float, float] = (7.0, 4.0),
) -> mpl_figure.Figure:
    """Plot the rotational transform profile and the rationals it crosses.

    Args:
        equilibrium: the VMEC equilibrium.
        crossings: crossings to mark, as the SPECTRE screen returns them. Each is
            drawn as a line at its rational n/m and a marker at its crossing.
        spectre_profile: normalised toroidal flux and rotational transform of the
            field lines of a SPECTRE field, to compare with the VMEC profile.
        figsize: the figure size.

    Returns:
        The figure with the rotational transform plotted.
    """
    fig, ax = plt.subplots(figsize=figsize)
    psi_n = equilibrium.normalized_toroidal_flux_full_grid_mesh
    ax.plot(psi_n, np.abs(equilibrium.iotaf), c="tab:blue", label="VMEC")
    if spectre_profile is not None:
        ax.plot(*spectre_profile, ".", c="tab:red", ms=5, label="SPECTRE")
    rationals = {(r.n, r.m) for r in crossings or []}
    colors = mpl.colormaps["tab10"](np.linspace(0, 1, 10))
    for color, (n, m) in zip(colors[2:], sorted(rationals, key=lambda nm: nm[::-1])):
        ax.axhline(n / m, c=color, lw=0.8, ls="--", label=f"{n}/{m}")
        positions = [r.psi_n for r in crossings or [] if (r.n, r.m) == (n, m)]
        ax.plot(positions, [n / m] * len(positions), "o", c=color)
    ax.set_xlabel("Normalized toroidal flux")
    ax.set_ylabel(r"Rotational transform $|\iota|$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    return fig


def plot_poincare_section(
    equilibrium: vmec_utils.VmecppWOut,
    field_lines: list[np.ndarray],
    chain_points: list[tuple[np.ndarray, np.ndarray]] | None = None,
    normalized_toroidal_angle: float = 0.5,
    n_surfaces: int = 12,
    figsize: tuple[float, float] = (6.0, 7.0),
) -> mpl_figure.Figure:
    """Plot VMEC flux surfaces against the Poincare section of a SPECTRE field.

    On a stellarator-symmetric plane the section is up-down symmetric, so the two
    are drawn in one frame: the VMEC flux surfaces in the upper half and the SPECTRE
    punctures in the lower half. The two halves are different kinds of object: the
    VMEC surfaces are nested by assumption, the punctures are traced field lines.

    Args:
        equilibrium: the VMEC equilibrium.
        field_lines: (R, Z) punctures of each field line with the plane, one
            ``(n, 2)`` array per field line.
        chain_points: the O-points and X-points of island chains on that plane, as
            two ``(m, 2)`` arrays of (R, Z) per chain; drawn in the lower half.
        normalized_toroidal_angle: the plane, in units of the field period; 0 or 0.5.
        n_surfaces: the number of VMEC flux surfaces.
        figsize: the figure size.

    Returns:
        The figure with the two half sections plotted.
    """
    fig, ax = plt.subplots(figsize=figsize)
    phi = normalized_toroidal_angle * 2 * np.pi / equilibrium.nfp
    surfaces = np.linspace(0, 1.0, n_surfaces + 1)
    r, z = _flux_surfaces_rz(equilibrium, surfaces, phi)
    upper = np.where(z >= 0, z, np.nan)
    ax.plot(r[1:].T, upper[1:].T, c="tab:blue", lw=0.8)
    ax.plot(r[-1], -upper[-1], c="k", lw=0.8)
    for line in field_lines:
        lower = line[line[:, 1] <= 0]
        ax.plot(lower[:, 0], lower[:, 1], ".", c="k", ms=0.6)
    for i, (o_points, x_points) in enumerate(chain_points or []):
        for points, marker, label in (
            (o_points, "o", "O-points"),
            (x_points, "x", "X-points"),
        ):
            lower = points[points[:, 1] <= 1e-9]
            ax.plot(
                lower[:, 0],
                lower[:, 1],
                marker,
                c="tab:red",
                ms=6,
                ls="none",
                fillstyle="none",
                label=label if i == 0 else None,
            )
    if chain_points:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.axhline(0.0, c="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("VMEC flux surfaces (top), SPECTRE field lines (bottom)")
    return fig
