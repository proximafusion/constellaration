import pathlib

import booz_xform
import matplotlib as mpl
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes
from plotly import graph_objects as go
from constellaration.boozer import boozer as boozer_module
from constellaration.geometry import surface_rz_fourier, surface_utils
from constellaration.mhd import vmec_utils
from scipy import interpolate
from simsopt import mhd


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

    zmns = interpolate.interp1d(s_full_grid, equilibrium.zmns, kind="linear", axis=0)(
        surfaces
    )[..., None, None]
    rmnc = interpolate.interp1d(s_full_grid, equilibrium.rmnc, kind="linear", axis=0)(
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
    return fig
