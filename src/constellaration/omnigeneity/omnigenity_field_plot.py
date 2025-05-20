import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from matplotlib import axes as mlp_axes
from matplotlib import figure as mpl_figure

from constellaration.omnigeneity import omnigenity_field


def plot_modb_well(
    field: omnigenity_field.OmnigenousField,
    rho: jnp.ndarray = jnp.asarray([1.0]),
    n_eta: int = 100,
) -> mpl_figure.Figure:
    """Plots the B well strenght as a function of eta for different values of rho.

    Args:
        field: an omnigenous field
        rho: A 1D array of radial coordinates to evaluate the magnetic well
        n_eta: The number of points in a linearly spaced grid for the $\\eta$
            coordinate

    Returns:

        fig: A matplotlib figure containing the plot of the magnetic well strength
    """

    eta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_eta, endpoint=False)

    wells = omnigenity_field._compute_magnetic_well_at_rho_eta(field, rho=rho, eta=eta)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for rho_ind, rho in enumerate(rho):
        ax.plot(eta, wells[rho_ind, ...], label=r"$\rho = {:.2f}$".format(rho))
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\mathrm{|B|}$")
    ax.legend()

    plt.close()

    return fig


def plot_boozer_field(
    field: omnigenity_field.OmnigenousField,
    rho: float = 1.0,
    iota: float = 0.0,
    n_eta: int = 100,
    n_alpha: int = 100,
    levels: int = 30,
    n_fieldlines: int | None = None,
    ax: mlp_axes.Axes | None = None,
    **kwargs,
) -> mpl_figure.Figure:
    """Plots the magnetic field stregnth iso-contour lines in Boozer coordinates of the
    omnigenous field.

    Args:
        field: an omnigenous field
        rho: A radial coordinate to evaluate the magnetic well
        iota: The rotational transform of the omnigenous field
        n_eta: The number of points in a linearly spaced grid for the $\\eta$ coordinate
        n_alpha: The number of point in a linearly spaced grid for the $\alpha$
            coordiante
        levels: The number of levels to plot in the contour plot
        n_fieldlines: The number of field lines to plot. If None, no field lines are
            plotted.
        ax: The axis to plot on. If None, a new figure is created.

    Returns:
        fig: A matplotlib figure containing the plot of the magnetic well strength
            iso-contour lines in Boozer coordinates.
    """
    eta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_eta, endpoint=False)

    theta_b, phi_b = omnigenity_field.get_theta_and_phi_boozer(
        field, rho=rho, iota=iota
    )
    modB = (
        omnigenity_field._compute_magnetic_well_at_rho_eta(
            field, rho=jnp.asarray([rho]), eta=eta
        )[0, None, :]
        .repeat(n_alpha, axis=0)
        .flatten()
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.tricontour(phi_b, theta_b, modB, levels=levels, **kwargs)

    if n_fieldlines is not None:
        theta0 = np.linspace(0, 2 * np.pi, n_fieldlines, endpoint=False)
        phi = np.linspace(0, 2 * np.pi / field.n_field_periods, 100)
        alpha = np.atleast_2d(theta0) + iota * np.atleast_2d(phi).T
        alpha1 = np.where(np.logical_and(alpha >= 0, alpha <= 2 * np.pi), alpha, np.nan)
        alpha2 = np.where(
            np.logical_or(alpha < 0, alpha > 2 * np.pi),
            alpha % (np.sign(iota) * 2 * np.pi) + (np.sign(iota) < 0) * (2 * np.pi),
            np.nan,
        )
        alphas = np.hstack((alpha1, alpha2))
        ax.plot(phi, alphas, color="k", ls="-", lw=1)

    ax.set_xlabel(r"$\phi_{Boozer}$")
    ax.set_ylabel(r"$\theta_{Boozer}$")

    fig = ax.get_figure()

    assert isinstance(fig, mpl_figure.Figure)
    fig.colorbar(ax.collections[0], ax=ax)

    return fig
