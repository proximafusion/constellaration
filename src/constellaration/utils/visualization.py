import pathlib

import booz_xform
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
from constellaration.boozer import boozer as boozer_module
from constellaration.mhd import vmec as vmec_module
from simsopt import mhd


def plot_boozer_surfaces(
    equilibrium: vmec_module.VmecppWOut,
    settings: boozer_module.BoozerSettings | None = None,
    save_dir_path: pathlib.Path | None = None,
) -> list[mpl_figure.Figure]:
    """Creates Boozer surface plots."""
    if settings is None:
        settings = boozer_module.BoozerSettings()
    vmec = vmec_module.as_simsopt_vmec(equilibrium)
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
