import pathlib

import vmecpp
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import (
    ideal_mhd_parameters,
    vmec_settings,
    vmec_utils,
)


def to_vmec2000_wout_file(
    equilibrium: vmec_utils.VmecppWOut, output_file: pathlib.Path
) -> None:
    """Writes a VMEC equilibrium to a VMEC2000 wout file."""

    # If these quantities are padded to 100, remove the padding
    wout = vmecpp.VmecWOut._from_cpp_wout(
        equilibrium.model_copy(
            update={
                "fsqt": equilibrium.fsqt[: equilibrium.itfsq],
                "wdot": equilibrium.wdot[: equilibrium.itfsq],
            }
        )
    )
    return wout.save(output_file)


def to_vmec2000_input_file(
    boundary: surface_rz_fourier.SurfaceRZFourier, vmec2000_input_file: pathlib.Path
) -> None:
    """Writes a VMEC2000 input file from a boundary."""
    mhd_parameters = ideal_mhd_parameters.boundary_to_ideal_mhd_parameters(boundary)
    settings = vmec_settings.vmec_settings_high_fidelity_fixed_boundary(boundary)
    vmecpp_indata = vmec_utils.build_vmecpp_indata(mhd_parameters, boundary, settings)
    indata_contents = vmec_utils.vmecpp._util.vmecpp_json_to_indata(
        vmecpp_indata.model_dump(exclude_none=True)
    )
    vmec2000_input_file.write_text(indata_contents)
