import pathlib

from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import ideal_mhd_parameters, vmec_settings, vmec_utils


def to_vmec2000_wout_file(
    equilibrium: vmec_utils.VmecppWOut, output_file: pathlib.Path
) -> None:
    """Writes a VMEC equilibrium to a VMEC2000 wout file."""
    extras = None
    if equilibrium.model_extra is not None:
        extras = set(equilibrium.model_extra.keys())
    vmec_utils.VmecppWOut.model_validate(equilibrium.model_dump(exclude=extras)).save(
        output_file,
    )


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
