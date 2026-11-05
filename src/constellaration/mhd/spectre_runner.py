"""Solve one SPECTRE input file.

    python -m constellaration.mhd.spectre_runner x.toml

Computes the Beltrami field of the input and writes ``x.h5`` next to it, with the
Beltrami errors in its ``errors`` group. ``run_spectre`` runs this in a child process,
one per rung of its toroidal ladder.
"""

import sys
from typing import Any

import h5py
import spectre


def main(input_file: str) -> None:
    solver: Any = spectre.SPECTRE(input_file, verbose=False)
    solver.run(save_output=True)
    beltrami_avg, beltrami_max = spectre.get_beltrami_errors(solver, print_res=False)
    filename = solver.allglobal_mod.filename_h5
    if isinstance(filename, bytes):
        filename = filename.decode()
    with h5py.File(filename.strip(), "a") as h5:
        errors = h5.create_group("errors")
        errors.create_dataset("beltrami_avg", data=beltrami_avg)
        errors.create_dataset("beltrami_max", data=beltrami_max)


if __name__ == "__main__":
    main(sys.argv[1])
