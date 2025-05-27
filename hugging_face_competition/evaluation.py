import argparse
import json
from pathlib import Path

from constellaration import problems
from constellaration.geometry import surface_rz_fourier

PROBLEM_TYPES = ["geometrical", "simple_to_build", "mhd_stable"]


def load_boundary(data: str) -> surface_rz_fourier.SurfaceRZFourier:
    return surface_rz_fourier.SurfaceRZFourier.model_validate_json(data)


def load_boubdaries(data: str) -> list[surface_rz_fourier.SurfaceRZFourier]:
    data_json = json.loads(data)
    return [
        surface_rz_fourier.SurfaceRZFourier.model_validate_json(b) for b in data_json
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate constellation problems.")
    parser.add_argument(
        "--problem-type",
        type=str,
        required=True,
        help="Type of the problem",
        choices=PROBLEM_TYPES,
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="Path to input JSON file"
    )
    args = parser.parse_args()
    problem_type = args.problem_type

    with Path(args.input_file).open("r") as f:
        data = f.read()

    match problem_type:
        case "geometrical":
            boundary = load_boundary(data)
            result = problems.GeometricalProblem().evaluate(boundary)
        case "simple_to_build":
            boundary = load_boundary(data)
            result = problems.SimpleToBuildQIStellarator().evaluate(boundary)
        case "mhd_stable":
            boundaries = load_boubdaries(data)
            result = problems.MHDStableQIStellarator().evaluate(boundaries)
        case _:
            raise ValueError(f"Unknown problem type: {problem_type}")
    # TODO: save output to a file?
    print(result)


if __name__ == "__main__":
    main()
