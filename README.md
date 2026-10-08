<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/978b76bc-cd9b-4af8-b1f3-18efde7c079f">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/ec4e391a-9044-44ae-93f0-9dd8bed70001">
  <img alt="A dark Proxima logo in light color mode and a light one in dark color mode." src="https://github.com/user-attachments/assets/ec4e391a-9044-44ae-93f0-9dd8bed70001" width=400px>
</picture>

# ConStellaration: A dataset of QI-like stellarator plasma boundaries and optimization benchmarks

[ConStellaration](https://arxiv.org/abs/2506.19583) is a dataset of diverse QI-like stellarator plasma boundary shapes and optimization benchmarks, paired with their ideal-MHD equilibria and performance metrics.
The dataset is available on [Hugging Face](https://huggingface.co/datasets/proxima-fusion/constellaration).
The repository contains a suite of tools and notebooks for exploring the dataset, including a forward model for plasma simulation, scoring functions for optimization evaluation and data-driven generative modeling.

## Reproducibility and benchmark versions

This repository is under active development: new metrics, problems and dependency updates (e.g. [VMEC++](https://github.com/proximafusion/vmecpp)) can change the values computed by the forward model and the scoring functions.
To keep results comparable, the benchmark of [Cadena et al., NeurIPS 2025](https://openreview.net/forum?id=NQSbGKlCpx) is frozen to a tagged release.

| Use case | Version |
| --- | --- |
| [ConStellaration leaderboard](https://huggingface.co/spaces/proxima-fusion/constellaration-bench) (geometrical, simple-to-build QI, MHD-stable QI problems) | [`v0.2.6`](https://github.com/proximafusion/constellaration/releases/tag/v0.2.6) |
| Original NeurIPS 2025 paper experiments | [`0.2.1`](https://pypi.org/project/constellaration/0.2.1/) |
| Latest development (not comparable to the leaderboard) | `main` / latest release |

If you are working on the challenge or comparing against the leaderboard, install the benchmark version:

```bash
pip install constellaration==0.2.6
```

or, from source:

```bash
git clone --branch v0.2.6 https://github.com/proximafusion/constellaration.git
cd constellaration
pip install .
```

The leaderboard evaluates every submission with exactly this version, so scores computed locally with `constellaration==0.2.6` match the leaderboard.
Newer releases may produce different metrics and scores and are not used by the leaderboard unless announced here.
The PyPI version always equals the git tag of the release it was built from (`constellaration==X.Y.Z` ⇔ tag `vX.Y.Z`).

Differences that affect benchmark scores:
- **`0.2.1` → `0.2.6`:** the rotational transform constraint uses the absolute value of the edge rotational transform, so boundaries with negative iota are no longer penalized. Newer VMEC++ and other numerical dependencies cause only negligible numerical differences.
- **`0.2.6` → `0.3.0`:** the flux compression in regions of bad curvature (a constraint of the MHD-stable problem) is computed with the field-aligned (PEST) Jacobian instead of the VMEC one ([#109](https://github.com/proximafusion/constellaration/pull/109)). This fixes a bug but changes MHD-stable feasibility, so `0.3.0` and later are not comparable to the leaderboard.

> [!NOTE]
> From 2026-10-02 to 2026-10-08 the leaderboard was briefly evaluated with `0.3.0`. The affected submissions have been re-evaluated with `0.2.6`.

## Installation

The following instructions have been tested on **Ubuntu 22.04** and **Ubuntu 24.04**. Other platforms may require additional steps and have not been validated.

The system dependency `libnetcdf-dev` is required for running the forward model. On Ubuntu, please ensure it is installed before proceeding, by running:

  ```bash
  sudo apt-get update
  sudo apt-get install build-essential cmake libnetcdf-dev
  ```

### Install from PyPI

The package can be installed directly from PyPI:

```bash
pip install constellaration
```

### Install by cloning the repository

1. Clone the repository:

  ```bash
  git clone https://github.com/proximafusion/constellaration.git
  cd constellaration
  ```

2. Install the required system dependencies
   1. **On Ubuntu**: `sudo apt-get update && sudo apt-get install -y libnetcdf-dev`
   2. **On macOS**: `brew install netcdf`

3. Install the required Python dependencies:

  ```bash
  pip install .
  ```

  **Note for macOS:** building `booz-xform` from source calls `python` from the `PATH`. If `python` does not resolve to the interpreter you are installing into (e.g. with a pyenv `system` global, where only `python3` exists), the build fails with `Could not find a package configuration file provided by "pybind11"`. Install into an activated virtual environment so that `python` points to it:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install .
  ```

### Running with Docker

If you prefer not to install system dependencies, you can use the provided Dockerfile to build a Docker image and run your scripts in a container.

1. Build the Docker image:

  ```bash
  docker build -t constellaration .
  ```

2. Run your scripts by mounting a volume to the container:

  ```bash
  docker run --rm -v $(pwd):/workspace constellaration python relative/path/to/your_script.py
  ```

Replace `your_script.py` with the path to your script. The `$(pwd)` command mounts the current directory to `/workspace` inside the container.

## Explanation Notebook

You can explore the functionalities of the repo through the [Boundary Explorer Notebook](https://github.com/proximafusion/constellaration/blob/main/notebooks/boundary_explorer.ipynb).

## Contributing

To be able to run unit tests, please install the test and lint environment:

```bash
pip install -e ".[test,lint]"
```

**Note:** The development and test environment currently supports **Python 3.10** only. Other Python versions are not guaranteed to work.
### Linting

We use **pre-commit** to automatically lint and format code before each commit. Linting is static code analysis that catches style issues and potential errors. If any **hook** fails, the commit will be blocked until you fix the reported issues and re-stage your changes.

Install the hook (once per clone):
```bash
pip install pre-commit
pre-commit install
```

You can run all pre-commit hooks against all files like this:
```bash
pre-commit run --all-files
```
### Unit tests

To locally run all unit tests (while in the top directory of the repo)

```bash
pytest .
```

## Optimization baseline

The optimization baseline can be executed by running the individual files within the folder `optimization_examples`.

## Citation

```
@inproceedings{
cadena2025constellaration,
title={ConStellaration: A dataset of {QI}-like stellarator plasma boundaries and optimization benchmarks},
author={Santiago A Cadena and Andrea Merlo and Emanuel Laude and Alexander Bauer and Atul Agrawal and Maria Pascu and Marija Savtchouk and Lukas Bonauer and Enrico Guiraud and Stuart R. Hudson and Markus Kaiser},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=NQSbGKlCpx}
}
```
