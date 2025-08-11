<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/978b76bc-cd9b-4af8-b1f3-18efde7c079f">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/ec4e391a-9044-44ae-93f0-9dd8bed70001">
  <img alt="A dark Proxima logo in light color mode and a light one in dark color mode." src="https://github.com/user-attachments/assets/ec4e391a-9044-44ae-93f0-9dd8bed70001" width=400px>
</picture>

# ConStellaration: A dataset of QI-like stellarator plasma boundaries and optimization benchmarks

[ConStellaration](https://arxiv.org/abs/2506.19583) is a dataset of diverse QI-like stellarator plasma boundary shapes and optimization benchmakrs, paired with their ideal-MHD equilibria and performance metrics.
The dataset is available on [Hugging Face](https://huggingface.co/datasets/proxima-fusion/constellaration).
The repository contains a suite of tools and notebooks for exploring the dataset, including a forward model for plasma simulation, scoring functions for optimization evaluation and data-driven generative modeling.

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
   2. **On MAC-OS**: `brew install netcdf`

3. Install the required Python dependencies:

  ```bash
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
@article{cadena2025constellaration,
  title={ConStellaration: A dataset of QI-like stellarator plasma boundaries and optimization benchmarks},
  author={Cadena, Santiago A and Merlo, Andrea and Laude, Emanuel and Bauer, Alexander and Agrawal, Atul and Pascu, Maria and Savtchouk, Marija and Guiraud, Enrico and Bonauer, Lukas and Hudson, Stuart and others},
  journal={arXiv preprint arXiv:2506.19583},
  year={2025}
}
```
