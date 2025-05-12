# ConStellaration: A dataset of QI-like stellarator plasma boundaries and optimization benchmarks

ConStellaration is a dataset of diverse QI-like stellarator plasma boundary shapes, paired with their ideal-MHD equilibria and performance metrics.  
The dataset is available on [Hugging Face](https://huggingface.co/datasets/proxima-fusion/constellaration).  
The repository contains a suite of tools and notebooks for exploring the dataset, including a forward model for plasma simulation and scoring functions for optimization evaluation.

## Installation

The system dependency `libnetcdf-dev` is required for running the forward model. Please ensure it is installed before proceeding, by running:

  ```bash
  sudo apt-get install libnetcdf-dev
  ```

1. Clone the repository:

  ```bash
  git clone https://github.com/proximafusion/constellaration.git
  cd constellaration
  ```

2. Install the required Python dependencies:

  ```bash
  pip install .
  ```

### Install directly from GitHub

To install the package directly from the GitHub repository without cloning, ensure the system dependency is installed first, then run:

```bash
pip install git+https://github.com/proximafusion/constellaration.git
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

You can explore the functionalities of the repo through the [Boundary Explorer Notebook](./notebooks/boundary_explorer.ipynb).

## Citation

If you find this work useful, please cite us:
