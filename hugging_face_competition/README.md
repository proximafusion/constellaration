# Hugging Face Competition Evaluation: ConStellaration

This directory contains the  evaluation script and Docker setup for evaluating entries.

## How to Build and Run the Docker Image

1. **Build the Docker image:**

   Make sure you are in the `hugging_face_competition` directory (the one containing the `Dockerfile`) before running:
   ```bash
   docker build -t constellaration-eval .
   ```

2. **Run the evaluation:**

   The evaluation script expects two arguments:
   - `--problem-type`: One of `geometrical`, `simple_to_build`, or `mhd_stable`.
   - `--input-file`: Path to the input JSON file (see below for format).

   Example (for a single-objective problem):
   ```bash
   docker run --rm -v $(pwd)/inputs:/inputs constellaration-eval \
     --problem-type geometrical \
     --input-file /inputs/boundary.json
   ```

   Example (for a multi-objective problem):
   ```bash
   docker run --rm -v $(pwd)/inputs:/inputs constellaration-eval \
     --problem-type mhd_stable \
     --input-file /inputs/boundaries.json
   ```

## Input File Format

- For `geometrical` and `simple_to_build` problems, the input should be a single JSON object describing a boundary (see `inputs/boundary.json`).
- For `mhd_stable`, the input should be a JSON array of boundary objects, each as a string (see `inputs/boundaries.json`).

## Evaluation Script

- The evaluation logic is in `evaluation.py`.
- The script uses the [constellaration](https://pypi.org/project/constellaration/) package (version pinned in Dockerfile).
- Results are printed to stdout. You may redirect output to a file if needed.

## Example Inputs

- `inputs/boundary.json`: Example input for single-objective problems.
- `inputs/boundaries.json`: Example input for multi-objective problems.
