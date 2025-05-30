# Hugging Face Competition Evaluation: ConStellaration

This directory contains the Gradio web app and Docker setup for evaluating entries in the Hugging Face competition.

## How to Build and Run the Docker Image

1. **Build the Docker image:**

   Make sure you are in the `hugging_face_competition` directory (the one containing the `Dockerfile`) before running:
   ```bash
   docker build -t constellaration-eval .
   ```

2. **Run the Gradio app:**

   The Gradio app provides a web interface for uploading input files and selecting the problem type for evaluation. To start the app, run:
   ```bash
   docker run --rm -p 7860:7860 constellaration-eval
   ```
   This will launch the Gradio interface at [http://localhost:7860](http://localhost:7860).

## Input File Format

- For `geometrical` and `simple_to_build` problems, the input should be a single JSON object describing a boundary (see `inputs/boundary.json`).
- For `mhd_stable`, the input should be a JSON array of boundary objects, each as a string (see `inputs/boundaries.json`).

## Evaluation Logic

- The evaluation logic is in `evaluation.py`.
- The app uses the [constellaration](https://pypi.org/project/constellaration/) package (version pinned in Dockerfile).
- Results are displayed in the Gradio interface.
