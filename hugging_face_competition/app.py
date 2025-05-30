import pathlib
import tempfile
from typing import BinaryIO, Literal

import gradio as gr
from evaluation import evaluate_problem


def evaluate_boundary(
    problem_type: Literal["geometrical", "simple_to_build", "mhd_stable"],
    boundary_file: BinaryIO,
) -> str:
    file_path = boundary_file.name
    if not file_path:
        return "Error: Uploaded file object does not have a valid file path."
    path_obj = pathlib.Path(file_path)
    with (
        path_obj.open("rb") as f_in,
        tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp,
    ):
        tmp.write(f_in.read())
        tmp_path = pathlib.Path(tmp.name)
    try:
        result = evaluate_problem(problem_type, str(tmp_path))
        output = str(result)
    except Exception as e:
        output = f"Error during evaluation:\n{e}"
    finally:
        tmp_path.unlink()
    return output


PROBLEM_TYPES = ["geometrical", "simple_to_build", "mhd_stable"]


def gradio_interface() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            """
        # Plasma Boundary Evaluation App
        Upload your plasma boundary JSON and select the problem type to get your score.
        """
        )
        with gr.Row():
            problem_type = gr.Dropdown(
                PROBLEM_TYPES, label="Problem Type", value="geometrical"
            )
            boundary_file = gr.File(label="Boundary JSON File (.json)")
        output = gr.Textbox(label="Evaluation Result", lines=10)
        submit_btn = gr.Button("Evaluate")
        submit_btn.click(
            evaluate_boundary,
            inputs=[problem_type, boundary_file],
            outputs=output,
        )
    return demo


if __name__ == "__main__":
    gradio_interface().launch()
