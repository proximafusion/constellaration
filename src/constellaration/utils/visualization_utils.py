import io
import json
from typing import Any

import IPython.display as ipd
import matplotlib.figure as mpl_figure
from matplotlib import axes
from PIL import Image
from plotly import graph_objects as go
from plotly import subplots as plotly_subplots


def combine_figures_side_by_side(
    fig1: mpl_figure.Figure | axes.Axes, fig2: mpl_figure.Figure | axes.Axes
) -> None:
    """Combines two matplotlib figures or axes side by side and displays the result."""

    # Convert axes to figures if necessary
    fig1 = _to_figure(fig1)
    fig2 = _to_figure(fig2)

    img1 = _figure_to_image(fig1)
    img2 = _figure_to_image(fig2)

    # Create a new blank image with combined width
    combined_width = img1.width + img2.width
    combined_height = max(img1.height, img2.height)
    combined_img = Image.new(
        "RGB", (combined_width, combined_height), color=(255, 255, 255)
    )

    # Paste both images side by side
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width, 0))

    ipd.display(combined_img)


def combine_plotly_figures_side_by_side(fig1: go.Figure, fig2: go.Figure) -> None:
    """Combines two Plotly 3D figures into a single side-by-side static image and
    displays the result."""
    fig = plotly_subplots.make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
    )
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)
    fig.update_layout(width=1000, height=500, showlegend=False)

    img_bytes = fig.to_image(format="png")
    ipd.display(ipd.Image(data=img_bytes))


def print_dicts_side_by_side(dict1: dict[str, Any], dict2: dict[str, Any]) -> None:
    """Prints two dictionaries side by side in a Jupyter notebook."""
    html = f"""
    <div style="display: flex; justify-content: space-around; font-family: monospace;">
        <div style="margin-right: 20px;">
            <pre>{json.dumps(dict1, indent=4)}</pre>
        </div>
        <div>
            <pre>{json.dumps(dict2, indent=4)}</pre>
        </div>
    </div>
    """
    ipd.display(ipd.HTML(html))


def _figure_to_image(fig: mpl_figure.Figure) -> Image.Image:
    """Converts a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


def _to_figure(fig: mpl_figure.Figure | axes.Axes) -> mpl_figure.Figure:
    """Converts an Axes object to a Figure object."""
    if isinstance(fig, axes.Axes):
        ax_figure = fig.get_figure()
        assert ax_figure is not None
        fig = ax_figure
    return fig
