import re
from typing import Optional

import matplotlib as mpl  # TODO Fix this import not working
import matplotlib.text
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy.typing as npt
import seaborn as sns


# ============================
# ========== Colors ==========
# ============================

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"


def _generate_color_gradient(
    from_hex_color: str, to_hex_color: str, steps: int = 50
) -> list[str]:
    """
    Returns a list of <steps> html-style colour codes forming a gradient between the two supplied colours
    steps is an optional parameter with a default of 50
    If fromRGB or toRGB is not a valid colour code (omitting the initial hash sign is permitted),
    an exception is raised.

    Adapted from Peter Cahill 2020 (https://medium.com/@gobbagpete/hi-callum-32289a40fe79)
    """
    hex_rgb_re = re.compile(
        "#?[0–9a-fA-F]{6}"
    )  # So we can check format of input html-style colour codes
    # The code will handle upper and lower case hex characters, with or without a # at the front
    if not hex_rgb_re.match(from_hex_color) or not hex_rgb_re.match(to_hex_color):
        raise Exception(
            "Invalid parameter format"
        )  # One of the inputs isn"t a valid rgb hex code

    # Tidy up the parameters
    from_hex_letters = from_hex_color.split("#")[-1]
    to_hex_letters = to_hex_color.split("#")[-1]

    # Extract the three RGB fields as integers from each (from and to) parameter
    red_from, green_from, blue_from = [
        (int(from_hex_letters[n : n + 2], 16))
        for n in range(0, len(from_hex_letters), 2)
    ]
    red_to, green_to, blue_to = [
        (int(to_hex_letters[n : n + 2], 16)) for n in range(0, len(to_hex_letters), 2)
    ]

    # For each colour component, generate the intermediate steps
    red_steps = [
        f"{round(red_from + n * (red_to - red_from) / (steps - 1)):02x}"
        for n in range(steps)
    ]
    green_steps = [
        f"{round(green_from + n * (green_to - green_from) / (steps - 1)):02x}"
        for n in range(steps)
    ]
    blue_steps = [
        f"{round(blue_from + n * (blue_to - blue_from) / (steps - 1)):02x}"
        for n in range(steps)
    ]

    # Reassemble the components into a list of html-style #rrggbb codes
    return [f"#{r}{g}{b}" for r, g, b in zip(red_steps, green_steps, blue_steps)]


def set_color_pallete(colors: Optional[list[str]]):
    color_list = (
        [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]
        if colors is None
        else colors
    )
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=color_list)


def set_style(for_notebook: bool = True):
    sns.set(
        font="DejaVu Sans",
        rc={
            "axes.axisbelow": False,
            "axes.edgecolor": "lightgrey",
            "axes.facecolor": "None",
            "axes.grid": False,
            "axes.labelcolor": "dimgrey",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "white",
            "lines.solid_capstyle": "round",
            "patch.edgecolor": "w",
            "patch.force_edgecolor": True,
            "text.color": "dimgrey",
            "xtick.bottom": False,
            "xtick.color": "dimgrey",
            "xtick.direction": "out",
            "xtick.top": False,
            "ytick.color": "dimgrey",
            "ytick.direction": "out",
            "ytick.left": False,
            "ytick.right": False,
        },
    )
    if for_notebook:
        sns.set_context(
            "notebook", rc={"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18}
        )


def plt_prettify(
    remove_legend_frame: bool = True,
    despine_left: bool = False,
    despine_bottom: bool = False,
    despine_right: bool = False,
    despine_top: bool = False,
):
    """
    Prettify the plot.

    :param remove_legend_frame: Whether to remove the frame of the legend. Default is True. The little boxes around legends are often unnecessary, and add visual clutter.
    """
    if remove_legend_frame:
        plt.legend(frameon=False)

    if despine_left or despine_bottom or despine_right or despine_top:
        sns.despine(
            left=despine_left,
            bottom=despine_bottom,
            right=despine_right,
            top=despine_top,
        )


def add_numerical_labels_to_bar_chart(
    values: npt.ArrayLike,  # TODO retype to ArrayLike[Number]
    labels: list[matplotlib.text.Annotation],
    ax: matplotlib.axes.Axes,
):
    """
    Add numerical labels to a bar chart.

    :param values: The values of the bars.
    :param labels: The labels objects of the bars, as reutrned by matplotlib.
    """
    for n, i in enumerate(labels):
        # Create an axis text object
        ax.text(
            values[n] - 0.003,  # X location of text (with adjustment)
            n,  # Y location
            s=f"{round(values[n], 3)}%",  # Required label with formatting
            va="center",  # Vertical alignment
            ha="right",  # Horizontal alignment
            color="white",  # Font colour and size
            fontsize=12,
        )
