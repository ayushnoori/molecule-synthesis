from pathlib import Path
from typing import Any

import scikitplot as skplt
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt

N_CROSS_VALIDATION_MODELS = 10
N_ROWS = 3
N_COLS = 3


def _plt_plt_single_model_roc(
    model_id: int,
    truth_col_name: str,
    prediction_col_name: str,
    data: pd.DataFrame,
    ax: Axes,
):
    y_truth: npt.NDArray[np.uint16] = data[
        truth_col_name
    ].to_numpy()  # ground truth labels
    y_probabilities: npt.NDArray[np.float64] = data[
        prediction_col_name
    ].to_numpy()  # predicted probabilities generated by sklearn classifier
    y_predictions = np.stack((1 - y_probabilities, y_probabilities), axis=1)

    skplt.metrics.plot_roc(y_truth, y_predictions, title=f"Model {model_id}", ax=ax)


def _format_axes(ax: Any) -> npt.NDArray[npt.NDArray[Axes]]:
    if not isinstance(ax, np.ndarray):
        return np.asarray([[ax]])
    elif not isinstance(ax[0], np.ndarray):
        return np.asarray([ax])
    return ax


def plt_plot_chemprop_predictor_roc(
    base_model_path: Path,
    truth_col_name: str,
    prediction_col_name: str = "prediction",
    n_cross_validation_models: int = N_CROSS_VALIDATION_MODELS,
    n_rows: int = N_ROWS,
    n_cols: int = N_COLS,
) -> None:
    """
    Plot ROC curves for the models trained by chemprop.

    :param base_model_path: The path to the directory containing the models.
    :param n_cross_validation_models: The number of cross validation models to plot. Default is 10.

    :raises FileNotFoundError: If the model directory does not exist.
    :raises ValueError: If the number of cross validation models is less than 1.
    """
    prediction_paths = [
        base_model_path / f"model_{i}_test_preds.csv"
        for i in range(0, n_cross_validation_models)
    ]

    predictions = [pd.read_csv(prediction_path) for prediction_path in prediction_paths]

    _fig, ax = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(n_cols * 10, n_rows / 3 * 20)
    )

    list_of_list_ax = _format_axes(ax)

    model_id = 0
    for row in list_of_list_ax:
        for col in row:
            if model_id >= len(predictions):
                break
            data = predictions[model_id]

            _plt_plt_single_model_roc(
                model_id=model_id,
                truth_col_name=truth_col_name,
                prediction_col_name=prediction_col_name,
                data=data,
                ax=col,
            )

            model_id += 1
