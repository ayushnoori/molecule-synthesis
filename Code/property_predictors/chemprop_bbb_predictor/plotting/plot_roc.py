from pathlib import Path

import scikitplot as skplt
import matplotlib.pyplot as plt
import pandas as pd
import metriculous
import numpy as np

N_CROSS_VALIDATION_MODELS = 10
base_path = Path("./Data/property_predictors/chemprop_bbb_predictor/")

prediction_paths = [
    base_path / f"model_{i}_test_preds.csv" for i in range(0, N_CROSS_VALIDATION_MODELS)
]

predictions = [pd.read_csv(prediction_path) for prediction_path in prediction_paths]

fig, ax = plt.subplots(nrows=3, ncols=3)

model_id = 0
for row in ax:
    for col in row:
        data = predictions[model_id]

        n_rows = len(data)
        y_truth = data["BBB"].to_numpy()  # ground truth labels
        y_probabilities = data[
            "prediction"
        ].to_numpy()  # predicted probabilities generated by sklearn classifier
        y_predictions = np.stack((1 - y_probabilities, y_probabilities), axis=1)

        print(y_predictions)

        plt.style.use("dark_background")
        skplt.metrics.plot_roc_curve(
            y_truth, y_predictions, title=f"Model {model_id}", ax=col
        )
        model_id += 1

plt.show()