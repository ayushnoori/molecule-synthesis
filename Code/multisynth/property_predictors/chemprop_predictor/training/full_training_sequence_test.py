"""Takes in the B3DB_classification.tsv file and trains the property predictor model."""
from pathlib import Path
# import tempfile
# import os

from tap import Tap
import pandas as pd

# from preprocess_classification_b3db import (
#     preprocess_classification_b3db,
#     Args as PreprocessArgs,
# )
from train_model import train_model, Args as TrainArgs


class Args(Tap):
    """Argument parser for this script"""

    data_path: Path  # Path to the classifications.csv file with columns "smiles" and "<property_column_name>"
    property_column_name: str  # Name of the column containing the property to predict
    save_dir: Path  # Path to the output directory where the models will be saved


def full_training_sequence(args: Args) -> None:
    args.save_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_data_path: Path = args.data_path
    # preprocessed_data_path: Path = args.save_dir / "preprocessed_data.csv"

    # print("========== BEGIN Preprocessing Data ==========")
    # print(f"Loading data from {args.data_path}")

    # preprocess_args = PreprocessArgs()
    # preprocess_args.from_dict(
    #     {
    #         "data_path": args.data_path,
    #         "save_path": preprocessed_data_path,
    #     }
    # )
    # preprocess_classification_b3db(preprocess_args)
    # print(f"Preprocessed data saved to {preprocessed_data_path}")
    # print("========== END Preprocessing Data ==========")

    print("========== BEGIN Training Model ==========")
    print(f"Training model on {preprocessed_data_path}")
    print(f"Saving models to {args.save_dir}")
    train_args = TrainArgs()
    train_args.from_dict(
        {
            "data_path": preprocessed_data_path,
            "save_dir": args.save_dir,
            "model_type": "chemprop",
            "dataset_type": "classification",
            "fingerprint_type": "rdkit",
            "property_column": args.property_column_name,
            "num_models": 10,
        }
    )
    train_model(train_args)
    print("========== END Training Model ==========")


if __name__ == "__main__":
    full_training_sequence(Args().parse_args())
