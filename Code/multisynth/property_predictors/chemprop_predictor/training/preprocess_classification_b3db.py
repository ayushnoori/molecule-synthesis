"""Transforms the B3DB classification data into a format that can be used by chemprop for training a classification model."""
from pathlib import Path

from tap import Tap
import pandas as pd


class Args(Tap):
    """Argument parser for this script"""

    data_path: Path  # Path to the B3DB_classification.tsv file
    save_path: Path  # Path to the output file


def preprocess_classification_b3db(args: Args) -> None:
    """Turns the BBB+/BBB- column to 0/1"""
    data = pd.read_csv(args.data_path, sep="\t")
    data.replace({"BBB+": 1, "BBB-": 0}, inplace=True)
    data.rename(columns={"SMILES": "smiles", "BBB+/BBB-": "BBB"}, inplace=True)
    lean_data = data[["smiles", "BBB"]]
    lean_data.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    preprocess_classification_b3db(Args().parse_args())
