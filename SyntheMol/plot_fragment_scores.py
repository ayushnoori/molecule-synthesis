"""Plots the distribution of model scores over molecular fragments."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tap import Tap


class Args(Tap):
    fragment_to_score_path: Path  # Path to JSON file containing a dictionary mapping from fragment SMILES to model scores.
    title: str  # Title of the plot.
    save_dir: Path  # Path to a directory where the plot will be saved.

    def process_args(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)


def plot_fragment_scores(args: Args) -> None:
    """Plots the distribution of model scores over molecular fragments."""
    # Load mapping from fragments to scores
    with open(args.fragment_to_score_path) as f:
        fragment_to_score: dict[str, float] = json.load(f)

    # Plot distribution of fragment scores
    scores = list(fragment_to_score.values())

    plt.hist(scores, bins=100)
    plt.xlabel('Model Score')
    plt.ylabel('Count')
    plt.title(args.title)
    plt.savefig(args.save_dir / 'fragment_scores.pdf', bbox_inches='tight')

    # Save data
    fig_data = pd.DataFrame({'score': scores})
    fig_data.to_csv(args.save_dir / 'fragment_scores.csv', index=False)


if __name__ == '__main__':
    plot_fragment_scores(Args().parse_args())
