"""Assess the quality and diversity of generated molecules."""
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import Tap

from chemfunc.molecular_similarities import compute_top_similarities


class Args(Tap):
    data_path: Path  # Path to CSV file containing generated molecules.
    save_dir: Path  # Path to directory where plots will be saved.
    reference_paths: Optional[
        list[Path]
    ] = None  # Optional list of paths to CSV files containing reference molecules for computing novelty.
    smiles_column: str = (
        "smiles"  # The name of the column containing SMILES in data_path.
    )
    reference_smiles_column: str = (
        "smiles"  # The name of the column containing SMILES in reference_paths.
    )
    score_column: str = "score"  # The name of the column containing scores.

    def process_args(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)


def plot_scores(
    scores: list[float],
    save_dir: Path,
    score_name: str = "score",
    score_color: Optional[str] = "#CDE0F6",
) -> None:
    """Plot score distribution.

    :param scores: A list of scores.
    :param save_dir: The directory where the plot will be saved.
    :param score_name: The name of the score.
    """
    # Plot score distribution
    # create a figure and axis
    plt.clf()
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # create a histogram
    ax.hist(
        scores,
        bins=100,
        color=score_color,
        edgecolor="black",
    )

    # label the axes
    plt.xlabel(score_name, fontweight="bold", size=12)
    plt.ylabel("Count", fontweight="bold", size=12)

    plt.title(f"{score_name} Distribution")

    # add a grid in the background
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)  # place grid lines behind bars

    # save as 600 DPI PNG in Results/figures
    plt.savefig(
        save_dir / f"{score_name}.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        save_dir / f"{score_name}.png",
        dpi=600,
        bbox_inches="tight",
    )

    # Save score distribution
    fig_data = pd.DataFrame({score_name: scores})
    fig_data.to_csv(save_dir / f"{score_name}.csv", index=False)


def plot_similarity(
    smiles: list[str],
    similarity_type: str,
    save_dir: Path,
    reference_smiles: Optional[list[str]] = None,
    reference_name: Optional[str] = None,
) -> None:
    """Plot similarity distribution within a list of SMILES or between that list and a reference list.

    :param smiles: A list of SMILES.
    :param similarity_type: The type of similarity.
    :param save_dir: The directory where the plot will be saved.
    :param reference_smiles: A list of reference SMILES to compare against.
    :param reference_name: The name of the reference list of SMILES.
    """
    # Compute similarities
    max_similarities = compute_top_similarities(
        similarity_type=similarity_type, mols=smiles, reference_mols=reference_smiles
    )

    # Get reference name
    if reference_name is None:
        reference_name = "Internal"

    # Get save name
    save_name = save_dir / f"{reference_name.lower()}_{similarity_type}_similarity"

    # Plot diversity distribution
    # create a figure and axis
    plt.clf()
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # create a histogram
    ax.hist(max_similarities, bins=100, color="#CDE0F6", edgecolor="black")

    # label the axes
    plt.xlabel(
        f"Maximum {reference_name} {similarity_type.title()} Similarity",
        fontweight="bold",
        size=12,
    )
    plt.ylabel("Count", fontweight="bold", size=12)

    plt.title(
        f"Maximum {reference_name} {similarity_type.title()} Similarity Distribution"
    )

    # add a grid in the background
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)  # place grid lines behind bars

    # save as 600 DPI PNG in Results/figures
    plt.savefig(
        f"{save_name}.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{save_name}.png",
        dpi=600,
        bbox_inches="tight",
    )

    # Save diversity distribution
    fig_data = pd.DataFrame({f"max_{similarity_type}_similarity": max_similarities})
    fig_data.to_csv(f"{save_name}.csv", index=False)


def plot_reference_similarity(
    smiles: list[str],
    reference_smiles: list[str],
    reference_name: str,
    similarity_type: str,
    save_dir: Path,
) -> None:
    """Plot similarity distribution between a list of SMILES and a reference list of SMILES.

    :param smiles: A list of SMILES.
    :param reference_smiles: A list of reference SMILES to compare against.
    :param reference_name: The name of the reference list of SMILES.
    :param similarity_type: The type of similarity.
    :param save_dir: The directory where the plot will be saved.
    """
    # Compute maximum similarity to reference SMILES
    max_similarities = compute_top_similarities(
        similarity_type=similarity_type, mols=smiles, reference_mols=reference_smiles
    )

    # Get reference file name
    reference_file_name = reference_name.lower().replace(" ", "_")

    # Plot diversity distribution compared to train
    # create a figure and axis
    plt.clf()
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # create a histogram
    ax.hist(max_similarities, bins=100, color="#CDE0F6", edgecolor="black")

    # label the axes
    plt.xlabel(
        f"Maximum {similarity_type.title()} Similarity from Generated to {reference_name}",
        fontweight="bold",
        size=12,
    )
    plt.ylabel("Count", fontweight="bold", size=12)

    plt.title(
        f"{reference_name} Maximum {similarity_type.title()} Similarity Distribution"
    )

    # add a grid in the background
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)  # place grid lines behind bars

    # save as 600 DPI PNG in Results/figures
    plt.savefig(
        save_dir / f"{reference_file_name}_{similarity_type}_similarity.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        save_dir / f"{reference_file_name}_{similarity_type}_similarity.png",
        dpi=600,
        bbox_inches="tight",
    )

    # Save similarity distribution
    fig_data = pd.DataFrame({f"max_{similarity_type}_similarity": max_similarities})
    fig_data.to_csv(
        save_dir / f"{reference_file_name}_{similarity_type}_similarity.csv",
        index=False,
    )


def plot_reactions_counts(num_reactions: list[int], save_dir: Path) -> None:
    """Plot the frequency of each number of reactions.

    :param num_reactions: A list of numbers of reactions per molecule.
    :param save_dir: The directory where the plot will be saved.
    """
    # Get reaction counts
    reaction_counts = Counter(num_reactions)
    min_reactions, max_reactions = min(reaction_counts), max(reaction_counts)
    reaction_nums = range(min_reactions, max_reactions + 1)
    reaction_counts = [
        reaction_counts[num_reactions] for num_reactions in reaction_nums
    ]

    # Plot reaction counts
    # create a figure and axis
    plt.clf()
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # create a histogram
    ax.bar(reaction_nums, reaction_counts, color="#CDE0F6", edgecolor="black")
    plt.xticks(reaction_nums)

    # label the axes
    plt.xlabel("Number of Reactions", fontweight="bold")
    plt.ylabel("Count", fontweight="bold")
    plt.title("Number of Reactions")

    # add a grid in the background
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)  # place grid lines behind bars

    # save as 600 DPI PNG in Results/figures
    plt.savefig(
        save_dir / "reaction_numbers.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        save_dir / "reaction_numbers.png",
        dpi=600,
        bbox_inches="tight",
    )

    # Save reaction counts
    fig_data = pd.DataFrame(
        {"reaction_nums": reaction_nums, "reaction_counts": reaction_counts}
    )
    fig_data.to_csv(save_dir / "reaction_numbers.csv", index=False)


def plot_reaction_usage(data: pd.DataFrame, save_dir: Path) -> None:
    """Plot the frequency with which each reaction is used (unique reactions per molecule).

    :param data: DataFrame containing reaction usage.
    :param save_dir: The directory where the plot will be saved.
    """
    # Get reaction usage
    reaction_columns = [
        column
        for column in data.columns
        if column.startswith("reaction_") and column.endswith("_id")
    ]
    reaction_data = data[reaction_columns]
    reaction_counter = Counter(
        reaction
        for _, reaction_row in reaction_data.iterrows()
        for reaction in {int(reaction) for reaction in reaction_row.dropna()}
    )

    reactions = sorted(reaction_counter)  # ! We don't want it sorted
    # reactions = [p[0] for p in reaction_counter.most_common()]
    # reactions.reverse()

    reaction_counts = [reaction_counter[reaction] for reaction in reactions]
    xticks = np.arange(len(reaction_counts))

    # Plot reaction usage
    # create a figure and axis
    plt.clf()
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # create a histogram
    ax.bar(xticks, reaction_counts, color="#CDE0F6", edgecolor="black")
    plt.xticks(ticks=xticks, labels=reactions, rotation=45)

    # label the axes
    plt.title("Reaction Counts")
    plt.xlabel("Reaction", fontweight="bold")
    plt.ylabel("Count (# molecules containing the reaction)", fontweight="bold")

    # add a grid in the background
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)  # place grid lines behind bars

    # save as 600 DPI PNG in Results/figures
    plt.savefig(
        save_dir / "reaction_counts.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        save_dir / "reaction_counts.png",
        dpi=600,
        bbox_inches="tight",
    )

    # Save reaction usage
    fig_data = pd.DataFrame({"reaction": reactions, "count": reaction_counts})
    fig_data.to_csv(save_dir / "reaction_counts.csv", index=False)


def plot_fragment_usage(data: pd.DataFrame, save_dir: Path) -> None:
    """Plot the frequency with which each fragment is used (unique fragments per molecule).

    :param data: DataFrame containing fragment usage.
    :param save_dir: The directory where the plot will be saved.
    """
    # Get fragment usage
    reagent_columns = [
        column
        for column in data.columns
        if column.startswith("building_block_") and column.endswith("_id")
    ]
    reagent_data = data[reagent_columns]
    fragment_counter = Counter(
        reagent
        for _, reagent_row in reagent_data.iterrows()
        for reagent in {int(reagent) for reagent in reagent_row.dropna()}
        if reagent != -1
    )

    fragments_with_counts = fragment_counter.most_common()
    print(fragments_with_counts[:10])
    fragments, fragment_counts = zip(*fragments_with_counts)

    # Plot fragment usage
    # create a figure and axis
    plt.clf()
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # create a histogram
    ax.scatter(
        np.arange(len(fragment_counts)),
        fragment_counts,
        color="#CDE0F6",
        edgecolor="black",
    )

    # label the axes
    plt.xlabel("Sorted Fragment Index", fontweight="bold")
    plt.ylabel("Count (# molecules containing the fragment)", fontweight="bold")
    plt.title("Fragment Counts")

    # add a grid in the background
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)  # place grid lines behind bars

    # save as 600 DPI PNG in Results/figures
    plt.savefig(
        save_dir / "fragment_counts.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        save_dir / "fragment_counts.png",
        dpi=600,
        bbox_inches="tight",
    )

    # Save fragment usage
    fig_data = pd.DataFrame({"fragment": fragments, "count": fragment_counts})
    fig_data.to_csv(save_dir / "fragment_counts.csv", index=False)


def assess_generated_molecules(args: Args) -> None:
    """Assess the quality and diversity of generated molecules."""
    # Load generated molecules
    data = pd.read_csv(args.data_path)

    # Count molecules
    print(f"Number of molecules = {len(data):,}")

    if True:
        print("Plotting score distribution...")

        is_random = "random" in str(args.data_path).lower()

        # Score distribution
        plot_scores(
            scores=data[args.score_column],
            save_dir=args.save_dir,
            score_name="Score",
            score_color="#E84147" if is_random else "#56AE72",
        )

    if True:
        print("Plotting similarity within generated molecules...")

        # Similarity within generated molecules
        plot_similarity(
            smiles=data[args.smiles_column],
            similarity_type="tanimoto",
            save_dir=args.save_dir,
        )

    if True:
        if args.reference_paths is not None:
            for reference_path in args.reference_paths:
                # Load reference molecules
                reference_molecules = pd.read_csv(reference_path)

                print(
                    f"Number of reference molecules in {reference_path.stem} = {len(reference_molecules):,}"
                )

                # Similarity between generated molecules and reference molecules
                plot_similarity(
                    smiles=data[args.smiles_column],
                    similarity_type="tversky",
                    save_dir=args.save_dir,
                    reference_smiles=reference_molecules[args.reference_smiles_column],
                    reference_name=reference_path.stem,
                )

    if True:
        print("Plotting reaction counts...")

        # Number of reactions
        plot_reactions_counts(
            num_reactions=data["num_reactions"], save_dir=args.save_dir
        )

    if True:
        print("Plotting reaction usage...")

        # Usage of reactions
        plot_reaction_usage(data=data, save_dir=args.save_dir)

    if True:
        # Usage of fragments
        plot_fragment_usage(data=data, save_dir=args.save_dir)


if __name__ == "__main__":
    assess_generated_molecules(Args().parse_args())
