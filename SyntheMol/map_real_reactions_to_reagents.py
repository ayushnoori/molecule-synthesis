"""Determines which REAL reagents can be used in which REAL reactions."""
import json
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from tap import Tap
from tqdm import tqdm

from constants import (
    REAL_REACTION_COL,
    REAL_REAGENT_COLS,
    REAL_SPACE_SIZE,
    REAL_TYPE_COL
)


class Args(Tap):
    data_dir: Path  # Path to directory with CXSMILES files containing the REAL database.
    save_path: Path  # Path to JSON file where mapping will be saved.
    parallel: bool = False  # Whether to run the script in parallel across files rather than sequentially.

    def process_args(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)


def map_reactions_for_file(path: Path) -> tuple[str, int, dict[int, dict[int, dict[str, set[int]]]]]:
    """Computes the mapping for a single file."""
    # Create mapping from reaction ID to reagent number to reaction type to valid reagent IDs
    reaction_to_reagents: dict[int, dict[int, dict[str, set[int]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # Load REAL data file (ensures cols are in the order of usecols for itertuples below)
    usecols = [REAL_REACTION_COL] + REAL_REAGENT_COLS + [REAL_TYPE_COL]
    data = pd.read_csv(path, sep='\t', usecols=usecols)[usecols]

    # Update mapping
    for reaction, reagent_1, reagent_2, reagent_3, reagent_4, reaction_type in data.itertuples(index=False):
        for reagent_index, reagent in enumerate([reagent_1, reagent_2, reagent_3, reagent_4]):
            if not np.isnan(reagent):
                reaction_to_reagents[reaction][reagent_index][reaction_type].add(int(reagent))

    # Convert to regular dict for compatibility with multiprocessing
    reaction_to_reagents = {
        reaction: {
            reagent_index: {
                reaction_type: reagent_ids
                for reaction_type, reagent_ids in reaction_type_to_reagent_ids.items()
            }
            for reagent_index, reaction_type_to_reagent_ids in reagent_mapping.items()
        }
        for reaction, reagent_mapping in reaction_to_reagents.items()
    }

    # Get data name
    name = path.stem.split('.')[0]

    return name, len(data), reaction_to_reagents


def map_real_reactions_to_reagents(args: Args) -> None:
    """Determines which REAL reagents can be used in which REAL reactions."""
    # Get paths to data files
    data_paths = sorted(args.data_dir.rglob('*.cxsmiles.bz2'))
    print(f'Number of files = {len(data_paths):,}')

    # Create combined dictionary
    combined_reaction_to_reagents = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # Set up map function
    if args.parallel:
        # TODO: is spawn necessary instead of fork or was it just memory issues?
        pool = get_context('spawn').Pool()
        map_fn = pool.imap
    else:
        pool = None
        map_fn = map

    # Loop through all REAL space files
    num_files = total_size = 0
    with tqdm(total=REAL_SPACE_SIZE) as progress_bar:
        for name, data_size, reaction_to_reagents in map_fn(map_reactions_for_file, data_paths):
            num_files += 1
            total_size += data_size
            print(f'{name}: file num = {num_files:,} / {len(data_paths):,} | '
                  f'num mols = {data_size:,} | cumulative num mols = {total_size:,}')
            print()

            # Merge dictionary with combined dictionary
            for reaction, reagent_mapping in reaction_to_reagents.items():
                for reagent_index, reaction_type_to_reagent_ids in reagent_mapping.items():
                    for reaction_type, reagent_ids in reaction_type_to_reagent_ids.items():
                        combined_reaction_to_reagents[reaction][reagent_index][reaction_type] |= reagent_ids

            # Update progress bar
            progress_bar.update(data_size)

    print(f'Total number of molecules = {total_size:,}')

    if pool is not None:
        pool.close()

    # Convert sets to sorted lists for JSON serializability
    combined_reaction_to_reagents = {
        reaction: {
            reagent_index: {
                reaction_type: sorted(reagent_ids)
                for reaction_type, reagent_ids in reaction_type_to_reagent_ids.items()
            }
            for reagent_index, reaction_type_to_reagent_ids in reagent_mapping.items()
        }
        for reaction, reagent_mapping in combined_reaction_to_reagents.items()
    }

    # Save mapping
    with open(args.save_path, 'w') as f:
        json.dump(combined_reaction_to_reagents, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    map_real_reactions_to_reagents(Args().parse_args())
