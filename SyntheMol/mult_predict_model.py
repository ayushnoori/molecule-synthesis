"""Make predictions with a model."""
import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from tap import Tap
from tqdm import tqdm

from chemfunc.molecular_fingerprints import compute_fingerprints
from chemprop.utils import load_checkpoint

from train_model import build_chemprop_data_loader, chemprop_predict, sklearn_predict


from admet_ai import ADMETModel
from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    admet_ai_base_key: str
    multiplier: float


admet_ai_dataset_metadata: list[DatasetMetadata] = [
    # Absorption
    DatasetMetadata("HIA_Hou", 1.0),  # Good
    DatasetMetadata("Pgp_Broccatelli", -1.0),  # Bad
    DatasetMetadata("Bioavailability_Ma", 1.0),  # Good
    DatasetMetadata("PAMPA_NCATS", 1.0),  # Good
    # Distribution
    # DatasetMetadata("BBB_Martins", 1.0),  # ! CONSIDER TAKING OUT BC OF OTHER MODEL
    # Metabolism
    # DatasetMetadata(
    #     "CYP1A2_Veith", 0
    # ),  # Unclear - https://pubmed.ncbi.nlm.nih.gov/18473749/
    # DatasetMetadata(
    #     "CYP2C19_Veith", 0
    # ),  # Unclear - https://pubmed.ncbi.nlm.nih.gov/18473749/
    # DatasetMetadata(
    #     "CYP2C9_Veith", 0
    # ),  # Unclear - https://pubmed.ncbi.nlm.nih.gov/18473749/
    # DatasetMetadata(
    #     "CYP2D6_Veith", 0
    # ),  # Unclear - https://pubmed.ncbi.nlm.nih.gov/18473749/
    # DatasetMetadata(
    #     "CYP3A4_Veith", 0
    # ),  # Unclear - https://pubmed.ncbi.nlm.nih.gov/18473749/
    # DatasetMetadata("CYP2C9_Substrate_CarbonMangels", 0),  # Unclear
    # DatasetMetadata("CYP2D6_Substrate_CarbonMangels", 0),  # Unclear
    # DatasetMetadata(
    #     "CYP3A4_Substrate_CarbonMangels", 0
    # ),  # Unclear - https://pubmed.ncbi.nlm.nih.gov/18473749/
    # Excretion - No binary excretion datasets
    # Toxicity
    DatasetMetadata("Skin_Reaction", -1.0),  # Bad
    DatasetMetadata("Carcinogens_Lagunin", -1.0),  # Bad
    DatasetMetadata("hERG", -1.0),  # Bad - Blocks hERG
    DatasetMetadata("AMES", -1.0),  # Bad
    DatasetMetadata("DILI", -1.0),  # Bad
    DatasetMetadata("ClinTox", -1.0),  # Bad - Likelihood of toxicity
]

min_total_score = sum(
    list(
        [
            min(dataset_metadata.multiplier, 0)
            for dataset_metadata in admet_ai_dataset_metadata
        ]
    )
)

max_total_score = sum(
    list(
        [
            max(0, dataset_metadata.multiplier)
            for dataset_metadata in admet_ai_dataset_metadata
        ]
    )
)


def normalize_score(score: float) -> float:
    return (score - min_total_score) / (max_total_score - min_total_score)


admet_ai_model = ADMETModel()


def calculate_admet_score(smiles: str) -> float:
    predictions: dict[str, float] = admet_ai_model.predict(smiles=smiles)

    raw_score = sum(
        list(
            [
                dataset_metadata.multiplier
                * predictions[dataset_metadata.admet_ai_base_key]
                for dataset_metadata in admet_ai_dataset_metadata
                if dataset_metadata.admet_ai_base_key in predictions
            ]
        )
    )

    final_score = normalize_score(raw_score)  # Normalize to 0 to 1
    return final_score


class Args(Tap):
    data_path: Path  # Path to a CSV file containing SMILES.
    model_path: Path  # Path to a directory of model checkpoints or to a specific PKL or PT file containing a trained model.
    save_path: Optional[Path] = None  # Path to a CSV file where model predicitions will be saved. If None, defaults to data_path.
    smiles_column: str = 'smiles'  # Name of the column containing SMILES.
    preds_column_prefix: Optional[str] = None  # Prefix for the column containing model predictions.
    model_type: Literal['rf', 'mlp', 'chemprop']  # Type of model to use. 'rf' = random forest. 'mlp' = multilayer perceptron.
    fingerprint_type: Optional[Literal['morgan', 'rdkit']] = None  # Type of fingerprints to use as input features.
    average_preds: bool = False  # Whether to average predictions across models for an ensemble model.

    def process_args(self) -> None:
        if self.save_path is None:
            self.save_path = self.data_path

        self.save_path.parent.mkdir(parents=True, exist_ok=True)


def predict_sklearn(fingerprints: np.ndarray,
                    model_path: Path,
                    model_type: str) -> np.ndarray:
    """Make predictions with an sklearn model."""
    # Load model
    with open(model_path, 'rb') as f:
        if model_type == 'rf':
            model: RandomForestClassifier | RandomForestRegressor = pickle.load(f)
        elif model_type == 'mlp':
            model: MLPClassifier | MLPRegressor = pickle.load(f)
        else:
            raise ValueError(f'Model type "{model_type}" is not supported.')

    # Make predictions
    preds = sklearn_predict(model=model, fingerprints=fingerprints)

    return preds


def predict_chemprop(smiles: list[str],
                     fingerprints: Optional[np.ndarray],
                     model_path: Path) -> np.ndarray:
    """Make predictions with a chemprop model."""
    # Ensure reproducibility
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    # Build data loader
    data_loader = build_chemprop_data_loader(
        smiles=smiles,
        fingerprints=fingerprints
    )

    # Load model
    model = load_checkpoint(path=model_path)

    # Make predictions
    preds = chemprop_predict(
        model=model,
        data_loader=data_loader
    )

    return preds


def predict_model(smiles: list[str],
                  fingerprints: Optional[np.ndarray],
                  model_type: str,
                  model_path: Path) -> np.ndarray:
    """Make predictions with a model."""
    # Map fragments to model scores
    if model_type == 'chemprop':
        preds = predict_chemprop(
            smiles=smiles,
            fingerprints=fingerprints,
            model_path=model_path
        )
    else:
        preds = predict_sklearn(
            fingerprints=fingerprints,
            model_path=model_path,
            model_type=model_type
        )

    return preds


def predict_ensemble(smiles: list[str],
                     fingerprint_type: Optional[str],
                     model_type: str,
                     model_path: Path,
                     average_preds: bool = True) -> np.ndarray:
    """Make predictions with an ensemble of models."""
    # Check compatibility of model and fingerprint type
    if model_type != 'chemprop' and fingerprint_type is None:
        raise ValueError('Must define fingerprint_type if using sklearn model.')

    # Compute fingerprints
    if fingerprint_type is not None:
        fingerprints = compute_fingerprints(smiles, fingerprint_type=fingerprint_type)
    else:
        fingerprints = None

    # Get model paths
    if model_path.is_dir():
        model_paths = list(model_path.glob('*.pt' if model_type == 'chemprop' else '*.pkl'))

        if len(model_paths) == 0:
            raise ValueError(f'Could not find any models in directory {model_path}.')
    else:
        model_paths = [model_path]

    preds = np.array([
        predict_model(
            smiles=smiles,
            fingerprints=fingerprints,
            model_type=model_type,
            model_path=model_path
        ) for model_path in tqdm(model_paths, desc='models')
    ])

    admet_preds = np.array(
        [calculate_admet_score(smile) for smile in tqdm(smiles, desc="molecules")]
    )

    print(preds)

    print(admet_preds)

    full_preds = np.vstack((preds, admet_preds))

    model_names = [Path(model_path).stem for model_path in model_paths] + ['admet']

    if average_preds:
        mult_preds = np.prod(full_preds, axis=0)
        model_names += ['mult_comb']
        full_preds = np.vstack((full_preds, mult_preds))

    return (model_names, full_preds)


def make_predictions(args: Args) -> None:
    """Make predictions with a model and save them to a file."""
    # Load SMILES
    data = pd.read_csv(args.data_path)
    smiles = list(data[args.smiles_column])

    # Make predictions
    model_names, all_preds = predict_ensemble(
        smiles=smiles,
        fingerprint_type=args.fingerprint_type,
        model_type=args.model_type,
        model_path=args.model_path,
        average_preds=args.average_preds
    )

    # Define model string
    model_string = f'{args.model_type}{f"_{args.fingerprint_type}" if args.fingerprint_type is not None else ""}'
    preds_string = f'{f"{args.preds_column_prefix}_" if args.preds_column_prefix is not None else ""}{model_string}'

    # if args.average_preds:
    #     data[f'{preds_string}_ensemble_preds'] = all_preds
    # else:
    #     for model_num, (model_name, preds) in enumerate(zip(model_names, all_preds)):
    #         data[f"{preds_string}_model_{model_name}_preds"] = preds

    for model_num, (model_name, preds) in enumerate(zip(model_names, all_preds)):
        data[f"{preds_string}_model_{model_name}_preds"] = preds

    # Save predictions
    data.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    make_predictions(Args().parse_args())
