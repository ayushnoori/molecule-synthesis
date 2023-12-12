from dataclasses import dataclass
from typing import Optional, TypeVar, cast, Union, Any
from pathlib import Path

from tdc.single_pred.single_pred_dataset import DataLoader as TdcDataLoader
from DeepPurpose import utils as DeepPurposeUtils, CompoundPred

from utils.file_sanitization import sanitize_filesystem_name

DRUG_ENCODING = "rdkit_2d_normalized"


def make_single_prediction_model() -> CompoundPred.Property_Prediction:
    config = DeepPurposeUtils.generate_config(
        drug_encoding=DRUG_ENCODING,
        train_epoch=30,  # TODO Tune this because 3 was very low and 3 was very high, probably in the tens magnitude
        LR=0.001,
        batch_size=128,
        mpnn_hidden_size=32,
        mpnn_depth=2,
    )
    model = CompoundPred.model_initialize(**config)
    return model


MockMultiModel = TypeVar("MockMultiModel", bound="MultiModel")


@dataclass
class MultiModelSinglePredictionModelArgument:
    name: str
    model: Optional[CompoundPred.Property_Prediction]
    dataloader: Optional[TdcDataLoader]


@dataclass
class MultiModelSinglePredictionModel:
    name: str
    model: CompoundPred.Property_Prediction
    dataloader: Optional[TdcDataLoader]


class MultiModel:
    """Utility class for maintaining multiple single prediction models."""

    augmented_models: list[MultiModelSinglePredictionModel] = []

    def __init__(
        self, augmented_models: list[MultiModelSinglePredictionModelArgument]
    ) -> None:
        """
        Initialize a MultiModel.

        :param augmented_models: A list of trained models or their dataloaders. The list must contain at least one object.

        :raises ValueError: If a MultiModelSinglePredictionModel object does not have a model nor a dataloader. It should have one.
        :raises ValueError: If a MultiModelSinglePredictionModel object has both a model and a dataloader. It should only have one.
        """
        if len(augmented_models) == 0:
            raise ValueError(
                f"Argument models = {augmented_models} must be a list of at least one model."
            )

        for augmented_model in augmented_models:
            if augmented_model.model is None and augmented_model.dataloader is None:
                raise ValueError(
                    "Either models or dataloaders must be provided to initialize a MultiModel."
                )
            if augmented_model.dataloader is not None:
                if augmented_model.model is not None:
                    raise ValueError(
                        "Both models and dataloaders were provided. Only provide one of them. Either you provide the trained models or the dataloaders to train the models."
                    )
                augmented_model.model = make_single_prediction_model()

        # Sanitize model names
        for augmented_model in augmented_models:
            augmented_model.name = sanitize_filesystem_name(augmented_model.name)

        self.augmented_models = cast(
            list[MultiModelSinglePredictionModel], augmented_models
        )

    @classmethod
    def load_pretrained_multi_model(
        cls: type[MockMultiModel],
        model_dir: Path,
    ) -> MockMultiModel:
        """
        Load a pretrained multi model from a directory.

        :param model_dir: The directory containing the model.

        :return: The loaded model, instance of MultiModel class.
        """

        if not model_dir.is_dir():
            raise ValueError(f"Path model_dir = {model_dir} is not a directory.")

        model_subdirectories = [
            child for child in model_dir.iterdir() if child.is_dir()
        ]

        if len(model_subdirectories) == 0:
            raise ValueError(
                f"No subdirectories found in model_dir = {model_dir}. This means that the MultiModel doesn not contain any single prediction models."
            )

        def load_model(subdir: Path) -> MultiModelSinglePredictionModelArgument:
            raw_loaded_model = CompoundPred.model_pretrained(subdir)
            return MultiModelSinglePredictionModelArgument(
                name=subdir.name,
                model=raw_loaded_model,
                dataloader=None,
            )

        loaded_models = [load_model(subdir) for subdir in model_subdirectories]

        return cls(augmented_models=loaded_models)

    def save(self, model_dir: Union[Path, str]) -> None:
        """
        Save the model to a directory.

        :param model_dir: The directory to save the model to.
        """
        model_dir = Path(model_dir)  # Ensure that model_dir is a Path object
        model_dir.mkdir(parents=True, exist_ok=True)
        for augmented_model in self.augmented_models:
            augmented_model.model.save_model(
                str(
                    model_dir / augmented_model.name
                )  # No need to sanizize the name here, it is already sanitized in the __init__ method
            )

    def train(self) -> None:
        """
        Train the models in the MultiModel using the dataloaders.
        """
        for i, augmented_model in enumerate(self.augmented_models):
            if augmented_model.dataloader is None:
                print(
                    f"Model [{i}/{len(self.augmented_models)}] '{augmented_model.name}' does not have a dataloader. Skipped."
                )
            X, y = augmented_model.dataloader.get_data(format="DeepPurpose")
            train, val, test = DeepPurposeUtils.data_process(
                X_drug=X,
                y=y,
                drug_encoding=DRUG_ENCODING,
                random_seed=0,  # ! Set seed to 0 for reproducibility
            )
            augmented_model.model.train(train, val, test)

    def predict(self, smiles_strings: list[str], fingerprint: Optional[Any]=None) -> dict[str, float]:
        """
        Predict the properties of a list of compounds.

        :param smiles_strings: A list of SMILES strings.
        :param fingerprint: A fingerprint object. If None, the fingerprint will be calculated using the RDKit library based on the smiles.

        :return: A dictionary with the predicted properties. The key is the name of the .
        """
        raise Error("This method is not implemented yet.")
