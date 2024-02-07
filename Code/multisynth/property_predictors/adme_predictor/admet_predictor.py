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
