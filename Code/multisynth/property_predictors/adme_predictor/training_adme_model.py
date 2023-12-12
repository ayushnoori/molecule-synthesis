from typing import Any, cast

from DeepPurpose import utils, CompoundPred
from tdc.single_pred import ADME
from tdc.utils import retrieve_dataset_names

from .property_predictors.multi_model import MultiModel, MultiModelSinglePredictionModelArgument

adme_datasets = cast(list[str], retrieve_dataset_names("ADME"))
print(f"Number of ADME datasets available from TDC: {len(adme_datasets)}")
print("**Datasets available:")
for i, dataset in enumerate(adme_datasets):
    print(f"{i + 1}. {dataset}")
