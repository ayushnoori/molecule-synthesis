from ..property_predictors.multi_model import MultiModel

class SynthesisGuider:
    multi_models: list[MultiModel] = []

    def __init__(self, multi_models: list[MultiModel]) -> None:
        """
        Initialize a SynthesisGuider.

        :param multi_models: A list of MultiModel objects. The list must contain at least one object.

        :raises ValueError: If a MultiModel object does not have any models. It should have at least one.
        """
        if len(multi_models) == 0:
            raise ValueError(
                f"Argument multi_models = {multi_models} must be a list of at least one MultiModel."
            )

        self.multi_models = multi_models

    
