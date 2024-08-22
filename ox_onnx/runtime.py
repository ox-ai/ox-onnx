import importlib
from typing import List, Dict
from ox_onnx.model_id import MODEL_IDS, MODEL_INTERF_MAP
from ox_onnx.raise_errors import ModelNotInterfacedError

class OnnxModel:
    def __init__(self, model_ID: str = MODEL_IDS[0]) -> None:
        """
        Initialize the OnnxModel class with the given model ID.

        Args:
            model_ID (str): The ID of the model to be used. Must be one of the pre-interfaced models
                            listed in `MODEL_IDS`. Defaults to the first model in `MODEL_IDS`.

        Raises:
            ModelNotInterfasedError: If the given `model_ID` is not valid or has not been interfaced.
        """
        if model_ID not in MODEL_IDS:
            raise ModelNotInterfacedError(
                f"""ox_onnx : given model model_ID = {model_ID} is not valid or not interfaced yet. 
                Use interfaced models: {MODEL_IDS}"""
            )
        
        model_import: str = MODEL_INTERF_MAP[model_ID]["model_import"]
        # Dynamically import the model class from the specified module interfaces
        self.model = importlib.import_module(model_import).InterfaceModel()


    def load(self):
        """
        loads and returns the model with its methods 
        """

        return self.model
    

