import importlib
from typing import List, Dict, Type, cast
from ox_onnx.config import MODEL_IDS, MODEL_INTERF_MAP
from ox_onnx.raise_errors import ModelNotInterfacedError
from ox_onnx.interface.base import BaseInterfaceModel

class OnnxModel:
    def __init__(self) -> None:
        """
        Initialize the OnnxModel class 
        """
    @staticmethod
    def load( model_ID: str = MODEL_IDS[0]):
        """
        Initialize the OnnxModel class with the given model ID.

        Args:
            model_ID (str): The ID of the model to be used. Must be one of the pre-interfaced models
                            listed in `MODEL_IDS`. Defaults to the first model in `MODEL_IDS`.

        Returns:
            loads and returns the model with its methods 
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
        model_module = importlib.import_module(model_import)
        
        # Dynamically retrieve the Singleton InterfaceModel
        model_class= getattr(model_module, "InterfaceModel")


        # the singleton instance is returned
        model = cast(BaseInterfaceModel,model_class())
       
        return model
    

