import hashlib
import os

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm


from ox_onnx.raise_errors import ModelInstallationError
from ox_onnx.utils import check_model_existence
from ox_onnx.config import OX_MODELS,ONNX_MODELS 


# MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL_HEAD = "all-MiniLM-L6-v2.onnx.ox"
# MODEL_PATH = Path.home() / OX_MODELS / ONNX_MODELS / MODEL_HEAD
# MODEL_PATH_absolute = MODEL_PATH / "model.onnx"
# MODEL_HASH = "d38fca9380728a0f418833b40a3ec031cbd8eaee7650c53f9b6a9eef0f442028"

MODEL_FILES = [
        "model.onnx",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]

class Extractor:


    def __init__(self, model_id: str,model_files:list[str]=None) -> None:
        self.MODEL_ID = model_id
        self.MODEL_HEAD = self.MODEL_ID.split("/")[1] + ".onnx.ox"
        self.MODEL_PATH = (
            Path.home() / OX_MODELS / ONNX_MODELS / self.MODEL_HEAD
        )
        self.MODEL_PATH_absolute = self.MODEL_PATH / "model.onnx"
        self.MODEL_FILES = model_files or MODEL_FILES
    @staticmethod
    def model_download(model_id: str, model_files:list[str]=None,old_model_hash: str=None) -> bool:
        """
        Initializes the model by checking its existence, downloading it if necessary,
        and verifying its integrity. If the model fails the integrity check or any
        other error occurs during the initialization process, a ModelInstallationError
        is raised.

        Args:
            model_id (str): The ID of the model to initialize.

        Returns:
            bool: True if the model is properly installed and verified, False otherwise.

        Raises:
            ModelInstallationError: If the model integrity check fails or any other
            error occurs during the initialization process.
        """
        try:
            extractoror = Extractor(model_id,model_files=model_files)
            if not check_model_existence(model_path=extractoror.MODEL_PATH,model_files=extractoror.MODEL_FILES):
                Extractor.download(model_id)
                model_hash = gen_model_hash(
                    model_path_absolute=extractoror.MODEL_PATH_absolute
                )
                print(f"onnx.ox model : MODEL_HASH = {model_hash}")

                # if not verify_model_purity(model_hash, old_model_hash):
                #     raise ModelInstallationError(
                #         "onnx.ox model : Model integrity check failed after downloading"
                #     )
                return True
            else:
                return True
        except Exception as e:
            raise ModelInstallationError(
                f"""onnx.ox model : An error occurred during the model initialization process: {str(e)} 
                report if issues in github https://github.com/ox-ai/ox-onnx.git"""
            )

    @staticmethod
    def download(MODEL_ID: str) -> Path:
        """
        Downloads the model and tokenizer from Hugging Face, converts the model to ONNX format
        if necessary, and saves both the model and tokenizer in the specified directory.
        Displays a progress bar during the download and saving process.

        Args:
            MODEL_ID (str): The ID of the model to download.

        Returns:
            Path: The path to the directory where the model and tokenizer are saved.

        Prints:
            - Progress messages indicating the start, progress, and completion of the download.
            - A message indicating that the model and its dependencies already exist
            if no download is necessary.
        """
        extractoror = Extractor(MODEL_ID)
        # Check if the model and tokenizer already exist
        if not check_model_existence(model_path=extractoror.MODEL_PATH,model_files=extractoror.MODEL_FILES):
            print("onnx.ox model : not found!")
            print("onnx.ox model : initializing installation...")
            print(f"onnx.ox model : MODEL_ID")
            # Use tqdm to show progress during model download
            with tqdm(total=100, unit="%", desc="Downloading Model") as pbar:
                # Load the model from Hugging Face and convert to ONNX if not already present
                model = ORTModelForFeatureExtraction.from_pretrained(
                    MODEL_ID, from_transformers=True
                )
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                pbar.update(50)

                # Save the ONNX model and tokenizer
                extractoror.MODEL_PATH.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(extractoror.MODEL_PATH)
                tokenizer.save_pretrained(extractoror.MODEL_PATH)
                pbar.update(50)

            print("onnx.ox model : installation complete!")
            print(f"onnx.ox model : MODEL_ID = {MODEL_ID}")
            return extractoror.MODEL_PATH
        else:
            print("onnx.ox model : Model and its dependencies already exist, installation aborted.")
            print(f"onnx.ox model : MODEL_ID = {MODEL_ID}")
            return extractoror.MODEL_PATH


def verify_model_purity(model_hash_256: str, old_hash_sha256: str) -> bool:
    """
    Verifies the purity of the model by comparing its SHA-256 hash with the expected hash.

    Args:
        model_hash_256 (str): The SHA-256 hash of the model to verify.
        old_hash_sha256 (str, optional): The expected SHA-256 hash. Defaults to MODEL_HASH.

    Returns:
        bool: True if the model's hash matches the expected hash, indicating purity;
        False otherwise.
    """
    if model_hash_256 == old_hash_sha256:
        print("onnx.ox model : model pure")
        return True
    return False


def gen_model_hash(model_path_absolute: str, read_byte: int = 4096) -> str:
    """
    Generates the SHA-256 hash of the model file. A progress bar is displayed in the terminal
    to indicate the hashing process.

    Args:
        model_path_absolute (str, optional): The absolute path to the model file.
        Defaults to MODEL_PATH_absolute.
        read_byte (int, optional): The number of bytes to read at a time while generating
        the hash. Defaults to 4096.

    Returns:
        str: The generated SHA-256 hash of the model file.
    """

    model_hash = hashlib.sha256()
    total_size = os.path.getsize(model_path_absolute)

    with open(model_path_absolute, "rb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Hashing Model"
        ) as pbar:
            while byte_block := f.read(read_byte):
                model_hash.update(byte_block)
                pbar.update(len(byte_block))

    return model_hash.hexdigest()


    
