"""
onnx model file from hugging fase
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/onnx/model.onnx
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/train_script.py
"""
# runtime for vec onnx model

from functools import cached_property
import importlib
from pathlib import Path
from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np
from typing import List

from ox_onnx.utils import check_model_existence
from ox_onnx.config import OX_MODELS,ONNX_MODELS 

# MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL_INTERF_MAP = "sentence_transformers_all_MiniLM_L6_v2"
# MODEL_HEAD = MODEL_ID.split("/")[1] + ".onnx.ox"
# MODEL_PATH = Path.home() / OX_MODELS / ONNX_MODELS / MODEL_HEAD
# MODEL_PATH_absolute = MODEL_PATH / "model.onnx"
# MODEL_HASH = "d38fca9380728a0f418833b40a3ec031cbd8eaee7650c53f9b6a9eef0f442028"
# MODEL_FILES = [
#         "model.onnx",
#         "config.json",
#         "special_tokens_map.json",
#         "tokenizer_config.json",
#         "tokenizer.json",
#     ]


# Interface implementation of the default sentence-transformers model using ONNX
class InterfaceModel:
    _instance = None
    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_INTERF_MAP = "sentence_transformers_all_MiniLM_L6_v2"
    MODEL_HEAD = MODEL_ID.split("/")[1] + ".onnx.ox"
    MODEL_PATH = Path.home() / OX_MODELS / ONNX_MODELS / MODEL_HEAD
    MODEL_PATH_absolute = MODEL_PATH / "model.onnx"
    MODEL_HASH = "d38fca9380728a0f418833b40a3ec031cbd8eaee7650c53f9b6a9eef0f442028"
    MODEL_FILES = [
            "model.onnx",
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ]

    def __init__(self):
        self.max_context_length = 256
        if check_model_existence(model_path=self.MODEL_PATH,model_files=self.MODEL_FILES):
            #from ox_onnx.extract import Extractor
            extract = importlib.import_module("ox_onnx.extract")
            extract.Extractor.model_download(model_id=self.MODEL_ID,model_files=self.MODEL_FILES)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(InterfaceModel, cls).__new__(cls, *args, **kwargs)
            # Initialize your model here
        return cls._instance
    

    @cached_property
    def tokenizer(self):  
        
        tokenizer = Tokenizer.from_file(str(self.MODEL_PATH/"tokenizer.json"))
        tokenizer.enable_truncation(max_length=self.max_context_length)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
        
        return tokenizer

    @cached_property
    def model(self):

        return ort.InferenceSession(self.MODEL_PATH/"model.onnx")

    def generate(self, documents: List[str],batch_size: int = 32):

        return self.forward(documents=documents,batch_size = batch_size).tolist()
    
    def context_length(self)->int:

        return self.max_context_length

    def encode(self, data: str) -> List[int]:

        return self.tokenizer.encode(data).ids

    def decode(self, encoded_data: List[int]) -> str:

        return self.tokenizer.decode(encoded_data)

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v, axis=1)
        norm[norm == 0] = 1e-12
        return v / norm[:, np.newaxis]






