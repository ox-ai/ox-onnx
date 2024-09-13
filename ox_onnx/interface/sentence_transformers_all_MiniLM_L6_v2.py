"""
onnx model definition for  "sentence-transformers/all-MiniLM-L6-v2"
"""
# onnx model file from hugging face
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/onnx/model.onnx
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/train_script.py


# runtime for vec onnx model

from functools import cached_property
import importlib
from pathlib import Path
import threading
from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np
from typing import List

from ox_onnx.utils import check_model_existence
from ox_onnx.config import OX_MODELS, ONNX_MODELS

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


# Interface implementation of the default sentence-transformers model using ONNX
class InterfaceModel:

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(InterfaceModel, cls).__new__(
                        cls, *args, **kwargs
                    )
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#:~:text=By%20default%2C%20input%20text%20longer%20than%20256%20word%20pieces%20is%20truncated.
            self.max_context_length = 256

            if not check_model_existence(
                model_path=MODEL_PATH, model_files=MODEL_FILES
            ):
                # from ox_onnx.extract import Extractor
                extract = importlib.import_module("ox_onnx.extract")
                extract.Extractor.model_download(
                    model_id=MODEL_ID, model_files=MODEL_FILES
                )

            self.initialized = True  # Mark as initialized

    @cached_property
    def tokenizer(self):

        tokenizer = Tokenizer.from_file(str(MODEL_PATH / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=self.max_context_length)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)

        return tokenizer

    @cached_property
    def model(self):
        # to suppress onnxruntime warnings.
        # so = self.ort.SessionOptions()
        # so.log_severity_level = 3
        return ort.InferenceSession(MODEL_PATH / "model.onnx")

    def generate(self, documents: List[str], batch_size: int = 32):
        """
        Generate embeddings in baths of 32  form input array

        Args:
            data (List[str]): A list of strings to be processed by the model.
            batch_size (int): default 32

        Returns:
            List[List[int]]: A list of vector embeddings, each represented as 256 np.float32 embeddings.
        """

        return self.forward(documents=documents, batch_size=batch_size).tolist()

    def context_length(self) -> int:
        """
        returns context_length of the model
        """
        return self.max_context_length

    def encode(self, data: str) -> List[int]:
        """
        Tokenize and encode a string into a list of token IDs.

        Args:
            data (str): The string to be tokenized and encoded.

        Returns:
            List[int]: A list of token IDs representing the encoded string.
        """
        return self.tokenizer.encode(data).ids

    def decode(self, encoded_data: List[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Args:
            encoded_data (List[int]): A list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(encoded_data)

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v, axis=1)
        norm[norm == 0] = 1e-12
        return v / norm[:, np.newaxis]

    # borrowed from https://github.com/chroma-core/chroma/blob/main/chromadb%2Futils%2Fembedding_functions%2Fonnx_mini_lm_l6_v2.py#L126
    def forward(self, documents: List[str], batch_size: int = 32):
        """
        Generate embeddings in baths of 32  form input array

        Args:
            data (List[str]): A list of strings to be processed by the model.
            batch_size (int): default 32

        Returns:
            List[List[int]]: A list of vector embeddings, each represented as 256 np.float32 embeddings.
        """
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            encoded = [self.tokenizer.encode(d) for d in batch]
            input_ids = np.array([e.ids for e in encoded])
            attention_mask = np.array([e.attention_mask for e in encoded])
            onnx_input = {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "token_type_ids": np.array(
                    [np.zeros(len(e), dtype=np.int64) for e in input_ids],
                    dtype=np.int64,
                ),
            }
            model_output = self.model.run(None, onnx_input)
            last_hidden_state = model_output[0]
            # Perform mean pooling with attention weighting
            input_mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask, -1), last_hidden_state.shape
            )
            embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(
                input_mask_expanded.sum(1), a_min=1e-9, a_max=None
            )
            embeddings = self.normalize(embeddings).astype(np.float32)
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings)
