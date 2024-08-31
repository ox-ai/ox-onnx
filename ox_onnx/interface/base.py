from abc import ABC, abstractmethod
from typing import List


class BaseInterfaceModel(ABC):
    """Base interface that all models should implement."""

    @abstractmethod
    def tokenizer(self):
        """
        tokenizer form tokenizers.Tokenizer()
        """
        pass

    @abstractmethod
    def generate(self, documents: List[str], batch_size: int = 32):
        """
        Generate embeddings in baths of 32  form input array

        Args:
            data (List[str]): A list of strings to be processed by the model.
            batch_size (int): default 32

        Returns:
            List[List[int]]: A list of vector embeddings, each represented as 256 np.float32 embeddings.
        """

        pass

    @abstractmethod
    def context_length(self) -> int:
        """
        returns context_length of the model
        """
        pass

    @abstractmethod
    def encode(self, data: str) -> List[int]:
        """
        Tokenize and encode a string into a list of token IDs.

        Args:
            data (str): The string to be tokenized and encoded.

        Returns:
            List[int]: A list of token IDs representing the encoded string.
        """
        pass

    @abstractmethod
    def decode(self, encoded_data: List[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Args:
            encoded_data (List[int]): A list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        pass

    @abstractmethod
    def forward(self, documents: List[str], batch_size: int = 32):
        """
        Generate embeddings in baths of 32  form input array

        Args:
            data (List[str]): A list of strings to be processed by the model.
            batch_size (int): default 32

        Returns:
            List[List[int]]: A list of vector embeddings, each represented as 256 np.float32 embeddings.
        """
        pass
