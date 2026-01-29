from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
import numpy as np

ArrayLike = Union[np.ndarray, List[List[float]]]

class BaseSemanticEncoder(ABC):
    """
    Abstract base class for all Semantic ID encoders.
    """

    @abstractmethod
    def fit(self, X: ArrayLike, *, device: str = "cpu") -> "BaseSemanticEncoder":
        """
        Train the encoder on embeddings X.

        Args:
            X: Input embeddings of shape (N, D).
            device: Calculation device ('cpu', 'cuda', 'mps').

        Returns:
            self: The fitted encoder instance.
        """
        pass

    @abstractmethod
    def encode(self, X: ArrayLike, *, device: str = "cpu", batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode embeddings X into discrete codes.

        Args:
            X: Input embeddings of shape (N, D).
            device: Calculation device.
            batch_size: Batch size for processing large datasets.

        Returns:
            codes: Discrete codes of shape (N, L) with dtype int32.
        """
        pass

    @abstractmethod
    def semantic_id(self, codes: np.ndarray, *, sep: str = "-") -> List[str]:
        """
        Convert discrete codes into string semantic IDs.

        Args:
            codes: Discrete codes of shape (N, L).
            sep: Separator string.

        Returns:
            List of semantic ID strings.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model artifacts (metadata, codebooks, etc.) to the specified path.

        Args:
            path: Directory path to save artifacts.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, *, device: str = "cpu") -> "BaseSemanticEncoder":
        """
        Load the model artifacts from the specified path.

        Args:
            path: Directory path where artifacts are saved.
            device: Device to load the model onto.

        Returns:
            Loaded encoder instance.
        """
        pass

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Approximates the original vectors from codes.
        Optional method.

        Args:
            codes: Discrete codes of shape (N, L).

        Returns:
            vectors_approx: Reconstructed vectors of shape (N, D).
        """
        raise NotImplementedError("Decode method is not implemented for this encoder.")

    def score(self, X: ArrayLike) -> Dict[str, float]:
        """
        Calculate internal quantization metrics.
        Optional method.

        Args:
            X: Input embeddings.

        Returns:
            Dictionary of metrics.
        """
        raise NotImplementedError("Score method is not implemented for this encoder.")
