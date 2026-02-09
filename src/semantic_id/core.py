import string
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, List[List[float]]]

# Letters used for token-format IDs: a, b, c, d, e, ...
_TOKEN_LETTERS = string.ascii_lowercase


def codes_to_ids(
    codes: np.ndarray,
    *,
    sep: str = "-",
    fmt: Literal["plain", "token"] = "plain",
) -> List[str]:
    """
    Convert a (N, L) array of integer codes to string IDs.

    Args:
        codes: Discrete codes of shape (N, L).
        sep: Separator string (only used for ``fmt="plain"``).
        fmt: Output format.

            - ``"plain"``: ``"3-9-1"`` (codes joined by *sep*)
            - ``"token"``: ``"<a_3><b_9><c_1>"`` (LLM-friendly token format)

    Returns:
        List of semantic ID strings.
    """
    N, L = codes.shape
    result: List[str] = []

    if fmt == "token":
        for i in range(N):
            parts = []
            for level in range(L):
                letter = _TOKEN_LETTERS[level % len(_TOKEN_LETTERS)]
                parts.append(f"<{letter}_{codes[i, level]}>")
            result.append("".join(parts))
    else:
        for i in range(N):
            result.append(sep.join(str(c) for c in codes[i]))

    return result


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
    def encode(
        self, X: ArrayLike, *, device: str = "cpu", batch_size: Optional[int] = None
    ) -> np.ndarray:
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

    def semantic_id(
        self,
        codes: np.ndarray,
        *,
        sep: str = "-",
        fmt: Literal["plain", "token"] = "plain",
    ) -> List[str]:
        """
        Convert discrete codes into string semantic IDs.

        Args:
            codes: Discrete codes of shape (N, L).
            sep: Separator string (used when ``fmt="plain"``).
            fmt: Output format — ``"plain"`` for ``"3-9-1"`` or
                ``"token"`` for ``"<a_3><b_9><c_1>"``.

        Returns:
            List of semantic ID strings.
        """
        return codes_to_ids(codes, sep=sep, fmt=fmt)

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
