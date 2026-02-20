import string
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np

from semantic_id.exceptions import ShapeMismatchError

ArrayLike = Union[np.ndarray, List[List[float]]]


def _validate_embeddings(
    X: ArrayLike, expected_dim: Optional[int] = None
) -> np.ndarray:
    """Convert *X* to float32 ndarray and validate shape/values."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ShapeMismatchError(
            f"Expected 2-D input (N, D), got {arr.ndim}-D array with shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ShapeMismatchError("Input must contain at least one sample (N > 0)")
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ShapeMismatchError(f"Expected D={expected_dim}, got D={arr.shape[1]}")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Input contains NaN or Inf values")
    return arr


# Letters used for token-format IDs: a, b, c, d, e, ...
_TOKEN_LETTERS = string.ascii_lowercase


def codes_to_ids(
    codes: np.ndarray,
    *,
    sep: str = "-",
    fmt: Literal["plain", "token"] = "plain",
    formatter: Optional[Callable[[np.ndarray], str]] = None,
) -> List[str]:
    """
    Convert a (N, L) array of integer codes to string IDs.

    Args:
        codes: Discrete codes of shape (N, L).
        sep: Separator string (only used for ``fmt="plain"``).
        fmt: Output format.

            - ``"plain"``: ``"3-9-1"`` (codes joined by *sep*)
            - ``"token"``: ``"<a_3><b_9><c_1>"`` (LLM-friendly token format)

        formatter: Optional callable that receives a single code row
            ``(L,)`` and returns a string.  When provided, *sep* and
            *fmt* are ignored.

    Returns:
        List of semantic ID strings.
    """
    N, L = codes.shape
    result: List[str] = []

    if formatter is not None:
        for i in range(N):
            result.append(formatter(codes[i]))
    elif fmt == "token":
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
        formatter: Optional[Callable[[np.ndarray], str]] = None,
    ) -> List[str]:
        """
        Convert discrete codes into string semantic IDs.

        Args:
            codes: Discrete codes of shape (N, L).
            sep: Separator string (used when ``fmt="plain"``).
            fmt: Output format — ``"plain"`` for ``"3-9-1"`` or
                ``"token"`` for ``"<a_3><b_9><c_1>"``.
            formatter: Optional callable that receives a single code row
                ``(L,)`` and returns a string.  Overrides *sep* and *fmt*.

        Returns:
            List of semantic ID strings.
        """
        return codes_to_ids(codes, sep=sep, fmt=fmt, formatter=formatter)

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
