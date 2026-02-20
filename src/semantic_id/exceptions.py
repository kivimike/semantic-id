"""Custom exception classes for semantic-id."""


class SemanticIdError(Exception):
    """Base exception for all semantic-id errors."""


class NotFittedError(SemanticIdError):
    """Raised when encode/decode is called before fit."""


class ShapeMismatchError(SemanticIdError, ValueError):
    """Raised when input arrays have unexpected shapes."""
