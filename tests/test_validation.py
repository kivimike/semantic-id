"""Tests for input validation and custom exceptions."""

import numpy as np
import pytest

from semantic_id import NotFittedError, RQKMeans, ShapeMismatchError
from semantic_id.core import _validate_embeddings


class TestValidateEmbeddings:
    def test_valid_2d(self):
        arr = _validate_embeddings(np.random.randn(10, 4))
        assert arr.dtype == np.float32
        assert arr.shape == (10, 4)

    def test_rejects_1d(self):
        with pytest.raises(ShapeMismatchError, match="2-D"):
            _validate_embeddings(np.array([1.0, 2.0, 3.0]))

    def test_rejects_3d(self):
        with pytest.raises(ShapeMismatchError, match="2-D"):
            _validate_embeddings(np.ones((2, 3, 4)))

    def test_rejects_empty(self):
        with pytest.raises(ShapeMismatchError, match="at least one sample"):
            _validate_embeddings(np.empty((0, 4)))

    def test_rejects_nan(self):
        X = np.array([[1.0, float("nan")]])
        with pytest.raises(ValueError, match="NaN"):
            _validate_embeddings(X)

    def test_rejects_inf(self):
        X = np.array([[1.0, float("inf")]])
        with pytest.raises(ValueError, match="Inf"):
            _validate_embeddings(X)

    def test_checks_expected_dim(self):
        X = np.random.randn(5, 8).astype(np.float32)
        _validate_embeddings(X, expected_dim=8)
        with pytest.raises(ShapeMismatchError, match="D=4"):
            _validate_embeddings(X, expected_dim=4)

    def test_accepts_list_of_lists(self):
        arr = _validate_embeddings([[1.0, 2.0], [3.0, 4.0]])
        assert arr.dtype == np.float32
        assert arr.shape == (2, 2)


class TestRQKMeansValidation:
    def test_fit_rejects_1d(self):
        model = RQKMeans(n_levels=2, n_clusters=5)
        with pytest.raises(ShapeMismatchError):
            model.fit(np.array([1.0, 2.0, 3.0]))

    def test_encode_checks_dim(self):
        X = np.random.randn(20, 8).astype(np.float32)
        model = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
        model.fit(X)
        with pytest.raises(ShapeMismatchError, match="D="):
            model.encode(np.random.randn(5, 4).astype(np.float32))

    def test_encode_before_fit(self):
        model = RQKMeans(n_levels=2, n_clusters=5)
        with pytest.raises(NotFittedError):
            model.encode(np.random.randn(5, 8).astype(np.float32))

    def test_decode_before_fit(self):
        model = RQKMeans(n_levels=2, n_clusters=5)
        with pytest.raises(NotFittedError):
            model.decode(np.zeros((5, 2), dtype=np.int32))


class TestExceptionHierarchy:
    def test_not_fitted_is_semantic_id_error(self):
        from semantic_id.exceptions import SemanticIdError

        assert issubclass(NotFittedError, SemanticIdError)

    def test_shape_mismatch_is_value_error(self):
        assert issubclass(ShapeMismatchError, ValueError)
