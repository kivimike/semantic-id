"""Tests for token-format semantic IDs."""

import numpy as np
import pytest

from semantic_id.algorithms.rq_kmeans import RQKMeans
from semantic_id.core import codes_to_ids


def test_codes_to_ids_plain():
    codes = np.array([[3, 9, 1], [0, 5, 2]], dtype=np.int32)
    result = codes_to_ids(codes, sep="-", fmt="plain")
    assert result == ["3-9-1", "0-5-2"]


def test_codes_to_ids_token():
    codes = np.array([[3, 9, 1], [0, 5, 2]], dtype=np.int32)
    result = codes_to_ids(codes, fmt="token")
    assert result == ["<a_3><b_9><c_1>", "<a_0><b_5><c_2>"]


def test_codes_to_ids_custom_sep():
    codes = np.array([[1, 2, 3]], dtype=np.int32)
    result = codes_to_ids(codes, sep="_", fmt="plain")
    assert result == ["1_2_3"]


def test_semantic_id_plain_via_encoder():
    N, D = 20, 8
    X = np.random.randn(N, D).astype(np.float32)

    model = RQKMeans(n_levels=3, n_clusters=5, random_state=42)
    model.fit(X)
    codes = model.encode(X)

    ids_plain = model.semantic_id(codes, fmt="plain")
    assert all("-" in sid for sid in ids_plain)
    assert len(ids_plain) == N


def test_semantic_id_token_via_encoder():
    N, D = 20, 8
    X = np.random.randn(N, D).astype(np.float32)

    model = RQKMeans(n_levels=3, n_clusters=5, random_state=42)
    model.fit(X)
    codes = model.encode(X)

    ids_token = model.semantic_id(codes, fmt="token")
    assert all(sid.startswith("<a_") for sid in ids_token)
    assert all("><b_" in sid for sid in ids_token)
    assert all("><c_" in sid for sid in ids_token)
    assert len(ids_token) == N


def test_token_format_many_levels():
    """Token format with more levels than alphabet letters should wrap."""
    codes = np.zeros((1, 28), dtype=np.int32)
    result = codes_to_ids(codes, fmt="token")
    # Level 26 wraps to 'a', level 27 wraps to 'b'
    assert "<a_0>" in result[0]  # level 0
    assert "<c_0>" in result[0]  # level 2 (and level 28 wraps)
