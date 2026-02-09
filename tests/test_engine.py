"""Tests for SemanticIdEngine save/load functionality."""

import numpy as np

from semantic_id.algorithms.rq_kmeans import RQKMeans
from semantic_id.engine import SemanticIdEngine
from semantic_id.uniqueness.resolver import UniqueIdResolver
from semantic_id.uniqueness.stores import SQLiteCollisionStore


def test_engine_basic_flow():
    """Test the basic fit -> unique_ids flow."""
    N, D = 50, 8
    X = np.random.randn(N, D).astype(np.float32)

    encoder = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
    engine = SemanticIdEngine(encoder=encoder)

    engine.fit(X)
    uids = engine.unique_ids(X)

    assert len(uids) == N
    assert all(isinstance(uid, str) for uid in uids)


def test_engine_save_load_in_memory(tmp_path):
    """Test save/load with InMemoryCollisionStore."""
    N, D = 50, 8
    X = np.random.randn(N, D).astype(np.float32)

    encoder = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
    engine = SemanticIdEngine(encoder=encoder)
    engine.fit(X)

    engine.unique_ids(X)

    save_path = str(tmp_path / "engine_test")
    engine.save(save_path)

    loaded_engine = SemanticIdEngine.load(save_path)
    uids_after = loaded_engine.unique_ids(X)

    # Loaded engine has fresh in-memory store, so same IDs get same first-occurrence treatment
    assert len(uids_after) == N
    # The raw semantic IDs should match (same codebooks)
    codes_before = encoder.encode(X)
    codes_after = loaded_engine.encoder.encode(X)
    np.testing.assert_array_equal(codes_before, codes_after)


def test_engine_save_load_sqlite(tmp_path):
    """Test save/load with SQLiteCollisionStore."""
    N, D = 50, 8
    X = np.random.randn(N, D).astype(np.float32)

    db_path = str(tmp_path / "test_collisions.db")
    store = SQLiteCollisionStore(db_path=db_path)
    resolver = UniqueIdResolver(store=store)
    encoder = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
    engine = SemanticIdEngine(encoder=encoder, unique_resolver=resolver)
    engine.fit(X)

    # Generate some IDs to populate the store
    engine.unique_ids(X)

    save_path = str(tmp_path / "engine_sqlite")
    engine.save(save_path)

    loaded_engine = SemanticIdEngine.load(save_path)

    # Should have SQLite store with the previous collision counts
    assert isinstance(loaded_engine.unique_resolver.store, SQLiteCollisionStore)


def test_engine_meta_file_contents(tmp_path):
    """Verify engine_meta.json contains expected fields."""
    import json

    encoder = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
    engine = SemanticIdEngine(encoder=encoder)

    X = np.random.randn(20, 8).astype(np.float32)
    engine.fit(X)

    save_path = str(tmp_path / "engine_meta_test")
    engine.save(save_path)

    import os

    meta_path = os.path.join(save_path, "engine_meta.json")
    assert os.path.exists(meta_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    assert meta["encoder_type"] == "RQKMeans"
    assert meta["store_type"] == "in_memory"
