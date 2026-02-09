"""Tests for SinkhornResolver collision resolution."""
import numpy as np
import pytest
import torch

from semantic_id.algorithms.rq_vae import RQVAE
from semantic_id.engine import SemanticIdEngine
from semantic_id.uniqueness.resolver import SinkhornResolver, UniqueIdResolver


def test_sinkhorn_resolver_requires_kwargs():
    """SinkhornResolver.assign should raise if embeddings/model not provided."""
    resolver = SinkhornResolver()
    with pytest.raises(ValueError, match="requires 'embeddings' and 'model'"):
        resolver.assign(["1-2-3", "1-2-3"])


def test_sinkhorn_resolver_with_rqvae():
    """Test SinkhornResolver with an RQVAE model through the engine."""
    N, D = 100, 16
    X = np.random.randn(N, D).astype(np.float32)

    encoder = RQVAE(
        in_dim=D,
        num_emb_list=[8, 8],
        e_dim=8,
        layers=[16],
        batch_size=32,
        epochs=3,
        lr=1e-3,
        device="cpu",
    )

    resolver = SinkhornResolver(max_iterations=5, sk_epsilon=0.01, fallback_suffix=True)
    engine = SemanticIdEngine(encoder=encoder, unique_resolver=resolver)

    engine.fit(X)
    uids = engine.unique_ids(X)

    assert len(uids) == N
    # All IDs should be strings
    assert all(isinstance(uid, str) for uid in uids)
    # With fallback_suffix=True, all should be unique
    assert len(set(uids)) == N


def test_sinkhorn_resolver_find_collision_groups():
    """Test the static collision group finder."""
    ids = np.array(["a", "b", "a", "c", "b", "d"])
    groups = SinkhornResolver._find_collision_groups(ids)

    # Should find 2 groups: indices of "a" and indices of "b"
    group_sets = [set(g) for g in groups]
    assert {0, 2} in group_sets
    assert {1, 4} in group_sets
    assert len(groups) == 2


def test_sinkhorn_resolver_no_collisions():
    """If no collisions exist, resolver should return same IDs."""
    ids = np.array(["a", "b", "c"])
    groups = SinkhornResolver._find_collision_groups(ids)
    assert len(groups) == 0
