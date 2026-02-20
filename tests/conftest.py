"""Shared fixtures for semantic-id test suite."""

import numpy as np
import pytest

from semantic_id import RQVAE, RQKMeans, SemanticIdEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N = 50
DEFAULT_D = 8
DEFAULT_N_LEVELS = 2
DEFAULT_N_CLUSTERS = 5
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def random_data() -> np.ndarray:
    """(50, 8) random float32 embeddings with fixed seed."""
    rng = np.random.RandomState(DEFAULT_SEED)
    return rng.randn(DEFAULT_N, DEFAULT_D).astype(np.float32)


@pytest.fixture()
def clustered_data() -> tuple[np.ndarray, np.ndarray]:
    """Six tight clusters far apart with matching codes."""
    rng = np.random.RandomState(0)
    clusters = []
    code_rows = []
    for i in range(6):
        center = np.zeros(8, dtype=np.float32)
        center[i % 8] = 20.0 * (i + 1)
        cluster = rng.randn(10, 8).astype(np.float32) * 0.01 + center
        clusters.append(cluster)
        code_rows.append(np.full((10, 2), [i // 3, i % 3], dtype=np.int32))
    X = np.vstack(clusters)
    codes = np.vstack(code_rows)
    return X, codes


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fitted_rq_kmeans(random_data: np.ndarray) -> RQKMeans:
    """RQKMeans(n_levels=2, n_clusters=5) fitted on random_data."""
    model = RQKMeans(
        n_levels=DEFAULT_N_LEVELS,
        n_clusters=DEFAULT_N_CLUSTERS,
        random_state=DEFAULT_SEED,
    )
    model.fit(random_data)
    return model


@pytest.fixture()
def fitted_rq_vae() -> RQVAE:
    """RQVAE fitted on small random data for fast tests."""
    X = np.random.RandomState(DEFAULT_SEED).randn(50, 16).astype(np.float32)
    model = RQVAE(
        in_dim=16,
        num_emb_list=[8, 8],
        e_dim=8,
        layers=[16],
        batch_size=32,
        epochs=2,
        device="cpu",
    )
    model.fit(X)
    return model


@pytest.fixture()
def fitted_engine(random_data: np.ndarray) -> SemanticIdEngine:
    """SemanticIdEngine with RQKMeans encoder, fitted on random_data."""
    encoder = RQKMeans(
        n_levels=DEFAULT_N_LEVELS,
        n_clusters=DEFAULT_N_CLUSTERS,
        random_state=DEFAULT_SEED,
    )
    engine = SemanticIdEngine(encoder=encoder)
    engine.fit(random_data)
    return engine
