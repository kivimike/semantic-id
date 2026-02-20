import numpy as np
import pytest

from semantic_id.algorithms.rq_kmeans import RQKMeans
from semantic_id.exceptions import NotFittedError


def test_encode_before_fit_raises():
    model = RQKMeans(n_levels=2, n_clusters=5)
    X = np.random.randn(10, 8).astype(np.float32)
    with pytest.raises(NotFittedError, match="not fitted"):
        model.encode(X)


def test_decode_before_fit_raises():
    model = RQKMeans(n_levels=2, n_clusters=5)
    codes = np.zeros((5, 2), dtype=np.int32)
    with pytest.raises(NotFittedError, match="not fitted"):
        model.decode(codes)


def test_mismatched_n_clusters_length():
    with pytest.raises(ValueError, match="must match n_levels"):
        RQKMeans(n_levels=3, n_clusters=[5, 10])


def test_cosine_metric_raises():
    with pytest.raises(NotImplementedError, match="Cosine"):
        RQKMeans(n_levels=2, n_clusters=5, metric="cosine")


def test_single_sample():
    X = np.random.randn(1, 8).astype(np.float32)
    model = RQKMeans(n_levels=2, n_clusters=1, random_state=42)
    model.fit(X)
    codes = model.encode(X)
    assert codes.shape == (1, 2)


def test_rq_kmeans_shapes():
    N, D = 100, 16
    X = np.random.randn(N, D)

    model = RQKMeans(n_levels=3, n_clusters=10, random_state=42)
    model.fit(X)

    codes = model.encode(X)
    assert codes.shape == (N, 3)
    assert codes.dtype == np.int32
    assert np.all(codes >= 0)
    assert np.all(codes < 10)


def test_rq_kmeans_determinism():
    N, D = 50, 8
    X = np.random.randn(N, D)

    model1 = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
    model1.fit(X)
    codes1 = model1.encode(X)

    model2 = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
    model2.fit(X)
    codes2 = model2.encode(X)

    np.testing.assert_array_equal(codes1, codes2)


def test_save_load(tmp_path):
    N, D = 50, 8
    X = np.random.randn(N, D)

    model = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
    model.fit(X)
    codes_orig = model.encode(X)

    save_path = tmp_path / "rq_model"
    model.save(str(save_path))

    loaded_model = RQKMeans.load(str(save_path))
    codes_loaded = loaded_model.encode(X)

    np.testing.assert_array_equal(codes_orig, codes_loaded)
    assert loaded_model.n_levels == model.n_levels
    assert loaded_model.n_clusters == model.n_clusters


def test_constrained_kmeans():
    try:
        import k_means_constrained  # noqa: F401
    except ImportError:
        pytest.skip("k-means-constrained not installed")

    N, D = 100, 8
    X = np.random.randn(N, D)

    # K=10, N=100 => perfect balance would be 10 per cluster
    model = RQKMeans(
        n_levels=2, n_clusters=10, implementation="constrained", random_state=42
    )
    model.fit(X)
    codes = model.encode(X)

    # Check level 1 balance
    counts = np.bincount(codes[:, 0], minlength=10)
    # k-means-constrained ensures min_size = N/K - 1 = 9, max_size = N/K + 1 = 11 during FIT.
    # However, encode() uses nearest neighbor which may slightly deviate.
    # We allow a small margin (e.g., +/- 3).
    assert np.all(counts >= 5)
    assert np.all(counts <= 15)


def test_decode_round_trip():
    N, D = 50, 8
    X = np.random.randn(N, D).astype(np.float32)

    model = RQKMeans(n_levels=3, n_clusters=10, random_state=42)
    model.fit(X)

    codes = model.encode(X)
    X_hat = model.decode(codes)

    assert X_hat.shape == X.shape
    assert X_hat.dtype == np.float32
    # Reconstruction should be closer to X than random noise
    mse = np.mean((X - X_hat) ** 2)
    random_mse = np.mean(X**2)
    assert mse < random_mse


def test_variable_clusters():
    N, D = 100, 8
    X = np.random.randn(N, D)

    # Level 1: 5 clusters, Level 2: 10 clusters
    n_clusters = [5, 10]
    model = RQKMeans(n_levels=2, n_clusters=n_clusters, random_state=42)
    model.fit(X)
    codes = model.encode(X)

    assert codes.shape == (N, 2)
    assert np.all(codes[:, 0] < 5)
    assert np.all(codes[:, 1] < 10)

    # Check codebook shapes
    assert len(model.codebooks_) == 2
    assert model.codebooks_[0].shape == (5, D)
    assert model.codebooks_[1].shape == (10, D)
