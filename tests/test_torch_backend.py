import numpy as np
import pytest
import torch

from semantic_id.algorithms.rq_kmeans import RQKMeans
from semantic_id.algorithms.rq_kmeans_torch import RQKMeansTorch


def test_torch_fit_encode_cpu_fallback():
    """Test that torch backend works."""
    N, D = 50, 8
    X = np.random.randn(N, D)

    # We force device='cpu' to RQKMeans, which should trigger the numpy path if we call .fit(X, device='cpu')
    # But if we want to test _fit_torch specifically, we can use 'cpu' as device for it too (since torch runs on cpu).

    # Instantiate RQKMeansTorch directly to verify its logic on CPU
    model = RQKMeansTorch(
        n_levels=2,
        n_clusters=[5, 5],
        metric="l2",
        implementation="kmeans",
        max_iter=10,
        tol=1e-4,
        random_state=42,
        verbose=False,
        device="cpu",
    )

    model.fit(X)
    codes = model.encode(X)

    assert codes.shape == (N, 2)
    assert np.all(codes >= 0)
    assert np.all(codes < 5)

    # Check that codebooks are tensors
    assert len(model.codebooks_) == 2
    assert isinstance(model.codebooks_[0], torch.Tensor)


def test_constrained_kmeans_torch_sinkhorn_logic():
    """Test the GPU/Torch Sinkhorn implementation of constrained K-Means logic directly."""
    N, D = 100, 8
    X = torch.randn(N, D)
    K = 10

    model = RQKMeansTorch(
        n_levels=1,
        n_clusters=[K],
        metric="l2",
        implementation="constrained",
        max_iter=20,
        tol=1e-4,
        random_state=42,
        verbose=False,
        device="cpu",
    )

    # Test internal method _constrained_kmeans_torch
    centers, labels = model._constrained_kmeans_torch(X, K, seed=42)

    counts = torch.bincount(labels, minlength=K)
    # Balanced check (relaxed for small N)
    assert torch.all(counts >= 6)
    assert torch.all(counts <= 14)
    assert centers.shape == (K, D)


def test_constrained_kmeans_torch_integration():
    """Test full fit/encode cycle with constrained implementation via RQKMeansTorch."""
    N, D = 100, 8
    X = np.random.randn(N, D)
    K = 10

    model = RQKMeansTorch(
        n_levels=1,
        n_clusters=[K],
        metric="l2",
        implementation="constrained",
        max_iter=20,
        tol=1e-4,
        random_state=42,
        verbose=False,
        device="cpu",
    )

    model.fit(X)
    codes = model.encode(X)

    counts = np.bincount(codes[:, 0], minlength=K)
    # Sinkhorn-based constrained K-Means produces approximately balanced clusters during fit,
    # but encode() uses nearest-neighbor which can deviate. Use relaxed bounds.
    assert np.all(counts >= 2)
    assert np.all(counts <= 20)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() and not torch.cuda.is_available(),
    reason="No GPU available",
)
def test_gpu_execution_integration():
    """Integration test for RQKMeans with actual GPU if available."""
    device = "mps" if torch.backends.mps.is_available() else "cuda"

    N, D = 100, 16
    X = np.random.randn(N, D)

    model = RQKMeans(n_levels=2, n_clusters=10, random_state=42)

    # This should trigger _fit_torch
    model.fit(X, device=device)

    # This should trigger _encode_torch
    codes = model.encode(X, device=device)

    assert codes.shape == (N, 2)
    assert isinstance(codes, np.ndarray)
