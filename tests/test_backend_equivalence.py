import numpy as np
import pytest
import torch

from semantic_id.algorithms.rq_kmeans import RQKMeans


def test_backend_equivalence_encode():
    """
    Verify that CPU (Numpy) and GPU (Torch) backends produce identical codes
    when using identical codebooks.
    """
    N, D = 100, 8
    X = np.random.randn(N, D).astype(np.float32)

    # 1. Setup a model with random codebooks
    n_levels = 2
    n_clusters = [10, 10]

    # Create manual codebooks (random)
    codebooks = []
    for _ in range(n_levels):
        codebooks.append(np.random.randn(10, D).astype(np.float32))

    # 2. CPU Model
    model_cpu = RQKMeans(n_levels=n_levels, n_clusters=n_clusters)
    # Manually inject codebooks (simulating a fitted model)
    model_cpu.codebooks_ = codebooks
    model_cpu.D_ = D

    codes_cpu = model_cpu.encode(X, device="cpu")

    # 3. Torch Model (run on CPU via RQKMeansTorch direct instantiation)
    from semantic_id.algorithms.rq_kmeans_torch import RQKMeansTorch

    # Torch backend running on CPU
    model_torch = RQKMeansTorch(
        n_levels=n_levels,
        n_clusters=n_clusters,
        metric="l2",
        implementation="kmeans",
        max_iter=10,
        tol=1e-4,
        random_state=42,
        verbose=False,
        device="cpu",
    )
    # Inject codebooks (convert to tensor)
    model_torch.codebooks_ = [torch.from_numpy(cb) for cb in codebooks]
    model_torch.D_ = D

    codes_torch = model_torch.encode(X)  # Returns numpy array

    # 4. Compare
    np.testing.assert_array_equal(codes_cpu, codes_torch)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() and not torch.cuda.is_available(),
    reason="No GPU available",
)
def test_backend_equivalence_gpu():
    """
    Same as above but actually using the GPU device.
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda"

    N, D = 100, 8
    X = np.random.randn(N, D).astype(np.float32)

    n_levels = 2
    n_clusters = [10, 10]

    codebooks = [np.random.randn(10, D).astype(np.float32) for _ in range(n_levels)]

    # CPU
    model_cpu = RQKMeans(n_levels=n_levels, n_clusters=n_clusters)
    model_cpu.codebooks_ = codebooks
    model_cpu.D_ = D
    codes_cpu = model_cpu.encode(X, device="cpu")

    # GPU (via public API since device != cpu)
    model_gpu = RQKMeans(n_levels=n_levels, n_clusters=n_clusters)
    model_gpu.codebooks_ = codebooks  # Public API uses numpy codebooks as master state
    model_gpu.D_ = D

    # encode() with device="cuda"/"mps" should trigger _encode_torch
    codes_gpu = model_gpu.encode(X, device=device)

    np.testing.assert_array_equal(codes_cpu, codes_gpu)
