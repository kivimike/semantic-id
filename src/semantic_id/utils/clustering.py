from typing import Optional, Tuple

import numpy as np
import torch


def sinkhorn_algorithm(
    distances: torch.Tensor, epsilon: float, sinkhorn_iterations: int
) -> torch.Tensor:
    """
    Sinkhorn algorithm to balance cluster assignment.

    Args:
        distances: (B, K) Distance matrix (usually squared Euclidean distance).
        epsilon: Regularization parameter.
        sinkhorn_iterations: Number of iterations.

    Returns:
        Q: (B, K) Assignment probability matrix where rows sum to 1 (soft assignment)
           and columns sum to B/K (balanced assignment).
    """
    # Reference: reference/rq/models/layers.py
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0]  # number of samples to assign
    K = Q.shape[1]  # how many centroids per block

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B  # the columns must sum to 1 so that Q is an assignment probability
    return Q


def center_distance_for_constraint(distances: torch.Tensor) -> torch.Tensor:
    """
    Adjust distances to help Sinkhorn stability.
    Centers the distances around the mean and scales by amplitude.

    Args:
        distances: (B, K) Distance matrix.

    Returns:
        centered_distances: (B, K) Adjusted distances.
    """
    # Reference: reference/rq/models/vq.py
    max_distance = distances.max()
    min_distance = distances.min()

    middle = (max_distance + min_distance) / 2
    amplitude = max_distance - middle + 1e-5

    # Ensure amplitude is positive
    # In reference code: assert amplitude > 0

    centered_distances = (distances - middle) / amplitude
    return centered_distances


def _initialize_centroids_kmeans_plus_plus(
    X: torch.Tensor, K: int, seed: Optional[int] = None
) -> torch.Tensor:
    """
    Initialize centroids using k-means++ algorithm on GPU.

    Args:
        X: (N, D) Input data tensor.
        K: Number of clusters.
        seed: Random seed.

    Returns:
        centers: (K, D) Initial centroids.
    """
    N, D = X.shape
    device = X.device

    if seed is not None:
        torch.manual_seed(seed)

    # 1. Choose first center randomly
    first_center_idx = torch.randint(0, N, (1,), device=device).item()

    centers = torch.empty((K, D), device=device, dtype=X.dtype)
    centers[0] = X[first_center_idx]

    # To store nearest squared distance for each point
    closest_dist_sq = torch.full((N,), float("inf"), device=device)

    # 2. Loop to find other K-1 centers
    for i in range(1, K):
        # Distance from the last added center to all points (Squared L2)
        current_center = centers[i - 1].unsqueeze(0)  # (1, D)

        # Calculate squared L2 distance efficiently
        # ||x - c||^2 = sum((x - c)^2)
        dist_sq_to_new_center = torch.sum((X - current_center) ** 2, dim=1)

        # Update closest distance for each point
        closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq_to_new_center)

        # 3. Choose new center with probability proportional to D(x)^2
        # torch.multinomial handles weighting

        if closest_dist_sq.sum() == 0:
            # Fallback if all points are identical to centers
            candidate_idx = torch.randint(0, N, (1,), device=device).item()
        else:
            # Multinomial expects probabilities (or weights)
            candidate_idx = torch.multinomial(closest_dist_sq, 1).item()

        centers[i] = X[candidate_idx]

    return centers


def kmeans_torch(
    X: torch.Tensor,
    num_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    PyTorch implementation of K-Means clustering.
    Aligns with sklearn.cluster.KMeans(init='k-means++') but runs on GPU/Tensor.

    Args:
        X: (N, D) Input data tensor.
        num_clusters: Number of clusters (K).
        max_iter: Maximum iterations.
        tol: Tolerance for convergence.
        seed: Random seed.
        verbose: Whether to print progress.

    Returns:
        centers: (K, D) Final cluster centers.
    """
    N, D = X.shape

    if seed is not None:
        torch.manual_seed(seed)

    # Initialization (k-means++)
    centers = _initialize_centroids_kmeans_plus_plus(X, num_clusters, seed)

    for i in range(max_iter):
        # E-step: Assign labels
        # cdist computes L2 norm (sqrt(sum(diff^2))).
        # For assignment, L2 and Squared L2 give same argmin.
        dists = torch.cdist(X, centers, p=2.0)
        labels = torch.argmin(dists, dim=1)

        # M-step: Update centers
        new_centers = torch.zeros_like(centers)

        # Optimized summation using index_add_ or manual loop
        # Since K is typically small (e.g. 256), a loop is acceptable and robust
        for k in range(num_clusters):
            mask = labels == k
            if mask.any():
                new_centers[k] = X[mask].mean(dim=0)
            else:
                # Handle empty cluster: Keep old center
                # (Alternative: re-initialize, but sklearn keeps old or warns)
                new_centers[k] = centers[k]

        # Check convergence
        shift = torch.norm(centers - new_centers)
        if verbose:
            print(f"K-Means iter {i}: shift={shift.item()}")

        centers = new_centers

        if shift < tol:
            if verbose:
                print(f"K-Means converged at iter {i}")
            break

    return centers
