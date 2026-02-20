from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    from scipy.stats import spearmanr

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from semantic_id.core import BaseSemanticEncoder


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------


def hierarchical_distance(
    codes_a: np.ndarray, codes_b: np.ndarray
) -> np.ndarray:
    """
    Vectorized hierarchical (prefix-based) distance between code vectors.

    Unlike Hamming distance, this respects the tree structure of Semantic IDs:
    the distance equals ``L - common_prefix_length`` where *L* is the number
    of levels.  If the first level already differs the distance is maximal,
    regardless of whether later levels happen to match.

    Args:
        codes_a: Codes of shape ``(..., L)``.
        codes_b: Codes of shape ``(..., L)`` (broadcastable with *codes_a*).

    Returns:
        Integer distances with the same shape as the broadcast of
        ``codes_a[..., 0]`` and ``codes_b[..., 0]``.
    """
    L = codes_a.shape[-1]
    match = codes_a == codes_b
    cum_match = np.cumprod(match, axis=-1)
    prefix_len = np.sum(cum_match, axis=-1)
    return L - prefix_len


def _hierarchical_metric(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar version of :func:`hierarchical_distance` for sklearn metrics."""
    for i in range(len(a)):
        if a[i] != b[i]:
            return float(len(a) - i)
    return 0.0


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

_CODE_METRICS = ("hierarchical", "hamming")


def recall_at_k(
    X: np.ndarray,
    codes: np.ndarray,
    k: int = 10,
    sample_size: int = 1000,
    metric: Literal["hierarchical", "hamming"] = "hierarchical",
) -> float:
    """
    Calculate Recall@K comparing exact search in X vs code search.

    Args:
        X: Original embeddings (N, D)
        codes: Semantic codes (N, L)
        k: Number of neighbors to check
        sample_size: Number of query items to sample for evaluation
        metric: Distance metric on codes — ``"hierarchical"`` (default)
            respects the tree structure of Semantic IDs, ``"hamming"``
            treats all levels equally.

    Returns:
        Average Recall@K score (0.0 to 1.0)
    """
    N = X.shape[0]
    n_queries = min(N, sample_size)

    rng = np.random.RandomState(42)
    query_indices = rng.choice(N, n_queries, replace=False)

    X_query = X[query_indices]
    codes_query = codes[query_indices]

    # 1. Ground Truth (Euclidean on X)
    nn_x = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn_x.fit(X)
    _, indices_x = nn_x.kneighbors(X_query)

    # 2. Semantic Search
    if metric == "hamming":
        nn_c = NearestNeighbors(n_neighbors=k + 1, metric="hamming", n_jobs=-1)
    else:
        nn_c = NearestNeighbors(
            n_neighbors=k + 1,
            metric=_hierarchical_metric,
            algorithm="brute",
            n_jobs=-1,
        )
    nn_c.fit(codes)
    _, indices_c = nn_c.kneighbors(codes_query)

    recalls = []
    for i in range(n_queries):
        true_neighbors = set(indices_x[i][1:])
        pred_neighbors = set(indices_c[i][1:])

        intersection = len(true_neighbors.intersection(pred_neighbors))
        recall = intersection / k
        recalls.append(recall)

    return float(np.mean(recalls))


def distance_correlation(
    X: np.ndarray,
    codes: np.ndarray,
    sample_size: int = 1000,
    n_pairs: int = 2000,
    seed: int = 42,
    metric: Literal["hierarchical", "hamming"] = "hierarchical",
) -> float:
    """
    Calculate Spearman correlation between Euclidean distances in X
    and code distances.

    Args:
        X: Original embeddings (N, D)
        codes: Semantic codes (N, L)
        sample_size: Unused (legacy parameter kept for API compatibility)
        n_pairs: Number of random pairs to sample for correlation
        seed: Random seed for reproducibility
        metric: Distance metric on codes — ``"hierarchical"`` (default)
            or ``"hamming"``.
    """
    if not HAS_SCIPY:
        return 0.0

    N = X.shape[0]

    rng = np.random.RandomState(seed)

    idx1 = rng.randint(0, N, n_pairs)
    idx2 = rng.randint(0, N, n_pairs)

    d_x = np.linalg.norm(X[idx1] - X[idx2], axis=1)

    if metric == "hamming":
        d_c = np.sum(codes[idx1] != codes[idx2], axis=1).astype(float)
    else:
        d_c = hierarchical_distance(codes[idx1], codes[idx2]).astype(float)

    corr, _ = spearmanr(d_x, d_c)
    return float(corr)


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------


def find_similar(
    codes: np.ndarray,
    query: Union[int, np.ndarray],
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the *k* most similar items by hierarchical Semantic ID distance.

    Args:
        codes: All codes of shape ``(N, L)``.
        query: Either an **integer index** into *codes*, or a code vector
            of shape ``(L,)``.
        k: Number of similar items to return.

    Returns:
        A ``(indices, distances)`` tuple where both arrays have shape
        ``(k,)`` and are sorted by distance in ascending order.
        When *query* is an index, the query item itself is excluded
        from the results.
    """
    if isinstance(query, (int, np.integer)):
        query_idx = int(query)
        query_code = codes[query_idx]
    else:
        query_idx = None
        query_code = np.asarray(query)

    dists = hierarchical_distance(query_code[np.newaxis, :], codes).ravel()

    if query_idx is not None:
        dists[query_idx] = np.iinfo(dists.dtype).max
        max_k = min(k, len(codes) - 1)
    else:
        max_k = min(k, len(codes))

    order = np.argsort(dists)
    top_k = order[:max_k]
    return top_k, dists[top_k]


def evaluate(
    X: np.ndarray,
    codes: np.ndarray,
    encoder: Optional[BaseSemanticEncoder] = None,
    k: int = 10,
    n_pairs: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate the quality of the semantic IDs.

    Args:
        X: Input embeddings (N, D)
        codes: Discrete codes (N, L)
        encoder: Optional encoder instance for decoding/sids
        k: K for Recall@K metric
        n_pairs: Number of pairs for distance correlation
        seed: Random seed

    Returns:
        Dictionary of metrics including:
        - collision_rate
        - recall_at_{k}
        - distance_correlation
        - quantization_mse (if encoder provided)
    """
    results = {}
    N = X.shape[0]

    # 1. Collision Rate (using unique rows of codes)
    # Convert codes to bytes to hash rows for uniqueness check
    codes_view = np.ascontiguousarray(codes).view(
        np.dtype((np.void, codes.dtype.itemsize * codes.shape[1]))
    )
    _, unique_counts = np.unique(codes_view, return_counts=True)
    n_unique = len(unique_counts)

    results["n_samples"] = N
    results["n_unique_codes"] = n_unique
    results["collision_rate"] = 1.0 - (n_unique / N)

    # 2. Retrieval Metrics
    results[f"recall_at_{k}"] = recall_at_k(X, codes, k=k)
    results["distance_correlation"] = distance_correlation(
        X, codes, n_pairs=n_pairs, seed=seed
    )

    # 3. Encoder-dependent metrics
    if encoder is not None:
        try:
            X_hat = encoder.decode(codes)
            mse = np.mean((X - X_hat) ** 2)
            results["quantization_mse"] = float(mse)
        except NotImplementedError:
            pass

    return results


# Alias for backward compatibility
evaluate_quality = evaluate
