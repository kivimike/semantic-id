import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

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
    seed: int = 42,
    metric: Literal["hierarchical", "hamming"] = "hierarchical",
) -> float:
    """
    Calculate Recall@K comparing exact search in X vs code search.

    Args:
        X: Original embeddings (N, D)
        codes: Semantic codes (N, L)
        k: Number of neighbors to check
        sample_size: Number of query items to sample for evaluation
        seed: Random seed for reproducibility
        metric: Distance metric on codes — ``"hierarchical"`` (default)
            respects the tree structure of Semantic IDs, ``"hamming"``
            treats all levels equally.

    Returns:
        Average Recall@K score (0.0 to 1.0)
    """
    N = X.shape[0]
    if k >= N:
        raise ValueError(
            f"k={k} must be less than N={N} (need at least k+1 samples)"
        )
    n_queries = min(N, sample_size)

    rng = np.random.RandomState(seed)
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
        warnings.warn(
            "scipy is not installed — distance_correlation requires scipy. "
            "Install it with `pip install scipy`. Returning 0.0.",
            stacklevel=2,
        )
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


def ndcg_at_k(
    X: np.ndarray,
    codes: np.ndarray,
    k: int = 10,
    sample_size: int = 1000,
    seed: int = 42,
    metric: Literal["hierarchical", "hamming"] = "hierarchical",
) -> float:
    """
    Calculate NDCG@K comparing embedding-space ranking to code-space ranking.

    For each sampled query, the ground-truth relevance of a neighbor is
    defined by its inverse Euclidean distance rank, and the predicted ranking
    comes from code-space nearest-neighbor search.

    Args:
        X: Original embeddings ``(N, D)``.
        codes: Semantic codes ``(N, L)``.
        k: Number of neighbors.
        sample_size: Number of query items to sample.
        seed: Random seed for reproducibility.
        metric: Code distance metric (``"hierarchical"`` or ``"hamming"``).

    Returns:
        Average NDCG@K score (0.0 to 1.0).
    """
    N = X.shape[0]
    if k >= N:
        raise ValueError(
            f"k={k} must be less than N={N} (need at least k+1 samples)"
        )
    n_queries = min(N, sample_size)

    rng = np.random.RandomState(seed)
    query_indices = rng.choice(N, n_queries, replace=False)

    X_query = X[query_indices]
    codes_query = codes[query_indices]

    nn_x = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn_x.fit(X)
    _, indices_x = nn_x.kneighbors(X_query)

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

    discount = 1.0 / np.log2(np.arange(2, k + 2))  # length k

    ndcgs = []
    for i in range(n_queries):
        true_set = list(indices_x[i][1:])  # k true neighbors in order
        pred_set = list(indices_c[i][1:])  # k predicted neighbors in order

        relevance_map = {idx: k - rank for rank, idx in enumerate(true_set)}

        dcg = 0.0
        for rank, idx in enumerate(pred_set):
            rel = relevance_map.get(idx, 0)
            dcg += rel * discount[rank]

        ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
        idcg = sum(r * d for r, d in zip(ideal_rels, discount))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs))


# ---------------------------------------------------------------------------
# Code-space diagnostics
# ---------------------------------------------------------------------------


def code_utilization_per_level(
    codes: np.ndarray,
    n_clusters: Optional[Union[int, List[int]]] = None,
) -> List[float]:
    """
    Fraction of codebook entries actually used at each level.

    Args:
        codes: Discrete codes ``(N, L)`` with integer dtype.
        n_clusters: Number of codebook entries per level.  If a single int,
            the same value is used for all levels.  When ``None`` the
            maximum code value + 1 at each level is used as the codebook
            size.

    Returns:
        List of utilization fractions (0.0 to 1.0), one per level.
    """
    L = codes.shape[1]
    if n_clusters is None:
        n_clusters_list = [int(codes[:, lvl].max()) + 1 for lvl in range(L)]
    elif isinstance(n_clusters, int):
        n_clusters_list = [n_clusters] * L
    else:
        n_clusters_list = list(n_clusters)

    utils = []
    for lvl in range(L):
        n_used = len(np.unique(codes[:, lvl]))
        utils.append(n_used / n_clusters_list[lvl])
    return utils


def code_entropy_per_level(codes: np.ndarray) -> List[float]:
    """
    Shannon entropy of the code distribution at each level (in nats).

    Higher entropy means codes are more uniformly distributed across the
    codebook; lower entropy means a few codes dominate.

    Args:
        codes: Discrete codes ``(N, L)``.

    Returns:
        List of entropy values, one per level.
    """
    L = codes.shape[1]
    entropies = []
    for lvl in range(L):
        _, counts = np.unique(codes[:, lvl], return_counts=True)
        probs = counts / counts.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        entropies.append(entropy)
    return entropies


def collision_rate_per_level(codes: np.ndarray) -> List[float]:
    """
    Collision rate at each prefix depth.

    For depth *d*, two items collide when their first *d* code levels are
    identical.  The collision rate is ``1 - n_unique_prefixes / N``.

    Args:
        codes: Discrete codes ``(N, L)``.

    Returns:
        List of collision rates, one per level (cumulative prefix depth).
    """
    N, L = codes.shape
    rates = []
    for depth in range(1, L + 1):
        prefix = np.ascontiguousarray(codes[:, :depth])
        prefix_view = prefix.view(
            np.dtype((np.void, prefix.dtype.itemsize * depth))
        )
        n_unique = len(np.unique(prefix_view))
        rates.append(1.0 - (n_unique / N))
    return rates


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


def evaluate(
    X: np.ndarray,
    codes: np.ndarray,
    encoder: Optional[BaseSemanticEncoder] = None,
    k: int = 10,
    n_pairs: int = 2000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Evaluate the quality of the semantic IDs.

    Args:
        X: Input embeddings ``(N, D)``.
        codes: Discrete codes ``(N, L)``.
        encoder: Optional encoder instance (enables ``quantization_mse``
            and provides ``n_clusters`` for utilization).
        k: *K* for Recall@K and NDCG@K metrics.
        n_pairs: Number of pairs for distance correlation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary of metrics:

        - ``n_samples`` -- number of input samples.
        - ``n_unique_codes`` -- number of distinct code tuples.
        - ``collision_rate`` -- overall collision rate.
        - ``collision_rate_per_level`` -- collision rate at each prefix depth.
        - ``recall_at_{k}`` -- Recall@K.
        - ``ndcg_at_{k}`` -- NDCG@K.
        - ``distance_correlation`` -- Spearman correlation.
        - ``code_utilization_per_level`` -- codebook utilization per level.
        - ``code_entropy_per_level`` -- Shannon entropy per level.
        - ``quantization_mse`` -- reconstruction error (if encoder supports
          ``decode()``).
    """
    results: Dict[str, object] = {}
    N = X.shape[0]

    # 1. Collision Rate
    codes_view = np.ascontiguousarray(codes).view(
        np.dtype((np.void, codes.dtype.itemsize * codes.shape[1]))
    )
    _, unique_counts = np.unique(codes_view, return_counts=True)
    n_unique = len(unique_counts)

    results["n_samples"] = N
    results["n_unique_codes"] = n_unique
    results["collision_rate"] = 1.0 - (n_unique / N)
    results["collision_rate_per_level"] = collision_rate_per_level(codes)

    # 2. Retrieval Metrics (only when k < N)
    if k < N:
        results[f"recall_at_{k}"] = recall_at_k(X, codes, k=k, seed=seed)
        results[f"ndcg_at_{k}"] = ndcg_at_k(X, codes, k=k, seed=seed)
    results["distance_correlation"] = distance_correlation(
        X, codes, n_pairs=n_pairs, seed=seed
    )

    # 3. Code-space diagnostics
    n_cl = None
    if encoder is not None:
        n_cl = getattr(encoder, "n_clusters", None) or getattr(
            encoder, "num_emb_list", None
        )
    results["code_utilization_per_level"] = code_utilization_per_level(
        codes, n_clusters=n_cl
    )
    results["code_entropy_per_level"] = code_entropy_per_level(codes)

    # 4. Encoder-dependent metrics
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
