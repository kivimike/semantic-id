import numpy as np
from typing import Dict, Any, Optional, List, Union
from sklearn.neighbors import NearestNeighbors
try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from semantic_id.core import BaseSemanticEncoder

def recall_at_k(X: np.ndarray, codes: np.ndarray, k: int = 10, sample_size: int = 1000) -> float:
    """
    Calculate Recall@K comparing exact search in X vs code search.
    
    Args:
        X: Original embeddings (N, D)
        codes: Semantic codes (N, L)
        k: Number of neighbors to check
        sample_size: Number of query items to sample for evaluation
        
    Returns:
        Average Recall@K score (0.0 to 1.0)
    """
    N = X.shape[0]
    n_queries = min(N, sample_size)
    
    # Sample indices for queries
    rng = np.random.RandomState(42)
    query_indices = rng.choice(N, n_queries, replace=False)
    
    X_query = X[query_indices]
    codes_query = codes[query_indices]
    
    # 1. Ground Truth (Euclidean on X)
    # k+1 because the point itself is included as distance 0
    nn_x = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1)
    nn_x.fit(X)
    _, indices_x = nn_x.kneighbors(X_query)
    
    # 2. Semantic Search (Hamming on codes)
    nn_c = NearestNeighbors(n_neighbors=k+1, metric='hamming', n_jobs=-1)
    nn_c.fit(codes)
    _, indices_c = nn_c.kneighbors(codes_query)
    
    # Calculate recall
    recalls = []
    for i in range(n_queries):
        # Remove self (first item)
        true_neighbors = set(indices_x[i][1:])
        pred_neighbors = set(indices_c[i][1:])
        
        intersection = len(true_neighbors.intersection(pred_neighbors))
        recall = intersection / k
        recalls.append(recall)
        
    return float(np.mean(recalls))

def distance_correlation(X: np.ndarray, codes: np.ndarray, sample_size: int = 1000, n_pairs: int = 2000, seed: int = 42) -> float:
    """
    Calculate Spearman correlation between Euclidean distances in X 
    and Hamming distances in codes.
    
    Args:
        X: Original embeddings (N, D)
        codes: Semantic codes (N, L)
        sample_size: Unused in current implementation (legacy)
        n_pairs: Number of random pairs to sample for correlation
        seed: Random seed for reproducibility
    """
    if not HAS_SCIPY:
        return 0.0
        
    N = X.shape[0]
    
    rng = np.random.RandomState(seed)
    
    # Sampling pairs is faster
    idx1 = rng.randint(0, N, n_pairs)
    idx2 = rng.randint(0, N, n_pairs)
    
    d_x = np.linalg.norm(X[idx1] - X[idx2], axis=1)
    d_c = np.sum(codes[idx1] != codes[idx2], axis=1) # Hamming distance
    
    corr, _ = spearmanr(d_x, d_c)
    return float(corr)

def evaluate(
    X: np.ndarray, 
    codes: np.ndarray,
    encoder: Optional[BaseSemanticEncoder] = None,
    k: int = 10,
    n_pairs: int = 2000,
    seed: int = 42
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
    codes_view = np.ascontiguousarray(codes).view(np.dtype((np.void, codes.dtype.itemsize * codes.shape[1])))
    _, unique_counts = np.unique(codes_view, return_counts=True)
    n_unique = len(unique_counts)
    
    results["n_samples"] = N
    results["n_unique_codes"] = n_unique
    results["collision_rate"] = 1.0 - (n_unique / N)
    
    # 2. Retrieval Metrics
    results[f"recall_at_{k}"] = recall_at_k(X, codes, k=k)
    results["distance_correlation"] = distance_correlation(X, codes, n_pairs=n_pairs, seed=seed)
    
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
