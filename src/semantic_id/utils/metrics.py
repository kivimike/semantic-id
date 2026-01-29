import numpy as np
from typing import Dict, Any

from semantic_id.core import BaseSemanticEncoder

def evaluate_quality(
    encoder: BaseSemanticEncoder, 
    X: np.ndarray, 
    codes: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate the quality of the semantic IDs.
    
    Metrics:
    - collision_rate: 1 - (unique_sids / total_samples)
    - quantization_error: MSE between X and decoded(codes) (if supported)
    """
    results = {}
    N = X.shape[0]
    
    # 1. Collision Rate
    sids = encoder.semantic_id(codes)
    n_unique = len(set(sids))
    collision_rate = 1.0 - (n_unique / N)
    
    results["n_samples"] = N
    results["n_unique_semantic"] = n_unique
    results["collision_rate"] = collision_rate
    
    # 2. Quantization Error
    try:
        X_hat = encoder.decode(codes)
        mse = np.mean((X - X_hat) ** 2)
        results["quantization_mse"] = float(mse)
    except NotImplementedError:
        pass
        
    return results
