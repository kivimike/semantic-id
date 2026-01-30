import numpy as np
import torch
from typing import Optional, Union, List, Literal
from sklearn.cluster import KMeans

from semantic_id.utils.clustering import (
    kmeans_torch, 
    sinkhorn_algorithm, 
    center_distance_for_constraint
)

# For constrained clustering fallback
try:
    from k_means_constrained import KMeansConstrained
    HAS_CONSTRAINED = True
except ImportError:
    HAS_CONSTRAINED = False

class RQKMeansTorch:
    """
    PyTorch-based backend for RQ-KMeans.
    Provides GPU acceleration for encode() and standard K-Means fit().
    For Constrained K-Means fit(), it currently falls back to CPU implementation,
    but moves centroids to GPU for fast inference.
    """
    
    def __init__(
        self,
        n_levels: int,
        n_clusters: List[int],
        metric: Literal["l2", "cosine"],
        implementation: Literal["kmeans", "constrained"],
        max_iter: int,
        tol: float,
        random_state: Optional[int],
        verbose: bool,
        device: str,
        n_init: Optional[int] = None
    ):
        self.n_levels = n_levels
        self.n_clusters = n_clusters
        self.metric = metric
        self.implementation = implementation
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.device = device
        self.n_init = n_init
        
        # We store codebooks as a List of torch Tensors on the target device
        self.codebooks_: List[torch.Tensor] = []
        self.D_ = None

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "RQKMeansTorch":
        # Ensure X is a tensor on the correct device
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.device, dtype=torch.float32)
        else:
            X = X.to(self.device, dtype=torch.float32)
            
        N, D = X.shape
        self.D_ = D
        self.codebooks_ = []
        
        # Residuals start as X
        residuals = X.clone()
        
        for l in range(self.n_levels):
            K = self.n_clusters[l]
            if self.verbose:
                print(f"Training level {l+1}/{self.n_levels} (K={K}) on {self.device}...")
            
            # Seed handling
            seed = self.random_state + l if self.random_state is not None else None
            
            centers = None
            labels = None
            
            if self.implementation == "constrained":
                # Attempt to use GPU-based Sinkhorn Constrained K-Means first
                try:
                    centers, labels = self._constrained_kmeans_torch(residuals, K, seed)
                except Exception as e:
                    if self.verbose:
                        print(f"GPU Constrained K-Means failed ({e}), falling back to CPU...")
                    
                    # FALLBACK to CPU for Constrained Fit
                    # k-means-constrained library is CPU only.
                    # Move residuals to CPU numpy
                    residuals_cpu = residuals.cpu().numpy()
                    
                    min_size = max(1, N // K - 1)
                    max_size = N // K + 1
                    
                    if not HAS_CONSTRAINED:
                        raise ImportError("k-means-constrained is required for implementation='constrained'")
                    
                    # Determine n_init (same logic as RQKMeans)
                    if self.n_init is not None:
                         current_n_init = self.n_init
                    else:
                         current_n_init = 3 # Constrained default
                    
                    kmeans = KMeansConstrained(
                        n_clusters=K,
                        size_min=min_size,
                        size_max=max_size,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        random_state=seed,
                        n_init=current_n_init,
                        n_jobs=-1
                    )
                    kmeans.fit(residuals_cpu)
                    
                    # Convert results back to Torch/GPU
                    centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device, dtype=torch.float32)
                    labels = torch.from_numpy(kmeans.labels_).to(self.device, dtype=torch.long)
                
            else:
                # Standard K-Means on GPU
                centers, labels = self._kmeans_torch(residuals, K, seed)
            
            self.codebooks_.append(centers)
            
            # Update residuals
            # residuals -= centers[labels]
            # Gather centers using labels
            selected_centers = centers[labels]
            residuals = residuals - selected_centers
            
        return self

    def encode(self, X: Union[np.ndarray, torch.Tensor], batch_size: Optional[int] = None) -> np.ndarray:
        if not self.codebooks_:
            raise RuntimeError("Model is not fitted yet.")
            
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.device, dtype=torch.float32)
        else:
            X = X.to(self.device, dtype=torch.float32)
            
        N = X.shape[0]
        if batch_size is None:
            batch_size = N
            
        # Output codes (on CPU numpy at the end)
        # We collect them on CPU to avoid OOM for large N
        codes = np.zeros((N, self.n_levels), dtype=np.int32)
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_X = X[start_idx:end_idx] # (B, D)
            
            residuals = batch_X.clone() # We modify residuals
            
            for l in range(self.n_levels):
                codebook = self.codebooks_[l] # (K, D) on device
                
                # Compute distances (B, K)
                # dist = ||x - c||^2
                # torch.cdist computes p-norm. squared euclidean needs manual or **2
                # For K-Means, we just need argmin, so squared vs not-squared doesn't change order.
                # cdist is efficient.
                dists = torch.cdist(residuals, codebook, p=2.0) # (B, K)
                
                # Argmin
                batch_codes = torch.argmin(dists, dim=1) # (B,)
                
                # Store codes (move to CPU)
                codes[start_idx:end_idx, l] = batch_codes.cpu().numpy()
                
                # Update residuals
                # selected_centers = codebook[batch_codes]
                # residuals -= selected_centers
                # We can do this in place
                residuals -= codebook[batch_codes]
        
        return codes

    def _kmeans_torch(self, X: torch.Tensor, K: int, seed: Optional[int]):
        """
        Simple K-Means implementation in PyTorch using shared utils.
        """
        centers = kmeans_torch(
            X, 
            num_clusters=K, 
            max_iter=self.max_iter, 
            tol=self.tol, 
            seed=seed
        )
        
        # Compute labels for residual update
        dists = torch.cdist(X, centers, p=2.0)
        labels = torch.argmin(dists, dim=1)
        
        return centers, labels

    def _constrained_kmeans_torch(self, X: torch.Tensor, K: int, seed: Optional[int]):
        """
        Constrained K-Means using Sinkhorn-Knopp algorithm for balanced assignment.
        This runs entirely on GPU.
        """
        N, D = X.shape
        
        if seed is not None:
            torch.manual_seed(seed)
            
        # Initialize centers randomly
        indices = torch.randperm(N, device=self.device)[:K]
        centers = X[indices].clone()
        
        for i in range(self.max_iter):
            # E-step: Assign labels with Sinkhorn balancing
            dists = torch.cdist(X, centers, p=2.0) # (N, K)
            dists_sq = dists ** 2
            
            d_centered = center_distance_for_constraint(dists_sq)
            # Epsilon and iters are hyperparameters. 0.1 and 30 are reasonable starts.
            Q = sinkhorn_algorithm(d_centered, epsilon=0.1, sinkhorn_iterations=30) 
            
            # Hard assignment from Q
            labels = torch.argmax(Q, dim=1)
            
            # M-step: Update centers (same as standard KMeans)
            new_centers = torch.zeros_like(centers)
            for k in range(K):
                mask = (labels == k)
                if mask.any():
                    new_centers[k] = X[mask].mean(dim=0)
                else:
                    new_centers[k] = centers[k] # Keep old if empty (rare with Sinkhorn)
            
            shift = torch.norm(centers - new_centers)
            centers = new_centers
            
            if shift < self.tol:
                break
                
        return centers, labels
