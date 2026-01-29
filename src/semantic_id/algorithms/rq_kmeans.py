import json
import os
import pickle
from typing import List, Optional, Union, Dict, Any, Literal
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from semantic_id.core import BaseSemanticEncoder, ArrayLike
from semantic_id.algorithms.rq_kmeans_torch import RQKMeansTorch

try:
    from k_means_constrained import KMeansConstrained
    HAS_CONSTRAINED = True
except ImportError:
    HAS_CONSTRAINED = False

class RQKMeans(BaseSemanticEncoder):
    """
    Residual Quantization with K-Means (RQ-KMeans).
    Supports standard K-Means and Constrained K-Means (balanced).
    """

from semantic_id.algorithms.rq_kmeans_torch import RQKMeansTorch

class RQKMeans(BaseSemanticEncoder):
    """
    Residual Quantization with K-Means (RQ-KMeans).
    Supports standard K-Means and Constrained K-Means (balanced).
    Delegates to Numpy or PyTorch backend based on device.
    """

    def __init__(
        self,
        n_levels: int = 4,
        n_clusters: Union[int, List[int]] = 256,
        metric: Literal["l2", "cosine"] = "l2",
        implementation: Literal["kmeans", "constrained"] = "kmeans",
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
        n_init: Optional[int] = None
    ):
        self.n_levels = n_levels
        
        if isinstance(n_clusters, int):
            self.n_clusters = [n_clusters] * n_levels
        else:
            if len(n_clusters) != n_levels:
                raise ValueError(f"len(n_clusters) {len(n_clusters)} must match n_levels {n_levels}")
            self.n_clusters = list(n_clusters)
            
        self.metric = metric
        self.implementation = implementation
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.n_init = n_init

        self.codebooks_: List[np.ndarray] = [] # List of (K_l, D) arrays (Always stored as Numpy on CPU for master state)
        self.D_ = None

        if self.implementation == "constrained" and not HAS_CONSTRAINED:
            raise ImportError(
                "k-means-constrained is required for implementation='constrained'. "
                "Please install it with `pip install k-means-constrained`."
            )

    def fit(self, X: ArrayLike, *, device: str = "cpu") -> "RQKMeans":
        if device == "cpu":
            self._fit_numpy(X)
        else:
            self._fit_torch(X, device)
        return self
        
    def _fit_numpy(self, X: ArrayLike):
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        self.D_ = D
        
        # Prepare codebooks storage
        self.codebooks_ = []
        
        residuals = X.copy()
        
        for l in range(self.n_levels):
            n_clusters_l = self.n_clusters[l]
            
            if self.verbose:
                print(f"Training level {l+1}/{self.n_levels} (K={n_clusters_l}) on CPU...")
            
            # Determine seed for this level
            # Use seed generation consistent with reference implementation
            level_seed = None
            if self.random_state is not None:
                 level_seed = int(np.random.RandomState(self.random_state + l).randint(0, 2**31 - 1))
            
            # Determine n_init
            if self.n_init is not None:
                current_n_init = self.n_init
            else:
                if self.implementation == "constrained":
                    current_n_init = 3 # Constrained default
                else:
                    current_n_init = 10 # Standard default
            
            if self.implementation == "constrained":
                # Calculate min and max cluster size for balanced clustering
                min_size = max(1, N // n_clusters_l - 1)
                max_size = N // n_clusters_l + 1
                
                kmeans = KMeansConstrained(
                    n_clusters=n_clusters_l,
                    size_min=min_size,
                    size_max=max_size,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=level_seed,
                    n_init=current_n_init, # k-means-constrained might not support cosine directly efficiently
                    n_jobs=-1
                )
            else:
                kmeans = KMeans(
                    n_clusters=n_clusters_l,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=level_seed,
                    n_init=current_n_init,
                )
            
            kmeans.fit(residuals)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            self.codebooks_.append(centers)
            
            # Update residuals
            # R_{l+1} = R_l - C_l[codes_l]
            residuals = residuals - centers[labels]

    def _fit_torch(self, X: ArrayLike, device: str):
        # Delegate to RQKMeansTorch
        torch_model = RQKMeansTorch(
            n_levels=self.n_levels,
            n_clusters=self.n_clusters,
            metric=self.metric,
            implementation=self.implementation,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            device=device,
            n_init=self.n_init
        )
        torch_model.fit(X)
        
        # Sync codebooks back to Numpy/CPU
        self.D_ = torch_model.D_
        self.codebooks_ = [cb.cpu().numpy() for cb in torch_model.codebooks_]

    def encode(self, X: ArrayLike, *, device: str = "cpu", batch_size: Optional[int] = None) -> np.ndarray:
        if not self.codebooks_:
            raise RuntimeError("Model is not fitted yet.")
            
        if device == "cpu":
            return self._encode_numpy(X, batch_size)
        else:
            return self._encode_torch(X, device, batch_size)

    def _encode_numpy(self, X: ArrayLike, batch_size: Optional[int] = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]
        
        if batch_size is None:
            batch_size = N
            
        codes = np.zeros((N, self.n_levels), dtype=np.int32)
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_X = X[start_idx:end_idx] # (B, D)
            
            residuals = batch_X.copy()
            
            for l in range(self.n_levels):
                codebook = self.codebooks_[l] # (K_l, D)
                
                # Find nearest centroid for each residual
                # We use euclidean distance
                dists = euclidean_distances(residuals, codebook, squared=True)
                batch_codes = np.argmin(dists, axis=1) # (B,)
                
                codes[start_idx:end_idx, l] = batch_codes
                
                # Update residuals
                residuals = residuals - codebook[batch_codes]
                
        return codes

    def _encode_torch(self, X: ArrayLike, device: str, batch_size: Optional[int] = None) -> np.ndarray:
        # Reconstruct torch model (lightweight wrapper)
        torch_model = RQKMeansTorch(
            n_levels=self.n_levels,
            n_clusters=self.n_clusters,
            metric=self.metric,
            implementation=self.implementation,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            device=device,
            n_init=self.n_init
        )
        # Load codebooks into torch model
        torch_model.codebooks_ = [torch.from_numpy(cb).to(device) for cb in self.codebooks_]
        
        return torch_model.encode(X, batch_size=batch_size)

    def semantic_id(self, codes: np.ndarray, *, sep: str = "-") -> List[str]:
        # codes: (N, L)
        result = []
        for i in range(codes.shape[0]):
            row_codes = codes[i]
            # Join codes as string
            sid = sep.join(map(str, row_codes))
            result.append(sid)
        return result

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if not self.codebooks_:
            raise RuntimeError("Model is not fitted yet.")
            
        N, L = codes.shape
        # Ensure codes are within bounds
        # assert np.all(codes < self.n_clusters)
        
        vectors_approx = np.zeros((N, self.D_), dtype=np.float32)
        
        for l in range(L):
            codebook = self.codebooks_[l]
            level_codes = codes[:, l]
            vectors_approx += codebook[level_codes]
            
        return vectors_approx

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        
        metadata = {
            "type": "RQKMeans",
            "n_levels": self.n_levels,
            "n_clusters": self.n_clusters, # Can be list
            "metric": self.metric,
            "implementation": self.implementation,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "D": self.D_
        }
        
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Save codebooks as separate arrays
        codebook_dict = {f"codebook_{i}": cb for i, cb in enumerate(self.codebooks_)}
        np.savez_compressed(
            os.path.join(path, "codebooks.npz"),
            **codebook_dict
        )
        
    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> "RQKMeans":
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
        if metadata["type"] != "RQKMeans":
            raise ValueError(f"Invalid model type: {metadata['type']}")
            
        instance = cls(
            n_levels=metadata["n_levels"],
            n_clusters=metadata["n_clusters"],
            metric=metadata["metric"],
            implementation=metadata.get("implementation", "kmeans"),
            max_iter=metadata["max_iter"],
            tol=metadata["tol"],
            random_state=metadata["random_state"],
            n_init=metadata.get("n_init", None)
        )
        instance.D_ = metadata["D"]
        
        # Load codebooks
        data = np.load(os.path.join(path, "codebooks.npz"))
        instance.codebooks_ = []
        for i in range(instance.n_levels):
            instance.codebooks_.append(data[f"codebook_{i}"])
        
        return instance
