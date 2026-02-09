import collections
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from semantic_id.uniqueness.stores import CollisionStore, InMemoryCollisionStore


class BaseResolver(ABC):
    """Abstract base class for collision resolvers."""
    
    @abstractmethod
    def assign(self, semantic_ids: List[str], **kwargs) -> List[str]:
        """
        Resolve collisions in a list of semantic IDs.
        
        Args:
            semantic_ids: List of raw semantic ID strings.
            
        Returns:
            List of unique IDs.
        """
        pass


class UniqueIdResolver(BaseResolver):
    """
    Resolves collisions by appending suffixes to semantic IDs.
    
    First occurrence: no suffix.
    Second occurrence: ``-1``.
    Third occurrence: ``-2``, etc.
    """
    
    def __init__(self, store: Optional[CollisionStore] = None):
        if store is None:
            self.store = InMemoryCollisionStore()
        else:
            self.store = store
            
    def assign(self, semantic_ids: List[str], **kwargs) -> List[str]:
        """
        Assign unique IDs for a list of semantic IDs.
        
        Args:
            semantic_ids: List of raw semantic ID strings.
            
        Returns:
            List of unique IDs (potentially with suffixes).
        """
        unique_ids = []
        for sid in semantic_ids:
            idx = self.store.next_suffix(sid)
            
            if idx == 0:
                unique_ids.append(sid)
            else:
                unique_ids.append(f"{sid}-{idx}")
                
        return unique_ids


class SinkhornResolver(BaseResolver):
    """
    Resolves collisions by re-encoding colliding items with Sinkhorn-Knopp
    balanced assignment enabled on the last VQ layer.
    
    This is based on the approach in the reference ``generate_indices.py`` from
    the MiniOneRec project. Instead of appending suffixes, it iteratively 
    re-encodes colliding items so they get different codes from the quantizer itself,
    producing more semantically meaningful unique IDs.
    
    Requires an RQVAE model with accessible VQ layers.
    
    Args:
        max_iterations: Maximum number of Sinkhorn re-encoding iterations.
        sk_epsilon: Sinkhorn epsilon for the last VQ layer during re-encoding.
            If None, uses 0.003 as default.
        fallback_suffix: If True, falls back to suffix-based resolution for any
            remaining collisions after max_iterations. If False, returns results
            with remaining collisions unresolved.
    """
    
    def __init__(
        self,
        max_iterations: int = 20,
        sk_epsilon: Optional[float] = None,
        fallback_suffix: bool = True,
    ):
        self.max_iterations = max_iterations
        self.sk_epsilon = sk_epsilon if sk_epsilon is not None else 0.003
        self.fallback_suffix = fallback_suffix
    
    def assign(self, semantic_ids: List[str], **kwargs) -> List[str]:
        """
        Resolve collisions using Sinkhorn re-encoding.
        
        Requires ``embeddings``, ``model``, and ``device`` to be passed as kwargs.
        
        Args:
            semantic_ids: List of raw semantic ID strings.
            embeddings: Original embedding tensor (N, D) on the target device.
            model: RQVAEModule instance (must have ``rq.vq_layers`` and ``get_indices``).
            device: Device string.
            sep: Separator used in semantic IDs.
            
        Returns:
            List of unique IDs.
        """
        import torch
        
        embeddings = kwargs.get("embeddings")
        model = kwargs.get("model")
        device = kwargs.get("device", "cpu")
        sep = kwargs.get("sep", "-")
        
        if embeddings is None or model is None:
            raise ValueError(
                "SinkhornResolver.assign() requires 'embeddings' and 'model' kwargs. "
                "Use SemanticIdEngine with an RQVAE encoder for Sinkhorn resolution."
            )
        
        N = len(semantic_ids)
        
        # Convert to mutable arrays for in-place updates
        all_ids = list(semantic_ids)
        all_ids_str = np.array([str(s) for s in all_ids])
        
        # Save original sk_epsilon values and disable Sinkhorn on all but last layer
        original_epsilons = []
        for vq in model.rq.vq_layers:
            original_epsilons.append(vq.sk_epsilon)
        
        # Disable Sinkhorn on all layers except last
        for vq in model.rq.vq_layers[:-1]:
            vq.sk_epsilon = 0.0
        
        # Enable Sinkhorn on the last layer
        last_vq = model.rq.vq_layers[-1]
        if last_vq.sk_epsilon <= 0:
            last_vq.sk_epsilon = self.sk_epsilon
        
        model.eval()
        
        for iteration in range(self.max_iterations):
            # Find collision groups
            collision_groups = self._find_collision_groups(all_ids_str)
            
            if not collision_groups:
                break  # No more collisions
            
            # Re-encode each collision group with Sinkhorn
            with torch.no_grad():
                for group_indices in collision_groups:
                    group_emb = embeddings[group_indices]
                    
                    # Re-encode with Sinkhorn enabled
                    new_indices = model.get_indices(group_emb, use_sk=True)
                    new_indices_np = new_indices.cpu().numpy()
                    
                    # Update IDs
                    for local_idx, global_idx in enumerate(group_indices):
                        new_code = new_indices_np[local_idx]
                        new_sid = sep.join(map(str, new_code))
                        all_ids[global_idx] = new_sid
                        all_ids_str[global_idx] = new_sid
        
        # Restore original sk_epsilon values
        for vq, eps in zip(model.rq.vq_layers, original_epsilons):
            vq.sk_epsilon = eps
        
        # Fallback: if still collisions, use suffix resolution
        if self.fallback_suffix:
            suffix_store = InMemoryCollisionStore()
            final_ids = []
            for sid in all_ids:
                idx = suffix_store.next_suffix(sid)
                if idx == 0:
                    final_ids.append(sid)
                else:
                    final_ids.append(f"{sid}-{idx}")
            return final_ids
        
        return all_ids
    
    @staticmethod
    def _find_collision_groups(ids_array: np.ndarray) -> List[List[int]]:
        """Find groups of indices that share the same ID string."""
        index_map: dict = {}
        for i, sid in enumerate(ids_array):
            if sid not in index_map:
                index_map[sid] = []
            index_map[sid].append(i)
        
        return [indices for indices in index_map.values() if len(indices) > 1]
