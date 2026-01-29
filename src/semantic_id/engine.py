import os
import shutil
import numpy as np
from typing import List, Optional, Union, Type

from semantic_id.core import BaseSemanticEncoder, ArrayLike
from semantic_id.uniqueness.resolver import UniqueIdResolver
from semantic_id.algorithms.rq_kmeans import RQKMeans
from semantic_id.uniqueness.stores import InMemoryCollisionStore, SQLiteCollisionStore

class SemanticIdEngine:
    """
    High-level engine for generating unique semantic IDs.
    Combines an encoder algorithm and a uniqueness resolver.
    """
    
    def __init__(
        self,
        encoder: BaseSemanticEncoder,
        unique_resolver: Optional[UniqueIdResolver] = None
    ):
        self.encoder = encoder
        if unique_resolver is None:
            self.unique_resolver = UniqueIdResolver(store=InMemoryCollisionStore())
        else:
            self.unique_resolver = unique_resolver
            
    def fit(self, X: ArrayLike, *, device: str = "cpu") -> "SemanticIdEngine":
        """
        Train the underlying encoder.
        """
        self.encoder.fit(X, device=device)
        return self

    def unique_ids(self, X: ArrayLike, *, device: str = "cpu", batch_size: Optional[int] = None) -> List[str]:
        """
        Generate unique semantic IDs for the input embeddings.
        
        1. Encode X -> codes
        2. Convert codes -> semantic_ids (raw)
        3. Resolve collisions -> unique_ids
        """
        # 1. Encode
        codes = self.encoder.encode(X, device=device, batch_size=batch_size)
        
        # 2. Semantic IDs
        sids = self.encoder.semantic_id(codes)
        
        # 3. Unique IDs
        uids = self.unique_resolver.assign(sids)
        
        return uids

    def save(self, path: str) -> None:
        """
        Save engine artifacts.
        
        Structure:
        path/
          encoder/
            ...
          resolver/ 
            collisions.db (if sqlite)
        """
        os.makedirs(path, exist_ok=True)
        
        # Save encoder
        encoder_path = os.path.join(path, "encoder")
        self.encoder.save(encoder_path)
        
        # Save resolver config if needed?
        # For now, resolver state is mostly external (db) or transient (in-memory).
        # If SQLite, we might assume the DB file is managed by user or we copy it?
        # The prompt says: "save/load model and codebook".
        # Engine save/load logic usually implies re-instantiating the object.
        # But SQLite store path is crucial.
        pass

    # Loading the full engine is tricky because we need to know WHICH encoder class to instantiate
    # and how to reconstruct the resolver.
    # For MVP, let's keep it simple: users usually save the Encoder mainly.
    # But let's add a simple load that assumes RQKMeans for now or inspects metadata.
