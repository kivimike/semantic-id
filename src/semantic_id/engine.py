import json
import os
import shutil
from typing import List, Optional, Type

import numpy as np

from semantic_id.core import ArrayLike, BaseSemanticEncoder
from semantic_id.uniqueness.resolver import (
    BaseResolver,
    SinkhornResolver,
    UniqueIdResolver,
)
from semantic_id.uniqueness.stores import (
    CollisionStore,
    InMemoryCollisionStore,
    SQLiteCollisionStore,
)

# Registry of encoder types for deserialization
_ENCODER_REGISTRY = {}


def _ensure_registry():
    """Lazily populate the encoder registry to avoid circular imports."""
    if not _ENCODER_REGISTRY:
        from semantic_id.algorithms.rq_kmeans import RQKMeans
        from semantic_id.algorithms.rq_vae import RQVAE

        _ENCODER_REGISTRY["RQKMeans"] = RQKMeans
        _ENCODER_REGISTRY["RQVAE"] = RQVAE


class SemanticIdEngine:
    """
    High-level engine for generating unique semantic IDs.
    Combines an encoder algorithm and a uniqueness resolver.
    """

    def __init__(
        self,
        encoder: BaseSemanticEncoder,
        unique_resolver: Optional[BaseResolver] = None,
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

    def unique_ids(
        self, X: ArrayLike, *, device: str = "cpu", batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Generate unique semantic IDs for the input embeddings.

        1. Encode X -> codes
        2. Convert codes -> semantic_ids (raw)
        3. Resolve collisions -> unique_ids
        """
        import torch

        # 1. Encode
        codes = self.encoder.encode(X, device=device, batch_size=batch_size)

        # 2. Semantic IDs
        sids = self.encoder.semantic_id(codes)

        # 3. Unique IDs -- pass extra context for SinkhornResolver
        resolver_kwargs = {}
        if isinstance(self.unique_resolver, SinkhornResolver):
            # SinkhornResolver needs the raw embeddings and model on device
            if isinstance(X, np.ndarray):
                emb_tensor = torch.from_numpy(X).float().to(device)
            else:
                emb_tensor = torch.tensor(X, dtype=torch.float).to(device)

            # Access the underlying module (RQVAE exposes .module)
            model = getattr(self.encoder, "module", None)
            if model is not None:
                model.to(device)

            resolver_kwargs["embeddings"] = emb_tensor
            resolver_kwargs["model"] = model
            resolver_kwargs["device"] = device

        uids = self.unique_resolver.assign(sids, **resolver_kwargs)

        return uids

    def save(self, path: str) -> None:
        """
        Save engine artifacts.

        Structure::

            path/
              engine_meta.json   # encoder type, resolver config
              encoder/           # encoder-specific artifacts
              resolver/
                collisions.db    # if SQLite store
        """
        os.makedirs(path, exist_ok=True)

        # Save encoder
        encoder_path = os.path.join(path, "encoder")
        self.encoder.save(encoder_path)

        # Save resolver state
        resolver_path = os.path.join(path, "resolver")
        os.makedirs(resolver_path, exist_ok=True)

        store = self.unique_resolver.store
        store_type = "in_memory"
        store_config = {}

        if isinstance(store, SQLiteCollisionStore):
            store_type = "sqlite"
            # Copy the SQLite DB file into the save directory
            dest_db = os.path.join(resolver_path, "collisions.db")
            if os.path.exists(store.db_path):
                shutil.copy2(store.db_path, dest_db)
            store_config["db_path"] = "collisions.db"  # relative path

        # Detect encoder type from metadata if possible
        encoder_type = type(self.encoder).__name__

        engine_meta = {
            "encoder_type": encoder_type,
            "store_type": store_type,
            "store_config": store_config,
        }

        with open(os.path.join(path, "engine_meta.json"), "w") as f:
            json.dump(engine_meta, f, indent=2)

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> "SemanticIdEngine":
        """
        Load a saved engine from disk.

        Args:
            path: Directory where engine was saved.
            device: Device to load the encoder onto.

        Returns:
            Loaded SemanticIdEngine instance.
        """
        _ensure_registry()

        with open(os.path.join(path, "engine_meta.json"), "r") as f:
            engine_meta = json.load(f)

        # Load encoder
        encoder_type_name = engine_meta["encoder_type"]
        encoder_cls = _ENCODER_REGISTRY.get(encoder_type_name)
        if encoder_cls is None:
            raise ValueError(
                f"Unknown encoder type '{encoder_type_name}'. "
                f"Registered types: {list(_ENCODER_REGISTRY.keys())}"
            )

        encoder_path = os.path.join(path, "encoder")
        encoder = encoder_cls.load(encoder_path, device=device)

        # Load resolver
        store_type = engine_meta.get("store_type", "in_memory")
        store_config = engine_meta.get("store_config", {})

        if store_type == "sqlite":
            resolver_path = os.path.join(path, "resolver")
            db_filename = store_config.get("db_path", "collisions.db")
            db_path = os.path.join(resolver_path, db_filename)
            store: CollisionStore = SQLiteCollisionStore(db_path)
        else:
            store = InMemoryCollisionStore()

        resolver = UniqueIdResolver(store=store)

        return cls(encoder=encoder, unique_resolver=resolver)
