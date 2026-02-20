from semantic_id.algorithms.rq_kmeans import RQKMeans
from semantic_id.algorithms.rq_vae import RQVAE
from semantic_id.engine import SemanticIdEngine
from semantic_id.exceptions import NotFittedError, SemanticIdError, ShapeMismatchError
from semantic_id.uniqueness.resolver import SinkhornResolver, UniqueIdResolver
from semantic_id.uniqueness.stores import InMemoryCollisionStore, SQLiteCollisionStore
from semantic_id.utils.metrics import (
    evaluate,
    find_similar,
    hierarchical_distance,
    ndcg_at_k,
)

__version__ = "0.2.5"

__all__ = [
    "__version__",
    "RQKMeans",
    "RQVAE",
    "SemanticIdEngine",
    "UniqueIdResolver",
    "SinkhornResolver",
    "InMemoryCollisionStore",
    "SQLiteCollisionStore",
    "SemanticIdError",
    "NotFittedError",
    "ShapeMismatchError",
    "evaluate",
    "find_similar",
    "hierarchical_distance",
    "ndcg_at_k",
]
